from __future__ import annotations

import hmac
import json
import mimetypes
import re
import secrets
import socket
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from .agent_store import redact
from .config import validate_access_token
from .paths import package_root
from .runtime import MomoRuntime

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
WILDCARD_HOSTS = {"0.0.0.0", "::"}
AGENT_PATH = re.compile(r"^/api/agents/([0-9a-f]{32})$")
AGENT_ACTION_PATH = re.compile(
    r"^/api/agents/([0-9a-f]{32})/(cancel|approve|events)$"
)


class ThreadingHTTPServerV6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


class MomoServer:
    def __init__(
        self,
        runtime: MomoRuntime,
        host: str | None = None,
        port: int | None = None,
        *,
        access_token: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.host = host if host is not None else runtime.config.host
        self.port = port if port is not None else runtime.config.port
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 0 <= self.port <= 65_535:
            raise ValueError("HTTP port must be an integer between 0 and 65535")
        normalized_host = self.host.lower().strip("[]")
        if (
            not normalized_host
            or any(ord(character) < 33 or ord(character) > 126 for character in normalized_host)
            or "/" in normalized_host
            or "@" in normalized_host
            or ":" in normalized_host
            and normalized_host not in {"::", "::1"}
        ):
            raise ValueError("Invalid HTTP bind host")
        self.host = normalized_host
        configured_token = access_token if access_token is not None else runtime.config.access_token
        if configured_token is None and normalized_host not in LOOPBACK_HOSTS:
            raise ValueError("A persistent access token is required for non-loopback binding")
        self.access_token = validate_access_token(configured_token or secrets.token_urlsafe(32))
        self.allowed_hosts = {host.lower().strip("[]") for host in runtime.config.allowed_hosts}
        if normalized_host not in WILDCARD_HOSTS:
            self.allowed_hosts.add(normalized_host)
        server_class = ThreadingHTTPServerV6 if ":" in self.host else ThreadingHTTPServer
        self.httpd = server_class((self.host, self.port), self._handler())

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        runtime = self.runtime
        web_root = package_root() / "web"
        access_token = self.access_token
        allowed_hosts = frozenset(self.allowed_hosts)

        class Handler(BaseHTTPRequestHandler):
            server_version = "Momo-LM/0.3.0"

            def log_message(self, format_string: str, *args: object) -> None:
                rendered = format_string % args
                safe = str(redact(rendered))
                safe = re.sub(r"([?&](?:token|access_token)=)[^\s&]+", r"\1[REDACTED]", safe)
                print(f"[web] {self.client_address[0]} {safe}")

            def _security_headers(self) -> None:
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")

            def _json(self, payload: object, status: int = HTTPStatus.OK) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self._security_headers()
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, object]:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Transfer-Encoding is not supported")
                lengths = self.headers.get_all("Content-Length", failobj=[])
                if len(lengths) > 1:
                    raise ValueError("Multiple Content-Length headers are not supported")
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise ValueError("Invalid Content-Length") from exc
                if length < 0 or length > 2_500_000:
                    raise ValueError("Request body exceeds 2.5 MB")
                raw = self.rfile.read(length)
                payload = json.loads(raw or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                return payload

            @staticmethod
            def _integer(value: object, name: str) -> int:
                if type(value) is not int:
                    raise ValueError(f"{name} must be an integer")
                return value

            def _file(self, path: Path, *, root: Path) -> None:
                try:
                    resolved = path.resolve(strict=True)
                    resolved.relative_to(root.resolve(strict=True))
                except (OSError, ValueError):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                content = resolved.read_bytes()
                mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(content)))
                self._security_headers()
                if resolved.suffix in {".html", ".js", ".css"}:
                    self.send_header(
                        "Content-Security-Policy",
                        "default-src 'self'; img-src 'self' blob:; media-src 'self' blob:; "
                        "style-src 'self'; script-src 'self'; connect-src 'self'; "
                        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
                    )
                self.end_headers()
                self.wfile.write(content)

            @staticmethod
            def _parse_authority(raw: str) -> tuple[str, int | None]:
                if not raw or any(ord(character) < 33 or ord(character) > 126 for character in raw):
                    raise ValueError("Invalid Host header")
                parsed = urlparse(f"//{raw}")
                if (
                    parsed.username is not None
                    or parsed.password is not None
                    or parsed.hostname is None
                    or parsed.path
                    or parsed.params
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError("Invalid Host header")
                try:
                    port = parsed.port
                except ValueError as exc:
                    raise ValueError("Invalid Host port") from exc
                return parsed.hostname.lower().strip("[]"), port

            def _valid_host(self) -> bool:
                if len(self.headers.get_all("Host", failobj=[])) != 1:
                    return False
                try:
                    hostname, port = self._parse_authority(self.headers.get("Host", ""))
                except ValueError:
                    return False
                return hostname in allowed_hosts and (
                    port is None or port == self.server.server_port
                )

            def _valid_origin(self) -> bool:
                if len(self.headers.get_all("Origin", failobj=[])) > 1:
                    return False
                raw = self.headers.get("Origin")
                if raw is None:
                    return True
                if any(ord(character) < 33 or ord(character) > 126 for character in raw):
                    return False
                parsed = urlparse(raw)
                if (
                    parsed.scheme != "http"
                    or parsed.hostname is None
                    or parsed.username is not None
                    or parsed.password is not None
                    or parsed.path not in {"", "/"}
                    or parsed.params
                    or parsed.query
                    or parsed.fragment
                ):
                    return False
                try:
                    port = parsed.port or 80
                except ValueError:
                    return False
                return (
                    parsed.hostname.lower().strip("[]") in allowed_hosts
                    and port == self.server.server_port
                )

            def _authorized(self) -> bool:
                if len(self.headers.get_all("X-Momo-Token", failobj=[])) != 1:
                    return False
                supplied = self.headers.get("X-Momo-Token", "")
                try:
                    validate_access_token(supplied)
                except ValueError:
                    return False
                return hmac.compare_digest(supplied, access_token)

            def _guard(self, *, api: bool, state_change: bool = False) -> bool:
                if not self._valid_host():
                    self._json({"error": "Forbidden host"}, HTTPStatus.FORBIDDEN)
                    return False
                if state_change and not self._valid_origin():
                    self._json({"error": "Forbidden origin"}, HTTPStatus.FORBIDDEN)
                    return False
                if api and not self._authorized():
                    self._json({"error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED)
                    return False
                return True

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                is_api = path.startswith("/api/")
                is_private_asset = path.startswith(("/generated/", "/speech/"))
                if not self._guard(api=is_api or is_private_asset):
                    return
                try:
                    if path == "/api/health":
                        self._json({"status": "ok"})
                    elif path == "/api/status":
                        self._json(runtime.status())
                    elif path == "/api/agents":
                        query = parse_qs(urlparse(self.path).query)
                        limit = int(query.get("limit", ["100"])[0])
                        self._json({"agents": runtime.list_agents(limit=limit)})
                    elif match := AGENT_PATH.fullmatch(path):
                        self._json({"agent": runtime.get_agent(match.group(1))})
                    elif (match := AGENT_ACTION_PATH.fullmatch(path)) and match.group(2) == "events":
                        query = parse_qs(urlparse(self.path).query)
                        after = int(query.get("after", ["0"])[0])
                        limit = int(query.get("limit", ["100"])[0])
                        self._json(
                            {"events": runtime.agent_events(match.group(1), after=after, limit=limit)}
                        )
                    elif path.startswith("/generated/"):
                        name = Path(unquote(path.removeprefix("/generated/"))).name
                        self._file(
                            runtime.config.home / "generated" / name,
                            root=runtime.config.home / "generated",
                        )
                    elif path.startswith("/speech/"):
                        name = Path(unquote(path.removeprefix("/speech/"))).name
                        self._file(
                            runtime.config.home / "speech" / name,
                            root=runtime.config.home / "speech",
                        )
                    elif is_api:
                        self._json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
                    else:
                        target = "index.html" if path in {"", "/"} else Path(unquote(path)).name
                        self._file(web_root / target, root=web_root)
                except KeyError as exc:
                    self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
                except (TypeError, ValueError) as exc:
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except Exception as exc:
                    self.log_error("Operation failed: %s", type(exc).__name__)
                    self._json({"error": "Operation failed"}, HTTPStatus.INTERNAL_SERVER_ERROR)

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if not self._guard(api=path.startswith("/api/"), state_change=True):
                    return
                try:
                    data = self._read_json()
                    if path == "/api/chat":
                        self._json(runtime.chat(str(data.get("message", "")), learn=data.get("learn")))
                    elif path == "/api/train":
                        self._json(
                            runtime.train(
                                str(data.get("text", "")),
                                epochs=int(data.get("epochs", 3)),
                                learning_rate=float(
                                    data.get("learning_rate", runtime.config.learning_rate)
                                ),
                                source=str(data.get("source", "web-ui")),
                            )
                        )
                    elif path == "/api/ingest":
                        self._json(
                            runtime.ingest(
                                str(data.get("text", "")),
                                source=str(data.get("source", "web-ui")),
                                train=bool(data.get("train", False)),
                            )
                        )
                    elif path == "/api/crawl":
                        self._json(
                            runtime.crawl(
                                str(data.get("url", "")),
                                max_pages=int(
                                    data.get("max_pages", runtime.config.max_crawl_pages)
                                ),
                                train=bool(data.get("train", False)),
                            )
                        )
                    elif path == "/api/image":
                        filename = f"momo-{secrets.token_hex(8)}.png"
                        output = runtime.config.home / "generated" / filename
                        style = str(data.get("style", "illustration"))
                        quality = str(data.get("quality", "standard"))
                        steps = (
                            self._integer(data["steps"], "steps")
                            if data.get("steps") not in (None, "")
                            else None
                        )
                        runtime.generate_image(
                            str(data.get("prompt", "")),
                            output,
                            width=self._integer(data.get("width", 512), "width"),
                            height=self._integer(data.get("height", 512), "height"),
                            seed=self._integer(data["seed"], "seed")
                            if data.get("seed") not in (None, "")
                            else None,
                            style=style,
                            negative_prompt=str(data.get("negative_prompt", "")),
                            quality=quality,
                            steps=steps,
                        )
                        resolved_steps = steps or runtime.image_model.QUALITY_STEPS[quality]
                        self._json(
                            {
                                "url": f"/generated/{filename}",
                                "path": str(output),
                                "style": style,
                                "quality": quality,
                                "steps": resolved_steps,
                            }
                        )
                    elif path == "/api/tts":
                        filename = f"momo-{secrets.token_hex(8)}.wav"
                        output = runtime.config.home / "speech" / filename
                        result = runtime.speech.synthesize(
                            str(data.get("text", "")), output, rate=int(data.get("rate", 170))
                        )
                        result["url"] = f"/speech/{filename}"
                        self._json(result)
                    elif path == "/api/mods/reload":
                        self._json(runtime.reload_mods())
                    elif path == "/api/agents":
                        capabilities = data.get("capabilities")
                        budgets = data.get("budgets")
                        if capabilities is not None and not isinstance(capabilities, list):
                            raise ValueError("capabilities must be a list")
                        if budgets is not None and not isinstance(budgets, dict):
                            raise ValueError("budgets must be an object")
                        agent = runtime.create_agent(
                            str(data.get("goal", "")),
                            profile=str(data.get("profile", "copilot")),
                            capabilities=capabilities,
                            budgets=budgets,
                            background=True,
                        )
                        self._json({"agent": agent}, HTTPStatus.CREATED)
                    elif match := AGENT_ACTION_PATH.fullmatch(path):
                        agent_id, action = match.groups()
                        if action == "cancel":
                            self._json({"agent": runtime.cancel_agent(agent_id)})
                        elif action == "approve":
                            approval_id = data.get("approval_id")
                            if not isinstance(approval_id, str) or not approval_id:
                                raise ValueError("approval_id is required")
                            self._json(
                                {
                                    "agent": runtime.approve_agent(
                                        agent_id, approval_id, background=True
                                    )
                                }
                            )
                        else:
                            self._json(
                                {"error": "Method not allowed"},
                                HTTPStatus.METHOD_NOT_ALLOWED,
                            )
                    else:
                        self._json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
                except KeyError as exc:
                    self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except PermissionError as exc:
                    self._json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
                except Exception as exc:  # keep the workbench alive without exposing internals
                    self.log_error("Operation failed: %s", type(exc).__name__)
                    self._json({"error": "Operation failed"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        return Handler

    @property
    def url(self) -> str:
        if self.host == "0.0.0.0":
            browser_host = "127.0.0.1"
        elif self.host == "::":
            browser_host = "::1"
        else:
            browser_host = self.host
        if ":" in browser_host and not browser_host.startswith("["):
            browser_host = f"[{browser_host}]"
        return f"http://{browser_host}:{self.httpd.server_port}"

    @property
    def browser_url(self) -> str:
        return f"{self.url}/#token={quote(self.access_token, safe='')}"

    def serve(self, *, open_browser: bool = True) -> None:
        print(f"Momo-LM is running at {self.url}")
        if open_browser:
            threading.Timer(0.8, webbrowser.open, args=(self.browser_url,)).start()
        else:
            print(f"Browser session URL: {self.browser_url}")
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.httpd.server_close()
