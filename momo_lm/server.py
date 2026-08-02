from __future__ import annotations

import json
import mimetypes
import secrets
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .paths import package_root
from .runtime import MomoRuntime


class MomoServer:
    def __init__(self, runtime: MomoRuntime, host: str | None = None, port: int | None = None) -> None:
        self.runtime = runtime
        self.host = host or runtime.config.host
        self.port = port or runtime.config.port
        self.httpd = ThreadingHTTPServer((self.host, self.port), self._handler())

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        runtime = self.runtime
        web_root = package_root() / "web"

        class Handler(BaseHTTPRequestHandler):
            server_version = "Momo-LM/0.1"

            def log_message(self, format_string: str, *args: object) -> None:
                print(f"[web] {self.address_string()} {format_string % args}")

            def _json(self, payload: object, status: int = HTTPStatus.OK) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, object]:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 2_500_000:
                    raise ValueError("Request body exceeds 2.5 MB")
                raw = self.rfile.read(length)
                payload = json.loads(raw or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                return payload

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
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(content)

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/api/health":
                    self._json({"status": "ok"})
                elif path == "/api/status":
                    self._json(runtime.status())
                elif path.startswith("/generated/"):
                    name = Path(unquote(path.removeprefix("/generated/"))).name
                    self._file(runtime.config.home / "generated" / name, root=runtime.config.home / "generated")
                elif path.startswith("/speech/"):
                    name = Path(unquote(path.removeprefix("/speech/"))).name
                    self._file(runtime.config.home / "speech" / name, root=runtime.config.home / "speech")
                else:
                    target = "index.html" if path in {"", "/"} else Path(unquote(path)).name
                    self._file(web_root / target, root=web_root)

            def do_POST(self) -> None:  # noqa: N802
                try:
                    data = self._read_json()
                    path = urlparse(self.path).path
                    if path == "/api/chat":
                        self._json(runtime.chat(str(data.get("message", "")), learn=data.get("learn")))
                    elif path == "/api/train":
                        self._json(
                            runtime.train(
                                str(data.get("text", "")),
                                epochs=int(data.get("epochs", 3)),
                                learning_rate=float(data.get("learning_rate", runtime.config.learning_rate)),
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
                                max_pages=int(data.get("max_pages", runtime.config.max_crawl_pages)),
                                train=bool(data.get("train", False)),
                            )
                        )
                    elif path == "/api/image":
                        filename = f"momo-{secrets.token_hex(8)}.png"
                        output = runtime.config.home / "generated" / filename
                        runtime.generate_image(
                            str(data.get("prompt", "")),
                            output,
                            width=int(data.get("width", 512)),
                            height=int(data.get("height", 512)),
                            seed=int(data["seed"]) if data.get("seed") not in (None, "") else None,
                        )
                        self._json({"url": f"/generated/{filename}", "path": str(output)})
                    elif path == "/api/tts":
                        filename = f"momo-{secrets.token_hex(8)}.wav"
                        output = runtime.config.home / "speech" / filename
                        result = runtime.speech.synthesize(str(data.get("text", "")), output, rate=int(data.get("rate", 170)))
                        result["url"] = f"/speech/{filename}"
                        self._json(result)
                    elif path == "/api/mods/reload":
                        self._json(runtime.reload_mods())
                    else:
                        self._json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except Exception as exc:  # keep the workbench alive and avoid exposing a traceback
                    self._json({"error": f"Operation failed: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

        return Handler

    @property
    def url(self) -> str:
        browser_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{browser_host}:{self.httpd.server_port}"

    def serve(self, *, open_browser: bool = True) -> None:
        print(f"Momo-LM is running at {self.url}")
        if open_browser:
            threading.Timer(0.8, webbrowser.open, args=(self.url,)).start()
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.httpd.server_close()
