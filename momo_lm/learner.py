from __future__ import annotations

import contextlib
import html
import re
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import deque
from html.parser import HTMLParser

from .knowledge import KnowledgeStore
from .model import NeuralTextModel


class _ReadableHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.links: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self._ignored += 1
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "noscript"} and self._ignored:
            self._ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored and data.strip():
            self.parts.append(data.strip())


class Learner:
    USER_AGENT = "Momo-LM/0.1 (+https://github.com/YanagiKH/Momo-LM)"

    def __init__(self, model: NeuralTextModel, store: KnowledgeStore) -> None:
        self.model = model
        self.store = store

    @staticmethod
    def _chunks(text: str, size: int = 1200, overlap: int = 120) -> list[str]:
        clean = re.sub(r"\s+", " ", text).strip()
        if not clean:
            return []
        chunks = []
        start = 0
        while start < len(clean):
            chunks.append(clean[start : start + size])
            start += max(1, size - overlap)
        return chunks

    def ingest_text(
        self,
        text: str,
        source: str = "manual",
        *,
        train: bool = True,
        epochs: int = 1,
        learning_rate: float = 0.02,
    ) -> dict[str, object]:
        chunks = self._chunks(text)
        for chunk in chunks:
            self.store.add_document(source, chunk)
        losses = self.model.train_text(text, epochs=epochs, learning_rate=learning_rate) if train else []
        return {
            "source": source,
            "characters": len(text),
            "chunks": len(chunks),
            "trained": train,
            "loss": losses[-1] if losses else None,
        }

    def crawl(
        self,
        start_url: str,
        *,
        max_pages: int = 8,
        timeout: float = 10.0,
        train: bool = True,
    ) -> dict[str, object]:
        parsed_start = urllib.parse.urlparse(start_url)
        if parsed_start.scheme not in {"http", "https"} or not parsed_start.netloc:
            raise ValueError("Only absolute HTTP/HTTPS URLs are supported")
        origin = f"{parsed_start.scheme}://{parsed_start.netloc}"
        robots = urllib.robotparser.RobotFileParser(urllib.parse.urljoin(origin, "/robots.txt"))
        with contextlib.suppress(OSError):
            robots.read()
        queue = deque([start_url])
        visited: set[str] = set()
        results = []
        while queue and len(visited) < max(1, min(max_pages, 50)):
            url = queue.popleft().split("#", 1)[0]
            if url in visited or not robots.can_fetch(self.USER_AGENT, url):
                continue
            visited.add(url)
            request = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    content_type = response.headers.get_content_type()
                    if content_type != "text/html":
                        continue
                    charset = response.headers.get_content_charset() or "utf-8"
                    raw = response.read(2_000_000).decode(charset, errors="replace")
            except (OSError, ValueError) as exc:
                results.append({"url": url, "error": str(exc)})
                continue
            parser = _ReadableHTML()
            parser.feed(raw)
            text = html.unescape(" ".join(parser.parts))
            result = self.ingest_text(text, url, train=train, epochs=1)
            results.append(result)
            for link in parser.links:
                absolute = urllib.parse.urljoin(url, link).split("#", 1)[0]
                parsed = urllib.parse.urlparse(absolute)
                if parsed.scheme in {"http", "https"} and parsed.netloc == parsed_start.netloc:
                    queue.append(absolute)
        return {"start_url": start_url, "pages": results, "visited": len(visited)}
