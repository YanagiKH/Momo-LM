import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from momo_lm.config import MomoConfig
from momo_lm.runtime import MomoRuntime


class _Site(BaseHTTPRequestHandler):
    def log_message(self, format_string: str, *args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        pages = {
            "/robots.txt": b"User-agent: *\nAllow: /\n",
            "/": b'<html><body>Momo crawl fact alpha <a href="/two">next</a></body></html>',
            "/two": b"<html><body>Momo crawl fact beta</body></html>",
        }
        body = pages.get(self.path, b"not found")
        self.send_response(200 if self.path in pages else 404)
        self.send_header("Content-Type", "text/plain" if self.path == "/robots.txt" else "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class LearnerTests(unittest.TestCase):
    def test_crawler_obeys_scope_and_page_limit(self) -> None:
        site = ThreadingHTTPServer(("127.0.0.1", 0), _Site)
        thread = threading.Thread(target=site.serve_forever, daemon=True)
        thread.start()
        temporary = tempfile.TemporaryDirectory()
        runtime = MomoRuntime(MomoConfig.defaults(Path(temporary.name)))
        try:
            result = runtime.crawl(f"http://127.0.0.1:{site.server_port}/", max_pages=2, train=False)
            self.assertEqual(result["visited"], 2)
            answer = runtime.chat("crawl fact beta", learn=False)
            self.assertIn("beta", answer["response"])
        finally:
            runtime.close()
            temporary.cleanup()
            site.shutdown()
            site.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
