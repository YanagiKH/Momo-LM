import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from momo_lm.config import MomoConfig
from momo_lm.runtime import MomoRuntime
from momo_lm.server import MomoServer


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.runtime = MomoRuntime(MomoConfig.defaults(Path(self.temporary.name)))
        self.server = MomoServer(self.runtime, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.httpd.shutdown()
        self.server.httpd.server_close()
        self.thread.join(timeout=3)
        self.runtime.close()
        self.temporary.cleanup()

    def test_health_status_and_chat_endpoints(self) -> None:
        with urllib.request.urlopen(f"{self.server.url}/api/health", timeout=3) as response:
            self.assertEqual(json.load(response)["status"], "ok")
        request = urllib.request.Request(
            f"{self.server.url}/api/chat",
            data=json.dumps({"message": "Hello", "learn": False}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertIn("Momo", json.load(response)["response"])


if __name__ == "__main__":
    unittest.main()
