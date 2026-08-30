import json
import tempfile
import threading
import time
import unittest
import urllib.error
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

    def request(
        self,
        path: str,
        *,
        data: dict[str, object] | None = None,
        token: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, object]]:
        request_headers = {"X-Momo-Token": token or self.server.access_token}
        request_headers.update(headers or {})
        raw = None
        method = "GET"
        if data is not None:
            raw = json.dumps(data).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
            method = "POST"
        request = urllib.request.Request(
            f"{self.server.url}{path}", data=raw, headers=request_headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    def test_health_status_and_chat_endpoints(self) -> None:
        status, payload = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")
        status, payload = self.request(
            "/api/chat", data={"message": "Hello", "learn": False}
        )
        self.assertEqual(status, 200)
        self.assertIn("Momo", payload["response"])

    def test_api_requires_token_and_rejects_host_and_origin_confusion(self) -> None:
        request = urllib.request.Request(f"{self.server.url}/api/health")
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(request, timeout=3)
        self.assertEqual(context.exception.code, 401)

        status, payload = self.request(
            "/api/health", headers={"Host": "attacker.example"}
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"], "Forbidden host")

        status, payload = self.request(
            "/api/chat",
            data={"message": "hello", "learn": False},
            headers={"Origin": "http://attacker.example"},
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"], "Forbidden origin")

        status, payload = self.request(
            "/api/train",
            data={"text": "unsafe rate", "epochs": 1, "learning_rate": 0.02},
        )
        self.assertEqual(status, 400)
        self.assertIn("learning_rate", payload["error"])

    def test_non_loopback_requires_explicit_token_and_browser_token_uses_fragment(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            MomoServer(self.runtime, "0.0.0.0", 0)
        with self.assertRaisesRegex(ValueError, "bind host"):
            MomoServer(self.runtime, "bad/host", 0)
        with self.assertRaisesRegex(ValueError, "HTTP port"):
            MomoServer(self.runtime, "127.0.0.1", True)
        self.assertIn("/#token=", self.server.browser_url)
        self.assertNotIn("?token=", self.server.browser_url)

    def test_agent_create_approve_list_cancel_and_events(self) -> None:
        status, payload = self.request(
            "/api/agents",
            data={
                "goal": "train: HTTP agent datum",
                "profile": "training",
                "capabilities": ["model.train"],
            },
        )
        self.assertEqual(status, 201)
        agent_id = payload["agent"]["id"]
        agent: dict[str, object] = payload["agent"]
        for _ in range(100):
            _, detail = self.request(f"/api/agents/{agent_id}")
            agent = detail["agent"]
            if agent["status"] == "waiting_approval":
                break
            time.sleep(0.01)
        self.assertEqual(agent["status"], "waiting_approval")
        approval = agent["pending_approval"]
        self.assertEqual(approval["tool"], "train_text")

        status, _ = self.request(
            f"/api/agents/{agent_id}/approve",
            data={"approval_id": approval["id"]},
        )
        self.assertEqual(status, 200)
        for _ in range(200):
            _, detail = self.request(f"/api/agents/{agent_id}")
            agent = detail["agent"]
            if agent["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        self.assertEqual(agent["status"], "completed")
        reused_status, reused = self.request(
            f"/api/agents/{agent_id}/approve",
            data={"approval_id": approval["id"]},
        )
        self.assertEqual(reused_status, 400)
        self.assertIn("not waiting", reused["error"])

        _, listing = self.request("/api/agents")
        self.assertTrue(any(item["id"] == agent_id for item in listing["agents"]))
        _, events = self.request(f"/api/agents/{agent_id}/events?after=0&limit=100")
        self.assertIn("approval_required", {event["type"] for event in events["events"]})
        self.assertIn("approved", {event["type"] for event in events["events"]})

        _, created = self.request(
            "/api/agents", data={"goal": "write: note.txt\nhello", "profile": "coding", "capabilities": ["workspace.write"]}
        )
        second_id = created["agent"]["id"]
        _, cancelled = self.request(f"/api/agents/{second_id}/cancel", data={})
        self.assertEqual(cancelled["agent"]["status"], "cancelled")

    def test_image_endpoint_exposes_v2_generation_options(self) -> None:
        status, payload = self.request(
            "/api/image",
            data={
                "prompt": "moon portrait",
                "negative_prompt": "watermark",
                "style": "manga",
                "quality": "draft",
                "steps": 1,
                "width": 128,
                "height": 128,
                "seed": 3,
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["style"], "manga")
        self.assertEqual(payload["quality"], "draft")
        self.assertEqual(payload["steps"], 1)
        self.assertTrue(Path(payload["path"]).exists())
        asset_url = f"{self.server.url}{payload['url']}"
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(asset_url, timeout=3)
        self.assertEqual(context.exception.code, 401)
        request = urllib.request.Request(
            asset_url, headers={"X-Momo-Token": self.server.access_token}
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            self.assertEqual(response.read(8), b"\x89PNG\r\n\x1a\n")
        invalid_status, invalid = self.request(
            "/api/image", data={"prompt": "moon", "steps": True}
        )
        self.assertEqual(invalid_status, 400)
        self.assertIn("steps must be an integer", invalid["error"])


if __name__ == "__main__":
    unittest.main()
