import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from momo_lm.config import MomoConfig, validate_access_token, validate_training_rate


class ConfigTests(unittest.TestCase):
    def test_agent_paths_and_security_settings_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MomoConfig.defaults(root)
            config.access_token = "visible-token_123"
            config.allowed_hosts.append("momo.local")
            saved = config.save()
            loaded = MomoConfig.load(saved)
            self.assertEqual(loaded.agent_database_path, root / "data" / "agents.db")
            self.assertEqual(loaded.agent_workspace_path, root / "agent-workspace")
            self.assertEqual(loaded.access_token, "visible-token_123")
            self.assertIn("momo.local", loaded.allowed_hosts)
            self.assertEqual(json.loads(saved.read_text(encoding="utf-8"))["access_token"], "visible-token_123")
            if os.name != "nt":
                self.assertEqual(saved.stat().st_mode & 0o777, 0o600)

    def test_environment_access_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"MOMO_ACCESS_TOKEN": "environment-token"}
        ):
            self.assertEqual(
                MomoConfig.defaults(Path(directory)).access_token, "environment-token"
            )

    def test_environment_token_overrides_saved_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MomoConfig.defaults(root)
            config.access_token = "saved-token"
            path = config.save()
            with patch.dict(os.environ, {"MOMO_ACCESS_TOKEN": "runtime-token"}):
                self.assertEqual(MomoConfig.load(path).access_token, "runtime-token")

    def test_legacy_config_derives_new_agent_paths_from_loaded_home(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured_home = root / "legacy-home"
            path = root / "legacy.json"
            path.write_text(
                json.dumps({"home": str(configured_home), "learning_rate": 0.025}),
                encoding="utf-8",
            )
            loaded = MomoConfig.load(path)
            self.assertEqual(loaded.agent_database_path, configured_home / "data" / "agents.db")
            self.assertEqual(loaded.agent_workspace_path, configured_home / "agent-workspace")
            self.assertEqual(loaded.learning_rate, 0.005)

    def test_adamw_incremental_learning_rate_is_small_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(MomoConfig.defaults(Path(directory)).learning_rate, 0.0005)
        self.assertEqual(validate_training_rate(0.004), 0.004)
        self.assertEqual(validate_training_rate(0.025, clamp=True), 0.005)
        for invalid in (0, -1, 0.00501, float("nan"), float("inf"), True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_training_rate(invalid)

    def test_access_token_is_visible_ascii_between_one_and_1024(self) -> None:
        self.assertEqual(validate_access_token("a"), "a")
        self.assertEqual(validate_access_token("x" * 1024), "x" * 1024)
        for invalid in ("", "has space", "line\nbreak", "密碼", "x" * 1025):
            with self.subTest(invalid=invalid[:20]), self.assertRaises(ValueError):
                validate_access_token(invalid)

    def test_invalid_hosts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = MomoConfig.defaults(Path(directory))
            for invalid in ([], ["host:123"], ["https://host"], ["user@host"]):
                config.allowed_hosts = invalid
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    config.validate()

    def test_invalid_numeric_config_is_reported_as_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = MomoConfig.defaults(Path(directory))
            config.port = None  # type: ignore[assignment]
            with self.assertRaisesRegex(ValueError, "port"):
                config.validate()
            config.port = 7860
            config.agent_approval_ttl_seconds = None  # type: ignore[assignment]
            with self.assertRaisesRegex(ValueError, "approval_ttl"):
                config.validate()


if __name__ == "__main__":
    unittest.main()
