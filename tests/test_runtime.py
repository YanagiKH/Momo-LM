import tempfile
import unittest
from pathlib import Path

from momo_lm.config import MomoConfig
from momo_lm.runtime import MomoRuntime


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        home = Path(self.temporary.name)
        self.runtime = MomoRuntime(MomoConfig.defaults(home))

    def tearDown(self) -> None:
        self.runtime.close()
        self.temporary.cleanup()

    def test_starter_conversation_and_incremental_learning(self) -> None:
        answer = self.runtime.chat("你能做什麼？", learn=False)
        self.assertIn("對話", answer["response"])
        self.assertFalse(answer["learned"])
        result = self.runtime.train("Momo-LM 的測試領域代號是 sakura-42。", epochs=2)
        self.assertEqual(result["chunks"], 1)
        learned = self.runtime.chat("測試領域代號是什麼？", learn=False)
        self.assertIn("sakura-42", learned["response"])

    def test_local_image_generation(self) -> None:
        output = Path(self.temporary.name) / "generated.png"
        self.runtime.generate_image(
            "pink moon",
            output,
            width=160,
            height=128,
            seed=4,
            style="anime",
            negative_prompt="text watermark",
            quality="draft",
            steps=1,
            tile_size=64,
        )
        self.assertTrue(output.exists())
        self.assertGreater(output.stat().st_size, 1000)

    def test_weight_inspection(self) -> None:
        status = self.runtime.status()
        self.assertEqual(status["weights"]["parameters"], 223_835)
        self.assertEqual(status["weights"]["format_version"], 3)
        self.assertIn(status["compute_backend"]["name"], {"numpy", "cpp", "rust"})
        self.assertGreater(status["image_weights"]["parameters"], 1_000)
        self.assertGreaterEqual(status["knowledge"]["documents"], 20)

    def test_offline_speech_always_produces_wav(self) -> None:
        output = Path(self.temporary.name) / "speech.wav"
        result = self.runtime.speech.synthesize("Momo local speech", output)
        self.assertTrue(result["engine"])
        self.assertEqual(output.read_bytes()[:4], b"RIFF")

    def test_agent_profiles_are_persistent_and_read_only_by_default(self) -> None:
        result = self.runtime.create_agent("Prepare a work update", profile="workplace")
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("workspace.write", result["capabilities"])
        self.assertEqual(self.runtime.agent_store.journal_mode, "wal")
        status = self.runtime.status()["agents"]
        self.assertGreaterEqual(status["total"], 1)
        self.assertEqual(
            {profile["name"] for profile in status["profiles"]},
            {"training", "coding", "workplace", "copilot"},
        )

    def test_manual_training_rejects_unsafe_adamw_learning_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            self.runtime.train("sample", epochs=1, learning_rate=0.02)


if __name__ == "__main__":
    unittest.main()
