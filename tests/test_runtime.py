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
        self.runtime.generate_image("pink moon", output, width=160, height=128, seed=4)
        self.assertTrue(output.exists())
        self.assertGreater(output.stat().st_size, 1000)

    def test_weight_inspection(self) -> None:
        status = self.runtime.status()
        self.assertGreater(status["weights"]["parameters"], 50_000)
        self.assertGreater(status["image_weights"]["parameters"], 1_000)
        self.assertGreaterEqual(status["knowledge"]["documents"], 20)

    def test_offline_speech_always_produces_wav(self) -> None:
        output = Path(self.temporary.name) / "speech.wav"
        result = self.runtime.speech.synthesize("Momo local speech", output)
        self.assertTrue(result["engine"])
        self.assertEqual(output.read_bytes()[:4], b"RIFF")


if __name__ == "__main__":
    unittest.main()
