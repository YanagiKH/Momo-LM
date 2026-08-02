import unittest

from momo_lm.tokenizer import ByteTokenizer


class TokenizerTests(unittest.TestCase):
    def test_round_trip_multilingual_text(self) -> None:
        tokenizer = ByteTokenizer()
        text = "Momo 你好 こんにちは 🌸"
        tokens = tokenizer.encode(text, bos=True, eos=True)
        self.assertEqual(tokens[0], tokenizer.BOS)
        self.assertEqual(tokens[-1], tokenizer.EOS)
        self.assertEqual(tokenizer.decode(tokens), text)


if __name__ == "__main__":
    unittest.main()
