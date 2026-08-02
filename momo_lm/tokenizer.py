from __future__ import annotations


class ByteTokenizer:
    """Lossless UTF-8 byte tokenizer with a fixed, language-neutral vocabulary."""

    PAD = 0
    BOS = 1
    EOS = 2
    OFFSET = 3
    vocab_size = 259

    def encode(self, text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
        tokens = [self.OFFSET + value for value in text.encode("utf-8", errors="replace")]
        if bos:
            tokens.insert(0, self.BOS)
        if eos:
            tokens.append(self.EOS)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        values = bytes(token - self.OFFSET for token in tokens if token >= self.OFFSET)
        return values.decode("utf-8", errors="ignore")
