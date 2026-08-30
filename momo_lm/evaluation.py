from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .model import NeuralTextModel
from .tokenizer import ByteTokenizer


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    negative_log_likelihood: float
    perplexity: float
    top1_accuracy: float
    targets: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_text(
    model: NeuralTextModel,
    text: str,
    *,
    batch_size: int = 256,
) -> EvaluationMetrics:
    """Evaluate next-token predictions without changing model or optimizer state."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(text, str):
        raise TypeError("evaluation text must be a string")
    tokens = ByteTokenizer().encode(text, bos=True, eos=True)
    context_stream, targets = model._context_stream(tokens)
    if not len(targets):
        return EvaluationMetrics(0.0, 1.0, 0.0, 0)
    total_loss = 0.0
    correct = 0
    for start in range(0, len(targets), batch_size):
        positions = np.arange(start, min(start + batch_size, len(targets)), dtype=np.int64)
        selected_contexts, selected_targets = model._context_batch(
            context_stream, targets, positions
        )
        scores, _, _ = model.logits(selected_contexts, inference=True)
        maximum = scores.max(axis=1, keepdims=True)
        log_normalizer = maximum[:, 0] + np.log(np.exp(scores - maximum).sum(axis=1))
        rows = np.arange(len(selected_targets))
        total_loss += float(np.sum(log_normalizer - scores[rows, selected_targets]))
        correct += int(np.sum(np.argmax(scores, axis=1) == selected_targets))
    nll = total_loss / len(targets)
    return EvaluationMetrics(
        negative_log_likelihood=float(nll),
        perplexity=float(math.exp(min(nll, 50.0))),
        top1_accuracy=float(correct / len(targets)),
        targets=int(len(targets)),
    )


def compare_text(
    baseline: NeuralTextModel,
    candidate: NeuralTextModel,
    text: str,
    *,
    batch_size: int = 256,
) -> dict[str, dict[str, float | int] | bool]:
    before = evaluate_text(baseline, text, batch_size=batch_size)
    after = evaluate_text(candidate, text, batch_size=batch_size)
    return {
        "baseline": before.to_dict(),
        "candidate": after.to_dict(),
        "nll_improved": after.negative_log_likelihood < before.negative_log_likelihood,
    }
