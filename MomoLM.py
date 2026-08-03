"""Compatibility import for projects that prefer ``import MomoLM``."""

from momo_lm import MomoConfig, MomoLM, MomoRuntime, load_model

__all__ = ["MomoConfig", "MomoLM", "MomoRuntime", "load_model"]
