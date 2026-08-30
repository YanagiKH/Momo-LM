"""Compatibility import for projects that prefer ``import MomoLM``."""

from momo_lm import MomoConfig, MomoLM, MomoRuntime, __version__, load_model

__all__ = ["MomoConfig", "MomoLM", "MomoRuntime", "__version__", "load_model"]
