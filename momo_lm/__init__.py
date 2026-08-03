"""Momo-LM: a local-first, trainable AI workbench."""

from .api import MomoLM, load_model
from .backend import get_backend
from .config import MomoConfig
from .runtime import MomoRuntime

__all__ = ["MomoConfig", "MomoLM", "MomoRuntime", "get_backend", "load_model"]
__version__ = "0.2.0"
