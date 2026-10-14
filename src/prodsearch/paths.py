"""Where data, models and built indexes live. Nothing here is a machine-specific path."""

from __future__ import annotations

import os
from pathlib import Path

APP = "hybrid-product-search"


def data_dir() -> Path:
    """Root for WANDS, embeddings and built indexes (`PRODSEARCH_DATA_DIR`)."""
    env = os.environ.get("PRODSEARCH_DATA_DIR")
    return Path(env).expanduser() if env else Path.home() / ".cache" / APP


def wands_dir() -> Path:
    return data_dir() / "wands"


def index_dir() -> Path:
    return data_dir() / "index"


def hf_home() -> Path:
    """The Hugging Face cache root, so models fetched by other tools are reused."""
    env = os.environ.get("HF_HOME")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "huggingface"
