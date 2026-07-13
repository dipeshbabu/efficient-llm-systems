"""REFRACT backend abstraction.

Each backend (llama.cpp, MLX, vLLM, SGLang) implements the :class:`Backend` ABC.
Axes call into the backend rather than into a hardcoded subprocess
wrapper, so the same scoring framework works on any inference engine
that can give us:

  - text-in / text-out completion under a given KV config
  - per-token model-token IDs at decode time (for trajectory)
  - per-token KL divergence vs a reference (for KLD@D)
  - chat-template application (so instruct models engage Q&A mode)
  - tokenization to integer ID list (for PLAD edit distance)

Backend selection
-----------------

`get_backend(name)`:
    Explicit selection by name ('llamacpp' | 'mlx' | 'vllm' | 'sglang').

`auto_backend(model)`:
    Pick by inspecting the path. ``.gguf`` → llama.cpp;
    a directory with recognizable MLX quantization metadata → MLX;
    anything else → vLLM. Ambiguous directories require an explicit backend.

Override via env var ``REFRACT_BACKEND``.

Status (v0.3.4):

  - llamacpp: production (primary dev target on macOS Apple Silicon
              + Linux Ubuntu via the patched binary)
  - mlx:      production (Apple Silicon native; mlx-lm 0.31+)
  - vllm:     production (HF safetensors on CUDA / ROCm; cached
              in-process LLM with evict-on-key-change for memory-pressured
              hybrid models; verified on AMD MI300X, ROCm 7.2)
  - sglang:   production (HTTP client; SGLang server runs separately,
              typically Docker; verified on AMD MI300X, ROCm 7.2)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .base import Backend, BackendCapabilityError


def get_backend(name: str) -> Backend:
    """Return a backend instance by name. Raises ValueError for unknown."""
    name = name.lower()
    if name == "llamacpp":
        from .llamacpp import LlamaCppBackend

        return LlamaCppBackend()
    if name == "mlx":
        from .mlx import MLXBackend

        return MLXBackend()
    if name == "vllm":
        from .vllm import VLLMBackend

        return VLLMBackend()
    if name == "sglang":
        from .sglang import SGLangBackend

        return SGLangBackend()
    raise ValueError(
        f"Unknown backend {name!r}. Valid: 'llamacpp', 'mlx', 'vllm', 'sglang'."
    )


def auto_backend(model: Path) -> Backend:
    """Pick the backend by inspecting the model path + REFRACT_BACKEND env.

    Resolution order:
      1. ``REFRACT_BACKEND`` env var (explicit override)
      2. Path suffix: ``.gguf`` → llama.cpp
      3. Directory with MLX quantization metadata or NPZ weights → mlx
      4. Anything else → vllm (which loads via Hugging Face IDs or local dirs)
    """
    env = os.environ.get("REFRACT_BACKEND")
    if env:
        return get_backend(env)
    if model.suffix == ".gguf":
        return get_backend("llamacpp")
    if model.is_dir():
        # MLX-LM converted models normally carry a top-level ``quantization``
        # block with ``bits`` and ``group_size``. A plain Hugging Face
        # ``config.json`` is ambiguous and must not silently route to MLX.
        config_path = model / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                quant = config.get("quantization") or {}
                if {"bits", "group_size"}.issubset(quant):
                    return get_backend("mlx")
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
        if any(model.glob("*.npz")):
            return get_backend("mlx")
    return get_backend("vllm")


__all__ = [
    "Backend",
    "BackendCapabilityError",
    "get_backend",
    "auto_backend",
]
