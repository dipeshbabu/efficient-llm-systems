"""Abstract Backend interface for REFRACT.

A backend wraps an inference engine (llama.cpp, MLX, vLLM, …) and exposes
the four primitives REFRACT axes need:

  - ``run_completion``       text-in/text-out with chat template + KV config
  - ``run_completion_trajectory``  decode-time token-ID capture
  - ``run_kld``              per-token KL divergence vs a reference, on a corpus
  - ``tokenize_to_ids``      tokenization for edit-distance and unit-matching
  - ``detect_thinking_mode`` runtime probe so axes can adapt n_predict / pre-fill
  - ``model_metadata``       framework version stamp + any backend-specific notes
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class BackendCapabilityError(RuntimeError):
    """Raised when a backend doesn't support a feature that an axis requires.

    Backends should raise this with a clear remediation hint so the user
    knows whether to switch backends or skip the axis.
    """


@dataclass
class CompletionResult:
    """Result of a backend.run_completion call."""

    text: str  # post-noise-strip completion text
    n_tokens: int  # tokens actually decoded
    metadata: dict = field(default_factory=dict)  # backend-specific extras


@dataclass
class TrajectoryResult:
    """Result of a backend.run_completion_trajectory call."""

    token_ids: list[int]  # actual sampled IDs at decode time
    metadata: dict = field(default_factory=dict)


@dataclass
class KLDResult:
    """Result of a backend.run_kld call (Axis B)."""

    mean_kld: float  # nats
    ppl: Optional[float] = None
    rms_dp_pct: Optional[float] = None
    same_topp_pct: Optional[float] = None
    chunks: int = 0
    ctx: int = 0
    metadata: dict = field(default_factory=dict)


def approximate_topk_kl(
    reference_logprobs: dict[int, float],
    candidate_logprobs: dict[int, float],
    *,
    log_floor: float = -30.0,
) -> float:
    """Return a normalized top-k KL estimate with an omitted-mass bucket.

    Native vLLM and SGLang APIs expose only top-k log probabilities. Treating
    their partial sum as a full distribution is not KL divergence and can
    produce misleading cross-backend values. This helper aligns the union of
    visible token IDs, assigns a small floor to tokens absent on one side,
    adds one bucket for all omitted vocabulary mass, normalizes both vectors,
    and computes ``KL(reference || candidate)``.

    It remains an approximation and callers must label it as such.
    """
    if not reference_logprobs or not candidate_logprobs:
        raise ValueError("both top-k distributions must be non-empty")

    floor = math.exp(log_floor)
    token_ids = sorted(set(reference_logprobs) | set(candidate_logprobs))
    p = [math.exp(reference_logprobs.get(tid, log_floor)) for tid in token_ids]
    q = [math.exp(candidate_logprobs.get(tid, log_floor)) for tid in token_ids]
    p.append(max(1.0 - sum(p), floor))
    q.append(max(1.0 - sum(q), floor))

    p_total = sum(p)
    q_total = sum(q)
    p = [value / p_total for value in p]
    q = [value / q_total for value in q]
    value = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q) if pi > 0.0)
    return max(value, 0.0)


class Backend(abc.ABC):
    """Abstract REFRACT backend.

    Implementations must be importable without their underlying inference
    engine being installed (use lazy imports inside methods). This lets a
    user with only llama.cpp run REFRACT without paying the cost of
    importing mlx or vllm at startup.
    """

    name: str = "abstract"

    @abc.abstractmethod
    def run_completion(
        self,
        *,
        model: Path,
        prompt: str,
        kv_config_str: str,
        n_predict: int = 128,
        ctx: int = 512,
        n_gpu_layers: int = 99,
        seed: int = 42,
        temperature: float = 0.0,
        timeout: float = 300.0,
        apply_chat_template: bool = True,
        system: Optional[str] = None,
        reasoning: str = "off",
    ) -> CompletionResult: ...

    @abc.abstractmethod
    def run_completion_trajectory(
        self,
        *,
        model: Path,
        prompt: str,
        kv_config_str: str,
        n_predict: int = 128,
        ctx: int = 512,
        n_gpu_layers: int = 99,
        seed: int = 42,
        temperature: float = 0.0,
        timeout: float = 300.0,
        apply_chat_template: bool = True,
        system: Optional[str] = None,
    ) -> TrajectoryResult: ...

    @abc.abstractmethod
    def run_kld(
        self,
        *,
        model: Path,
        corpus: Path,
        ref_kv_str: str,
        cand_kv_str: str,
        chunks: int = 32,
        ctx: int = 512,
        n_gpu_layers: int = 99,
    ) -> KLDResult: ...

    @abc.abstractmethod
    def tokenize_to_ids(
        self,
        *,
        model: Path,
        text: str,
        timeout: float = 120.0,
    ) -> list[int]: ...

    def detect_thinking_mode(
        self,
        *,
        model: Path,
        timeout: float = 30.0,
    ) -> tuple[bool, list[str]]:
        """Run a tiny probe and return ``(detected, markers_found)``.

        Default implementation issues a "What is 2+2?" generation and
        scans the response for canonical thinking markers. Subclasses can
        override with a cheaper signal (read GGUF chat_template, etc.).
        """
        markers = (
            "<think>",
            "</think>",
            "<|thinking|>",
            "<|end_thinking|>",
            "<|channel|>analysis",
            "<|channel|>commentary",
            "[Start thinking]",
            "[End thinking]",
            "<thinking>",
            "</thinking>",
        )
        try:
            result = self.run_completion(
                model=model,
                prompt="What is 2+2? Answer briefly.",
                kv_config_str="ctk=f16,ctv=f16",
                n_predict=64,
                ctx=128,
                temperature=0.0,
                seed=42,
                timeout=timeout,
            )
        except Exception:
            return False, []
        text = result.text or ""
        hit = [m for m in markers if m in text]
        return bool(hit), hit

    def model_metadata(self, *, model: Path) -> dict:
        """Return backend-specific metadata to embed in the JSON report.

        Default: backend name + model path basename. Overridable to capture
        commit hashes, library versions, GGUF metadata, etc.
        """
        return {
            "backend": self.name,
            "model": model.as_posix(),
        }
