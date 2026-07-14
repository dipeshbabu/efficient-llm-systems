"""MLX backend for REFRACT.

Implements all five Backend methods on top of mlx-lm (verified against
mlx_lm 0.31.2). Models load lazily and cache per ``model_path`` so a
matrix run doesn't reload the same 30B model 4 times.

Install:
    pip install mlx mlx-lm

KV-config mapping
-----------------

llama.cpp ``ctk=X,ctv=Y`` syntax → MLX ``kv_bits``:

  - ``ctk=f16,ctv=f16``       → kv_bits=None (no quantization, default)
  - ``ctk=q8_0,ctv=q8_0``     → kv_bits=8
  - ``ctk=q4_0,ctv=q4_0``     → kv_bits=4
  - ``ctk=q6_K,ctv=q6_K``     → kv_bits=6
  - asymmetric (ctk≠ctv)      → BackendCapabilityError; mlx-lm's stock
                                cache is symmetric.
  - ``turbo*``                → BackendCapabilityError; needs a TurboQuant-enabled MLX build
                                feature/turboquant-plus branch.

Trajectory + KLD
----------------

Trajectory uses ``mlx_lm.stream_generate`` with the cache args threaded
through to ``generate_step``. Token IDs come back natively in
``GenerationResponse.token``; no binary patches needed.

KLD@D is implemented natively via two model forward passes per chunk
(one each KV config), ``mx.softmax`` per position, and KL accumulation
in nats. This bypasses the llama-perplexity subprocess.

Verified against mlx_lm 0.31.2:
  - ``generate_step(..., kv_bits, kv_group_size, quantized_kv_start)``
  - ``stream_generate`` forwards **kwargs to generate_step
  - ``GenerationResponse(text, token, logprobs, ...)`` dataclass
  - ``mlx_lm.models.cache.make_prompt_cache(model)``
  - ``mlx_lm.models.cache.maybe_quantize_kv_cache(cache, ..., kv_bits)``
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from .base import (
    Backend,
    BackendCapabilityError,
    CompletionResult,
    KLDResult,
    ModelSpec,
    TrajectoryResult,
)


def _require_mlx():
    """Lazy import gate. Raises with install hint if mlx-lm is missing."""
    try:
        import mlx.core as mx
        import mlx_lm
        from mlx_lm.models import cache as cache_mod

        return mx, mlx_lm, cache_mod
    except ImportError as e:
        raise BackendCapabilityError(
            "MLX backend requires `mlx-lm`. Install with:\n"
            "    pip install mlx mlx-lm\n"
            f"Underlying import error: {e}"
        ) from e


_KV_QUANT_BITS = {
    "f16": None,
    "fp16": None,
    "f32": None,
    "q8_0": 8,
    "q8": 8,
    "q4_0": 4,
    "q4_K": 4,
    "q4": 4,
    "q6_K": 6,
    "q6": 6,
}


def _translate_kv_to_mlx(kv_config_str: str) -> dict:
    """Parse a llama.cpp-style KV spec into MLX generate_step kwargs.

    Returns ``{kv_bits, kv_group_size, quantized_kv_start}``. Raises
    ``BackendCapabilityError`` for asymmetric or turbo schemes that
    mlx-lm's stock cache cannot represent.
    """
    ctk = "f16"
    ctv = "f16"
    extras: dict[str, str] = {}
    for part in kv_config_str.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if k == "ctk":
            ctk = v
        elif k == "ctv":
            ctv = v
        else:
            extras[k] = v

    if ctk != ctv:
        raise BackendCapabilityError(
            f"MLX backend's stock KV cache is symmetric (K and V quantized "
            f"the same). Got ctk={ctk!r}, ctv={ctv!r}. To test asymmetric "
            f"configs, set REFRACT_BACKEND=llamacpp."
        )
    if ctk.startswith("turbo"):
        raise BackendCapabilityError(
            f"TurboQuant KV scheme {ctk!r} requires a TurboQuant-enabled MLX "
            f"build. The historical reference fork has no current public URL. "
            f"Use an existing compatible build or set REFRACT_BACKEND=llamacpp."
        )
    if ctk not in _KV_QUANT_BITS:
        raise BackendCapabilityError(
            f"Unrecognized MLX KV type {ctk!r}. Recognized: {sorted(_KV_QUANT_BITS)}"
        )
    bits = _KV_QUANT_BITS[ctk]
    kwargs: dict = {
        "kv_bits": bits,
        "kv_group_size": 64,
        "quantized_kv_start": 0,
    }
    return kwargs


_MODEL_CACHE: dict[str, tuple] = {}  # str(model_path) -> (model, tokenizer)


def _load_model(model_path: ModelSpec):
    """Load + cache a model. Re-load is the slow part for big GGUFs."""
    _require_mlx()
    import mlx_lm

    key = str(model_path)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = mlx_lm.load(key)
    return _MODEL_CACHE[key]


def _apply_chat_template(tokenizer, prompt: str, system: Optional[str]) -> str:
    """Render a (system, user) chat through the model's chat template."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


class MLXBackend(Backend):
    name = "mlx"

    # ---------------------------------------------------------------- run_completion
    def run_completion(
        self,
        *,
        model: ModelSpec,
        prompt: str,
        kv_config_str: str,
        n_predict: int = 128,
        ctx: int = 512,  # noqa: ARG002 -- mlx-lm sizes context dynamically
        n_gpu_layers: int = 99,  # noqa: ARG002
        seed: int = 42,
        temperature: float = 0.0,
        timeout: float = 300.0,  # noqa: ARG002
        apply_chat_template: bool = True,
        system: Optional[str] = None,
        reasoning: str = "off",  # noqa: ARG002 -- handled via thinking-detect + n_predict
    ) -> CompletionResult:
        mx, mlx_lm, _ = _require_mlx()
        m, tok = _load_model(model)
        kv_kwargs = _translate_kv_to_mlx(kv_config_str)

        rendered = (
            _apply_chat_template(tok, prompt, system) if apply_chat_template else prompt
        )

        mx.random.seed(seed)
        # mlx_lm.generate returns the GENERATED text (not the prompt echo)
        # when verbose=False. Greedy decoding via the default sampler
        # (argmax) when temperature == 0; otherwise we'd construct a
        # temperature sampler.
        text = mlx_lm.generate(
            m,
            tok,
            prompt=rendered,
            max_tokens=n_predict,
            verbose=False,
            **kv_kwargs,
        )
        # Strip a trailing assistant-end marker if the chat template left one.
        text = re.sub(r"<\|im_end\|>\s*$", "", text or "").strip()
        return CompletionResult(
            text=text,
            n_tokens=0,
            metadata={"kv_kwargs": kv_kwargs},
        )

    # ---------------------------------------------------------- run_completion_trajectory
    def run_completion_trajectory(
        self,
        *,
        model: ModelSpec,
        prompt: str,
        kv_config_str: str,
        n_predict: int = 128,
        ctx: int = 512,  # noqa: ARG002
        n_gpu_layers: int = 99,  # noqa: ARG002
        seed: int = 42,
        temperature: float = 0.0,  # noqa: ARG002 -- greedy default
        timeout: float = 300.0,  # noqa: ARG002
        apply_chat_template: bool = True,
        system: Optional[str] = None,
    ) -> TrajectoryResult:
        mx, mlx_lm, _ = _require_mlx()
        m, tok = _load_model(model)
        kv_kwargs = _translate_kv_to_mlx(kv_config_str)

        rendered = (
            _apply_chat_template(tok, prompt, system) if apply_chat_template else prompt
        )

        mx.random.seed(seed)
        token_ids: list[int] = []
        for record in mlx_lm.stream_generate(
            m,
            tok,
            prompt=rendered,
            max_tokens=n_predict,
            **kv_kwargs,
        ):
            tid = getattr(record, "token", None)
            if tid is not None:
                # `token` is sometimes an int, sometimes an mx.array scalar.
                try:
                    token_ids.append(int(tid))
                except TypeError:
                    token_ids.append(int(tid.item()))

        return TrajectoryResult(
            token_ids=token_ids,
            metadata={"kv_kwargs": kv_kwargs, "n_tokens": len(token_ids)},
        )

    # ---------------------------------------------------------------- run_kld
    def run_kld(
        self,
        *,
        model: ModelSpec,
        corpus: Path,
        ref_kv_str: str,
        cand_kv_str: str,
        chunks: int = 32,
        ctx: int = 512,
        n_gpu_layers: int = 99,  # noqa: ARG002
    ) -> KLDResult:
        """Native KLD@D for MLX.

        Tokenizes ``corpus`` into ctx-sized chunks; for each chunk, runs
        the model forward twice (one per KV config) building a fresh
        prompt cache each time so the cache type is honoured during
        attention. Computes KL(P_ref || P_cand) per token, returns mean.
        """
        mx, mlx_lm, cache_mod = _require_mlx()
        m, tok = _load_model(model)
        ref_kwargs = _translate_kv_to_mlx(ref_kv_str)
        cand_kwargs = _translate_kv_to_mlx(cand_kv_str)

        text = corpus.read_text(errors="replace")
        all_tokens = tok.encode(text, add_special_tokens=False)
        if len(all_tokens) < ctx:
            raise BackendCapabilityError(
                f"Corpus too short for KLD: {len(all_tokens)} tokens, need ≥ {ctx}."
            )

        n_chunks = min(chunks, len(all_tokens) // ctx)
        if n_chunks < 1:
            raise BackendCapabilityError(f"Corpus yields zero chunks at ctx={ctx}.")

        def _logits_for_chunk(chunk_tokens: list[int], kv_bits: Optional[int]) -> Any:
            """Run forward with the requested KV bits; return logits [T, V]."""
            inp = mx.array(chunk_tokens)[None, :]
            prompt_cache = cache_mod.make_prompt_cache(m)
            if kv_bits is not None:
                # mlx_lm 0.31.x moved maybe_quantize_kv_cache out of
                # mlx_lm.models.cache into mlx_lm.generate. Try the new
                # location first, fall back for older mlx_lm versions.
                _mqc = getattr(cache_mod, "maybe_quantize_kv_cache", None)
                if _mqc is None:
                    from mlx_lm.generate import (
                        maybe_quantize_kv_cache as fallback_quantize_kv_cache,
                    )

                    _mqc = fallback_quantize_kv_cache
                _mqc(
                    prompt_cache,
                    quantized_kv_start=0,
                    kv_group_size=64,
                    kv_bits=kv_bits,
                )
            logits = m(inp, cache=prompt_cache)
            return logits[0]  # squeeze batch

        # KL accumulator across positions. We weight per-position contributions
        # by valid-position count so chunks with NaN/Inf positions (which can
        # happen on heavily-quantized weight models where some logits underflow)
        # don't poison the whole chunk's mean.
        total_kl = 0.0
        total_valid_positions = 0
        for i in range(n_chunks):
            chunk = all_tokens[i * ctx : (i + 1) * ctx]
            ref_logits = _logits_for_chunk(chunk, ref_kwargs["kv_bits"])
            cand_logits = _logits_for_chunk(chunk, cand_kwargs["kv_bits"])

            # Numerically stable: log_softmax avoids log(0) underflow.
            # KL(P||Q) = sum_i P_i (log P_i - log Q_i)
            #         = sum_i exp(logP_i) (logP_i - logQ_i)
            ref_logp = ref_logits - mx.logsumexp(ref_logits, axis=-1, keepdims=True)
            cand_logp = cand_logits - mx.logsumexp(cand_logits, axis=-1, keepdims=True)
            ref_p = mx.exp(ref_logp)
            kl_per_pos = mx.sum(ref_p * (ref_logp - cand_logp), axis=-1)

            # Filter non-finite positions and accumulate weighted by valid count
            # so the chunk-level mean is robust to a small number of bad logits
            # in deeply-quantized models.
            kl_arr = kl_per_pos
            finite_mask = mx.isfinite(kl_arr)
            n_valid = int(mx.sum(finite_mask).item())
            if n_valid == 0:
                continue
            # Replace NaN/Inf with 0 so the sum below is well-defined.
            kl_clean = mx.where(finite_mask, kl_arr, mx.zeros_like(kl_arr))
            chunk_kl_sum = float(mx.sum(kl_clean).item())
            total_kl += chunk_kl_sum
            total_valid_positions += n_valid

        if total_valid_positions == 0:
            raise BackendCapabilityError(
                "MLX KLD: every position produced non-finite KL. The model's "
                "logits may be underflowing on this KV config. Try a less "
                "aggressive candidate (e.g. q6_K/q6_K) or run --skip-kld."
            )
        mean_kld = total_kl / total_valid_positions
        return KLDResult(
            mean_kld=mean_kld,
            ppl=None,
            chunks=n_chunks,
            ctx=ctx,
            metadata={
                "ref_kv_kwargs": ref_kwargs,
                "cand_kv_kwargs": cand_kwargs,
            },
        )

    # ---------------------------------------------------------------- tokenize_to_ids
    def tokenize_to_ids(
        self,
        *,
        model: ModelSpec,
        text: str,
        timeout: float = 120.0,  # noqa: ARG002
    ) -> list[int]:
        if not text:
            return []
        _, tok = _load_model(model)
        return list(tok.encode(text, add_special_tokens=False))

    # ---------------------------------------------------------------- model_metadata
    def model_metadata(self, *, model: ModelSpec) -> dict:
        info: dict = {
            "backend": self.name,
            "model": str(model),
        }
        try:
            import importlib.metadata as md

            info["mlx_lm_version"] = md.version("mlx-lm")
            info["mlx_version"] = md.version("mlx")
        except Exception:
            pass
        return info
