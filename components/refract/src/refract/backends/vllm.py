"""vLLM backend for REFRACT.

Drives single-process vLLM via the in-tree ``vllm.LLM`` class. Handles
chat-template formatting, KV-config translation (llama.cpp's
``ctk=...,ctv=...`` strings → vLLM's ``kv_cache_dtype=...``), greedy
decode trajectories, and KL divergence via ``SamplingParams.prompt_logprobs``.

LLM instances are cached per ``(model, kv_cache_dtype, max_model_len)``
tuple across calls so back-to-back ``run_completion`` and ``run_kld``
against the same config don't pay the ~3 minute weight-load cost twice.

Env knobs
---------

  REFRACT_VLLM_GPU_MEMORY_UTILIZATION   default 0.45 (fits two parallel
                                        LLM instances on one GPU)
  REFRACT_VLLM_MAX_MODEL_LEN            default 4096; bumped per-call as
                                        needed by ctx + n_predict
  REFRACT_VLLM_KLD_TOPK                 default 64; top-K used for the
                                        prompt_logprobs distribution
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Optional

from .base import (
    Backend,
    BackendCapabilityError,
    CompletionResult,
    KLDResult,
    TrajectoryResult,
    approximate_topk_kl,
)

# llama.cpp KV cache string → vLLM kv_cache_dtype.
# Every config the score axes will pass must round-trip here.
_CTK_CTV_TO_VLLM: dict[tuple[str, str], str] = {
    ("f16", "f16"): "auto",
    ("bf16", "bf16"): "auto",
    ("q8_0", "q8_0"): "fp8_e4m3",
    # TurboQuant presets (historical vLLM fork or upstream PR #38479)
    ("q8_0", "turbo4"): "turboquant_k8v4",
    ("q8_0", "turbo3"): "turboquant_k8v3",
    ("turbo4", "turbo4"): "turboquant_4bit_nc",
    ("turbo3", "turbo3"): "turboquant_3bit_nc",
    ("turbo3", "turbo4"): "turboquant_k3v4_nc",
    # TQ+ extensions in dipeshbabu fork
    ("turbo4", "turbo4_rv"): "turboquant_4bit_nc_rv",
    ("turbo4", "turbo4_cv"): "turboquant_4bit_nc_cv_rv",
}

_VLLM_LLM_CACHE: dict[tuple[str, str, int], Any] = {}


def _kv_str_to_vllm_dtype(kv_str: str) -> str:
    """Translate a REFRACT ``ctk=...,ctv=...`` string to vLLM kv_cache_dtype.

    Raises BackendCapabilityError when the combination isn't representable.
    """
    parts = dict(p.split("=", 1) for p in kv_str.split(",") if "=" in p)
    ctk = parts.get("ctk", "f16").lower()
    ctv = parts.get("ctv", "f16").lower()
    key = (ctk, ctv)
    if key not in _CTK_CTV_TO_VLLM:
        raise BackendCapabilityError(
            f"vLLM backend has no mapping for ctk={ctk}, ctv={ctv}. "
            f"Add it to _CTK_CTV_TO_VLLM or use --backend llamacpp."
        )
    return _CTK_CTV_TO_VLLM[key]


def _max_model_len_default() -> int:
    return int(os.environ.get("REFRACT_VLLM_MAX_MODEL_LEN", "4096"))


def _get_llm(model: Path, kv_dtype: str, max_model_len: int) -> Any:
    """Cache one LLM instance at a time (per process).

    Hybrid models like Qwen3.6-35B-A3B don't fit two simultaneous LLM
    instances on a single accelerator (model weights + Mamba state ~120GB
    × 2 > 192GB). When the requested key differs from the cached one, the
    cached LLM is evicted before loading the new one.
    """
    key = (str(model), kv_dtype, max_model_len)
    if key in _VLLM_LLM_CACHE:
        return _VLLM_LLM_CACHE[key]
    if _VLLM_LLM_CACHE:
        # Evict any prior LLMs to free GPU memory before loading a new one.
        import gc

        for k in list(_VLLM_LLM_CACHE.keys()):
            del _VLLM_LLM_CACHE[k]
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    from vllm import LLM

    gpu_mem = float(os.environ.get("REFRACT_VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
    max_num_seqs = int(os.environ.get("REFRACT_VLLM_MAX_NUM_SEQS", "32"))
    llm = LLM(
        model=str(model),
        dtype="bfloat16",
        kv_cache_dtype=kv_dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        max_num_seqs=max_num_seqs,
        disable_log_stats=True,
        enable_prefix_caching=False,
        max_num_batched_tokens=max(2048, max_model_len // 4),
    )
    _VLLM_LLM_CACHE[key] = llm
    return llm


def _format_prompt(
    llm: Any, prompt: str, *, system: Optional[str], apply_template: bool
) -> str:
    if not apply_template:
        return prompt
    tok = llm.get_tokenizer()
    msgs: list[dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt


class VLLMBackend(Backend):
    name = "vllm"

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
    ) -> CompletionResult:
        from vllm import SamplingParams

        kv_dtype = _kv_str_to_vllm_dtype(kv_config_str)
        max_len = max(_max_model_len_default(), ctx + n_predict + 32)
        llm = _get_llm(model, kv_dtype, max_len)
        text_prompt = _format_prompt(
            llm, prompt, system=system, apply_template=apply_chat_template
        )
        sp = SamplingParams(
            max_tokens=n_predict,
            temperature=temperature,
            seed=seed if temperature > 0 else None,
        )
        out = llm.generate([text_prompt], sp, use_tqdm=False)[0].outputs[0]
        return CompletionResult(
            text=out.text,
            n_tokens=len(out.token_ids),
            metadata={"kv_cache_dtype": kv_dtype},
        )

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
    ) -> TrajectoryResult:
        from vllm import SamplingParams

        kv_dtype = _kv_str_to_vllm_dtype(kv_config_str)
        max_len = max(_max_model_len_default(), ctx + n_predict + 32)
        llm = _get_llm(model, kv_dtype, max_len)
        text_prompt = _format_prompt(
            llm, prompt, system=system, apply_template=apply_chat_template
        )
        sp = SamplingParams(
            max_tokens=n_predict,
            temperature=temperature,
            seed=seed if temperature > 0 else None,
        )
        out = llm.generate([text_prompt], sp, use_tqdm=False)[0].outputs[0]
        return TrajectoryResult(
            token_ids=list(out.token_ids),
            metadata={"kv_cache_dtype": kv_dtype, "text": out.text},
        )

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
    ) -> KLDResult:
        """Per-token KL(P_ref || P_cand) on chunks of `corpus`.

        Top-K next-token distribution from ``SamplingParams.prompt_logprobs``;
        K cap controls fidelity vs cost (env REFRACT_VLLM_KLD_TOPK, default 64).
        """
        from vllm import SamplingParams

        ref_dtype = _kv_str_to_vllm_dtype(ref_kv_str)
        cand_dtype = _kv_str_to_vllm_dtype(cand_kv_str)
        max_len = max(_max_model_len_default(), ctx + 8)
        topk = int(os.environ.get("REFRACT_VLLM_KLD_TOPK", "64"))

        text = Path(corpus).read_text(encoding="utf-8", errors="replace")
        ref_llm = _get_llm(model, ref_dtype, max_len)
        tok = ref_llm.get_tokenizer()
        ids = tok.encode(text, add_special_tokens=False)
        chunk_len = ctx - 1
        slices = [
            ids[i : i + chunk_len] for i in range(0, len(ids) - chunk_len, chunk_len)
        ][:chunks]
        if not slices:
            raise BackendCapabilityError(
                f"corpus too short for ctx={ctx}, chunks={chunks} "
                f"(have {len(ids)} tokens, need at least {chunk_len})"
            )

        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=topk)

        def _run(llm: Any) -> list[list[dict[int, float]]]:
            res: list[list[dict[int, float]]] = []
            for ch in slices:
                out = llm.generate({"prompt_token_ids": ch}, sp, use_tqdm=False)[0]
                pl = out.prompt_logprobs or []
                pos: list[dict[int, float]] = []
                for entry in pl:
                    if entry is None:
                        pos.append({})
                    else:
                        pos.append({tid: lp.logprob for tid, lp in entry.items()})
                res.append(pos)
            return res

        ref_logp = _run(ref_llm)
        cand_llm = _get_llm(model, cand_dtype, max_len)
        cand_logp = _run(cand_llm)

        # Aligned, normalized top-k estimate. It includes a residual bucket
        # for omitted vocabulary mass but is not a full-vocabulary KL.
        LOG_FLOOR = -30.0
        total_kl = 0.0
        n_pos = 0
        sq_dp_sum = 0.0
        n_dp = 0
        same_topp_hits = 0
        same_topp_n = 0
        for ref_chunk, cand_chunk in zip(ref_logp, cand_logp, strict=False):
            for ref_pos, cand_pos in zip(ref_chunk, cand_chunk, strict=False):
                if not ref_pos or not cand_pos:
                    continue
                n_pos += 1
                for tid, ref_lp in ref_pos.items():
                    p = math.exp(ref_lp)
                    cand_lp = cand_pos.get(tid, LOG_FLOOR)
                    if p > 1e-9:
                        sq_dp_sum += ((math.exp(cand_lp) - p) / p) ** 2
                        n_dp += 1
                total_kl += approximate_topk_kl(ref_pos, cand_pos, log_floor=LOG_FLOOR)
                ref_top = max(ref_pos.items(), key=lambda kv: kv[1])[0]
                cand_top = max(cand_pos.items(), key=lambda kv: kv[1])[0]
                same_topp_hits += int(ref_top == cand_top)
                same_topp_n += 1

        mean_kl = total_kl / max(n_pos, 1)
        rms_dp_pct = 100.0 * math.sqrt(sq_dp_sum / max(n_dp, 1)) if n_dp else None
        same_topp_pct = (
            100.0 * same_topp_hits / max(same_topp_n, 1) if same_topp_n else None
        )
        return KLDResult(
            mean_kld=mean_kl,
            rms_dp_pct=rms_dp_pct,
            same_topp_pct=same_topp_pct,
            chunks=len(slices),
            ctx=ctx,
            metadata={
                "ref_kv_cache_dtype": ref_dtype,
                "cand_kv_cache_dtype": cand_dtype,
                "topk": topk,
                "n_positions_scored": n_pos,
                "kld_estimator": "normalized_top_k_with_other_bucket",
                "full_vocabulary": False,
            },
        )

    def tokenize_to_ids(
        self,
        *,
        model: Path,
        text: str,
        timeout: float = 120.0,
    ) -> list[int]:
        max_len = _max_model_len_default()
        llm = _get_llm(model, "auto", max_len)
        return llm.get_tokenizer().encode(text, add_special_tokens=False)

    def model_metadata(self, *, model: Path) -> dict:
        try:
            import vllm

            ver = vllm.__version__
        except Exception:
            ver = "unknown"
        return {
            "backend": self.name,
            "model": str(model),
            "vllm_version": ver,
        }
