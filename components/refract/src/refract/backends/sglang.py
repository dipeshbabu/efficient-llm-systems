"""SGLang backend for REFRACT.

HTTP-based: assumes one or more SGLang servers are already running at
configurable URLs. The backend doesn't manage server lifecycle (no
Docker/process control) — the user is responsible for launching the
right server with the right KV cache config before running REFRACT.

For ``run_kld`` (which needs logprobs from BOTH the reference and
candidate KV configs) the user must run two SGLang servers
simultaneously, one per config, and point the backend at both via
``REFRACT_SGLANG_REF_URL`` and ``REFRACT_SGLANG_CAND_URL``. For other
methods only ``REFRACT_SGLANG_URL`` is needed.

Env knobs
---------

  REFRACT_SGLANG_URL       single-server endpoint (default
                           http://127.0.0.1:30000)
  REFRACT_SGLANG_REF_URL   reference-config endpoint for run_kld
  REFRACT_SGLANG_CAND_URL  candidate-config endpoint for run_kld
  REFRACT_SGLANG_TIMEOUT   per-request timeout in seconds (default 600)
  REFRACT_SGLANG_KLD_TOPK  top-K for prompt logprobs during KLD
                           (default 64)

KV-config translation
---------------------

The kv_config_str is informational only — SGLang's KV dtype is set at
server launch, not per-request. The backend warns when the requested
ctk/ctv differs from what the server is running. For TurboQuant: SGLang
doesn't have a TurboQuant KV path, so any ``turbo*`` ctk/ctv raises
BackendCapabilityError.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .base import (
    Backend,
    BackendCapabilityError,
    CompletionResult,
    KLDResult,
    TrajectoryResult,
    approximate_topk_kl,
)

_SUPPORTED_KV: set[tuple[str, str]] = {
    ("f16", "f16"),
    ("bf16", "bf16"),
    ("q8_0", "q8_0"),  # mapped to fp8_e4m3 at server-launch time
}


def _validate_kv_str(kv_str: str) -> tuple[str, str]:
    parts = dict(p.split("=", 1) for p in kv_str.split(",") if "=" in p)
    ctk = parts.get("ctk", "f16").lower()
    ctv = parts.get("ctv", "f16").lower()
    if (ctk, ctv) not in _SUPPORTED_KV:
        raise BackendCapabilityError(
            f"SGLang has no TurboQuant KV path. ctk={ctk}, ctv={ctv} not "
            f"representable. Supported: f16/f16, bf16/bf16, q8_0/q8_0. "
            f"Use --backend llamacpp or --backend vllm for TurboQuant."
        )
    return ctk, ctv


def _url(env_name: str, default: str = "http://127.0.0.1:30000") -> str:
    return os.environ.get(env_name, default).rstrip("/")


def _timeout() -> float:
    return float(os.environ.get("REFRACT_SGLANG_TIMEOUT", "600"))


def _post(url: str, path: str, body: dict, *, timeout_s: float) -> dict:
    """POST a JSON body to a SGLang server endpoint.

    Lazy-imports requests so importing this backend without it doesn't fail.
    Raises BackendCapabilityError on connection refused / timeout with an
    actionable message about the server URL.
    """
    import requests

    try:
        r = requests.post(url + path, json=body, timeout=timeout_s)
    except requests.exceptions.ConnectionError as e:
        raise BackendCapabilityError(
            f"SGLang server unreachable at {url}{path}: {e}. "
            f"Set REFRACT_SGLANG_URL or launch a server."
        ) from e
    if r.status_code != 200:
        raise BackendCapabilityError(
            f"SGLang {path} returned {r.status_code}: {r.text[:300]}"
        )
    return r.json()


def _model_id(url: str) -> str:
    """Look up the served model id (used as the `model` field in some endpoints)."""
    import requests

    try:
        r = requests.get(url + "/v1/models", timeout=30)
        data = r.json().get("data", [])
        if data:
            return data[0]["id"]
    except Exception:
        pass
    return ""


@lru_cache(maxsize=4)
def _load_tokenizer(model_id: str):
    """Load the model tokenizer used to build identical SGLang prompts."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        raise BackendCapabilityError(
            f"Could not load tokenizer for {model_id!r}: {e}. Install the "
            "refract-sglang extra and ensure the model/tokenizer is locally "
            "available or accessible from Hugging Face."
        ) from e


def _prompt_token_ids(
    model: Path,
    url: str,
    prompt: str,
    *,
    system: Optional[str],
    apply_template: bool,
) -> list[int]:
    """Build the exact prompt IDs used by completion and trajectory paths."""
    if apply_template:
        tokenizer = _load_tokenizer(model.as_posix())
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        except Exception as e:
            raise BackendCapabilityError(
                f"Tokenizer for {model} could not apply its chat template: {e}"
            ) from e
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return [int(token_id) for token_id in ids]

    body = {"prompt": prompt, "model": _model_id(url) or model.as_posix()}
    response = _post(url, "/tokenize", body, timeout_s=120.0)
    ids = (
        response[0]["tokens"]
        if isinstance(response, list)
        else response.get("tokens", [])
    )
    return [int(token_id) for token_id in ids]


class SGLangBackend(Backend):
    name = "sglang"

    def run_completion(
        self,
        *,
        model: Path,
        prompt: str,
        kv_config_str: str,
        n_predict: int = 128,
        ctx: int = 512,
        n_gpu_layers: int = 99,  # ignored
        seed: int = 42,
        temperature: float = 0.0,
        timeout: float = 300.0,
        apply_chat_template: bool = True,
        system: Optional[str] = None,
        reasoning: str = "off",
    ) -> CompletionResult:
        _validate_kv_str(kv_config_str)
        url = _url("REFRACT_SGLANG_URL")
        ids = _prompt_token_ids(
            model,
            url,
            prompt,
            system=system,
            apply_template=apply_chat_template,
        )
        body = {
            "input_ids": ids,
            "sampling_params": {
                "max_new_tokens": n_predict,
                "temperature": temperature,
                "seed": seed if temperature > 0 else None,
            },
        }
        j = _post(url, "/generate", body, timeout_s=timeout)
        text = j.get("text", "")
        n_tok = j.get("meta_info", {}).get("completion_tokens", 0)
        return CompletionResult(text=text, n_tokens=n_tok, metadata={"url": url})

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
        _validate_kv_str(kv_config_str)
        url = _url("REFRACT_SGLANG_URL")
        # Use the same locally templated prompt IDs as ``run_completion``.
        ids = _prompt_token_ids(
            model,
            url,
            prompt,
            system=system,
            apply_template=apply_chat_template,
        )
        gen_body = {
            "input_ids": ids,
            "sampling_params": {
                "max_new_tokens": n_predict,
                "temperature": temperature,
                "seed": seed if temperature > 0 else None,
            },
            "return_logprob": True,
        }
        j = _post(url, "/generate", gen_body, timeout_s=timeout)
        otp = j.get("meta_info", {}).get("output_token_logprobs", []) or []
        # entries: [logprob, token_id, ?]
        token_ids = [int(e[1]) for e in otp if e and len(e) >= 2 and e[1] is not None]
        return TrajectoryResult(
            token_ids=token_ids,
            metadata={"url": url, "text": j.get("text", "")},
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
        _validate_kv_str(ref_kv_str)
        _validate_kv_str(cand_kv_str)
        ref_url = os.environ.get("REFRACT_SGLANG_REF_URL")
        cand_url = os.environ.get("REFRACT_SGLANG_CAND_URL")
        if not ref_url or not cand_url:
            raise BackendCapabilityError(
                "run_kld on SGLang requires two servers running concurrently "
                "with different KV configs. Set REFRACT_SGLANG_REF_URL and "
                "REFRACT_SGLANG_CAND_URL, or use --backend vllm / --backend "
                "llamacpp which can hot-swap KV configs in-process."
            )
        ref_url = ref_url.rstrip("/")
        cand_url = cand_url.rstrip("/")
        topk = int(os.environ.get("REFRACT_SGLANG_KLD_TOPK", "64"))

        # Tokenize via the reference server (assumes both servers serve the
        # same model and tokenizer).
        text = Path(corpus).read_text(encoding="utf-8", errors="replace")
        tok_body = {"prompt": text, "model": _model_id(ref_url) or str(model)}
        tok_j = _post(ref_url, "/tokenize", tok_body, timeout_s=300.0)
        ids = tok_j[0]["tokens"] if isinstance(tok_j, list) else tok_j.get("tokens", [])
        # SGLang reserves a few tokens internally; leave headroom on chunk_len
        chunk_len = ctx - 8
        slices = [
            ids[i : i + chunk_len] for i in range(0, len(ids) - chunk_len, chunk_len)
        ][:chunks]
        if not slices:
            raise BackendCapabilityError(
                f"corpus too short for ctx={ctx}, chunks={chunks} "
                f"(have {len(ids)} tokens)"
            )

        def _run(url: str) -> list[list[dict[int, float]]]:
            res: list[list[dict[int, float]]] = []
            for ch in slices:
                j = _post(
                    url,
                    "/generate",
                    {
                        "input_ids": ch,
                        "sampling_params": {
                            "max_new_tokens": 1,
                            "temperature": 0.0,
                        },
                        "return_logprob": True,
                        "logprob_start_len": 0,
                        "top_logprobs_num": topk,
                    },
                    timeout_s=_timeout(),
                )
                pl = j.get("meta_info", {}).get("input_token_top_logprobs", [])
                # Format: list[per-position top-K] where each is
                # list[[logprob, token_id, text_or_null]] or None
                pos: list[dict[int, float]] = []
                for entry in pl:
                    if not entry:
                        pos.append({})
                    else:
                        pos.append(
                            {
                                int(e[1]): float(e[0])
                                for e in entry
                                if e and e[0] is not None
                            }
                        )
                res.append(pos)
            return res

        ref_logp = _run(ref_url)
        cand_logp = _run(cand_url)

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
                "ref_url": ref_url,
                "cand_url": cand_url,
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
        url = _url("REFRACT_SGLANG_URL")
        body = {"prompt": text, "model": _model_id(url) or str(model)}
        j = _post(url, "/tokenize", body, timeout_s=timeout)
        if isinstance(j, list):
            return j[0]["tokens"]
        return j.get("tokens", [])

    def model_metadata(self, *, model: Path) -> dict:
        url = _url("REFRACT_SGLANG_URL")
        served = _model_id(url)
        return {
            "backend": self.name,
            "model": str(model),
            "sglang_url": url,
            "served_model_id": served,
        }
