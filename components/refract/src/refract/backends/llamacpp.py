"""llama.cpp backend for REFRACT.

Thin adapter over the legacy ``refract.runner`` functions, which already
implement subprocess wrappers around the four llama.cpp binaries:
  - llama-cli           (run_completion)
  - llama-completion    (run_completion_trajectory; needs REFRACT v0.1.4 patch)
  - llama-perplexity    (run_perplexity_kld[_base])
  - llama-tokenize      (tokenize_to_ids)

Configurable via env var ``LLAMA_CPP_BIN_DIR`` (default
``~/local_llms/llama.cpp/build-test/bin``).

We delegate to runner.* rather than reimplement so the existing v0.1.x
regression tests (which monkeypatch ``runner._bin`` and
``subprocess.run``) continue to work without modification.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .base import Backend, CompletionResult, KLDResult, TrajectoryResult


class LlamaCppBackend(Backend):
    name = "llamacpp"

    # ---------------------------------------------------------------- run_completion
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
        from ..runner import KVConfig, run_completion as _rc
        kv = KVConfig.parse(kv_config_str)
        text, meta = _rc(
            model=model, prompt=prompt, kv=kv,
            n_predict=n_predict, ctx=ctx, n_gpu_layers=n_gpu_layers,
            seed=seed, temperature=temperature, timeout=timeout,
            apply_chat_template=apply_chat_template,
            system=system, reasoning=reasoning,
        )
        return CompletionResult(text=text, n_tokens=0, metadata=meta)

    # ---------------------------------------------------------- run_completion_trajectory
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
        from ..runner import KVConfig, run_completion_trajectory as _rct
        kv = KVConfig.parse(kv_config_str)
        token_ids, meta = _rct(
            model=model, prompt=prompt, kv=kv,
            n_predict=n_predict, ctx=ctx, n_gpu_layers=n_gpu_layers,
            seed=seed, temperature=temperature, timeout=timeout,
            apply_chat_template=apply_chat_template, system=system,
        )
        return TrajectoryResult(token_ids=token_ids, metadata=meta)

    # ---------------------------------------------------------------- run_kld
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
        from ..runner import (
            KVConfig, run_perplexity_kld, run_perplexity_kld_base,
            corpus_identity, write_corpus_sidecar,
        )
        ref_kv = KVConfig.parse(ref_kv_str)
        cand_kv = KVConfig.parse(cand_kv_str)
        fd, base_path = tempfile.mkstemp(prefix="refract-kldbase-", suffix=".bin")
        os.close(fd)
        os.unlink(base_path)
        base_path_p = Path(base_path)
        try:
            run_perplexity_kld_base(
                model=model, corpus=corpus, kv=ref_kv,
                base_path=base_path_p, chunks=chunks, ctx=ctx,
                n_gpu_layers=n_gpu_layers,
            )
            write_corpus_sidecar(base_path_p, corpus)
            scored = run_perplexity_kld(
                model=model, corpus=corpus, kv=cand_kv,
                base_path=base_path_p, chunks=chunks, ctx=ctx,
                n_gpu_layers=n_gpu_layers,
            )
            return KLDResult(
                mean_kld=scored["mean_kld"],
                ppl=scored.get("ppl"),
                rms_dp_pct=scored.get("rms_dp_pct"),
                same_topp_pct=scored.get("same_topp_pct"),
                chunks=chunks, ctx=ctx,
                metadata={
                    "base_path": str(base_path_p),
                    "corpus": corpus_identity(corpus),
                },
            )
        finally:
            for p in (base_path_p,):
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass

    # ---------------------------------------------------------------- tokenize_to_ids
    def tokenize_to_ids(
        self,
        *,
        model: Path,
        text: str,
        timeout: float = 120.0,
    ) -> list[int]:
        from ..runner import tokenize_to_ids as _tti
        return _tti(model=model, text=text, timeout=timeout)

    # ---------------------------------------------------------------- model_metadata
    def model_metadata(self, *, model: Path) -> dict:
        from ..runner import DEFAULT_BIN_DIR
        commit = None
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(DEFAULT_BIN_DIR.parent.parent),
                capture_output=True, text=True, timeout=5,
            ).stdout.strip() or None
        except Exception:
            pass
        return {
            "backend": self.name,
            "model": str(model),
            "llama_cpp_bin_dir": str(DEFAULT_BIN_DIR),
            "llama_cpp_commit": commit,
        }
