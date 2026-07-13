"""REFRACT command-line interface.

Usage:
    python3 -m refract.cli score \\
        --model MODEL.gguf \\
        --reference "ctk=f16,ctv=f16" \\
        --candidate "ctk=q8_0,ctv=turbo4,attn_rot_v=0" \\
        --chunks 32 -c 512 -ngl 99
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .axes.gtm import run_gtm
from .axes.kld import run_kld
from .axes.plad import run_plad
from .axes.rniah import run_rniah
from .axes.trajectory import run_trajectory
from .report import json_report, text_report, to_json_string
from .runner import KVConfig
from .score import MIN_FLOOR, composite_score

_BACKEND_CHOICES = ("auto", "llamacpp", "mlx", "vllm", "sglang")


def _configure_utf8_stdio() -> None:
    """Make Unicode-rich help and reports reliable on Windows code pages."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _write_text(path: Path, content: str) -> None:
    """Write a UTF-8 text artifact, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


_SCORE_DESCRIPTION = """\
REFRACT — score a KV-cache quantization config against a fp16-KV reference
on the same model. Returns a 0–100 composite + EXCELLENT/PASS/DEGRADED/FAIL
band, with per-axis bands and one-line plain-English interpretation.

Four axes (A and B are cheap and run by default; C and D are opt-in):

  A — GTM / Trajectory  Greedy Trajectory Match.
                        Greedy-decode N tokens from each prompt under both
                        the reference and candidate KV configs; compare the
                        emitted token IDs. Score = shared prefix divided by
                        the longer reference/candidate trajectory for each
                        prompt. Catches token drift and unilateral early-stop
                        failures.

  B — KLD@D             KL Divergence at the Decoder.
                        Per-token KL between the candidate's next-token
                        distribution and the reference's, averaged across a
                        natural-text corpus. Bit-exact zero on Metal when
                        candidate == reference, so any non-zero is real
                        signal. Catches "distributions silently shift even
                        though argmax stayed the same" failures.

  C — R-NIAH (--full)   Retrieval Needle-In-A-Haystack.
                        Insert a sentinel fact ("the password is X") into a
                        long context at fractional positions; ask the model
                        to retrieve it. Score = 100 * (1 - mean candidate
                        degradation vs reference per (length, position) cell).
                        Catches "scores 99 on KLD@D but fails at 32K context"
                        failures that short-window axes miss.

  D — PLAD (--full)     Perturbation-Locality Aware Drift.
                        Generate anchor completions; perturb each prompt
                        minimally (typo, casing, punctuation, paraphrase);
                        compare candidate's drift vs reference's drift via
                        token edit distance. Catches "works on the demo,
                        breaks on real users with typos" brittleness.

Cost on a 7B Q8 model (Apple Silicon ballpark):
  default  Trajectory + KLD                        ~5-7 min
  --full   adds R-NIAH + PLAD                      ~25-30 min

Composite = harmonic mean of all axes scored. Any single axis being broken
drops the composite hard — the framework is intentionally fail-loud.
"""

_SCORE_EPILOG = """\
Examples:

  Quick go/no-go (default; just trajectory + KLD):
    python3 -m refract.cli score \\
        --model model.gguf \\
        --candidate "ctk=q8_0,ctv=q8_0" \\
        --axis-a trajectory

  Full audit (all four axes):
    python3 -m refract.cli score \\
        --model model.gguf \\
        --candidate "ctk=q8_0,ctv=turbo4" \\
        --axis-a trajectory \\
        --full \\
        --rniah-ctx-max 16384 \\
        --json-out report.json

  Borderline result triage (start cheap, add R-NIAH only if you need it):
    # First pass: composite=72 DEGRADED → see which axis dragged it down
    python3 -m refract.cli score ... (default)
    # If KLD is fine but trajectory is bad → don't run R-NIAH; the issue is
    #   short-context distribution drift, not long-context retrieval.
    # If KLD is fine AND trajectory is fine → run --axis-rniah; long-context
    #   retrieval is the surface most likely to be the hidden problem.

KV config syntax: comma-separated key=value pairs, e.g. "ctk=q8_0,ctv=turbo4".
Recognised keys: ctk, ctv, attn_rot_k, attn_rot_v, attn_rot_disable.
"""


_REFRACT_CACHE = Path.home() / ".cache" / "refract"
_WIKITEXT_2_URL = (
    "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
)


def _ensure_wikitext_2(cache_dir: Path = _REFRACT_CACHE, silent: bool = False) -> Path:
    """Make sure wikitext-2-raw is downloaded + extracted under
    ``cache_dir/wikitext-2-raw/``. Returns that directory.

    Idempotent: re-running is a no-op when files already exist.

    Network: ~10 MB single zip, ~30 s on a typical home connection.
    No third-party deps; uses stdlib urllib + zipfile.
    """
    import urllib.request
    import zipfile

    target = cache_dir / "wikitext-2-raw"
    test_p = target / "wiki.test.raw"
    train_p = target / "wiki.train.raw"
    if test_p.is_file() and train_p.is_file():
        return target

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "wikitext-2-raw-v1.zip"
    if not silent:
        print(f"Downloading wikitext-2-raw (~10MB) → {target} ...")
    urllib.request.urlretrieve(_WIKITEXT_2_URL, zip_path)
    if not silent:
        print("Unzipping ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    try:
        zip_path.unlink()
    except OSError:
        pass
    if not test_p.is_file() or not train_p.is_file():
        raise RuntimeError(
            f"Wikitext-2 unzip didn't produce expected files at {target}. "
            f"Inspect {cache_dir} or fetch manually."
        )
    if not silent:
        print(f"✓ Cached at {target}")
    return target


def _resolve_default_paths(args, *, need_corpus: bool, need_haystack: bool):
    """Auto-resolve --corpus and --rniah-haystack from the wikitext-2 cache.

    If a required cached file is missing and ``--no-auto-fetch`` is not set,
    this triggers a one-time wikitext-2 download. With ``--no-auto-fetch``,
    it raises with a clear remediation message.
    """
    if not (need_corpus or need_haystack):
        return
    have_corpus = bool(getattr(args, "corpus", None))
    have_haystack = bool(getattr(args, "rniah_haystack", None))
    if (need_corpus and have_corpus) and (not need_haystack or have_haystack):
        return
    no_fetch = bool(getattr(args, "no_auto_fetch", False))
    target = _REFRACT_CACHE / "wikitext-2-raw"
    default_corpus = target / "wiki.test.raw"
    default_haystack = target / "wiki.train.raw"
    missing_corpus = need_corpus and not have_corpus and not default_corpus.is_file()
    missing_haystack = (
        need_haystack and not have_haystack and not default_haystack.is_file()
    )
    if missing_corpus or missing_haystack:
        if no_fetch:
            missing_flags = []
            if missing_corpus:
                missing_flags.append("--corpus")
            if missing_haystack:
                missing_flags.append("--rniah-haystack")
            raise SystemExit(
                f"Missing cached default for {' and '.join(missing_flags)} "
                f"while --no-auto-fetch is set. Pass the missing path "
                f"explicitly, run `refract fetch` ahead of time, or omit "
                f"--no-auto-fetch to download wikitext-2-raw (~10MB)."
            )
        _ensure_wikitext_2(cache_dir=_REFRACT_CACHE)
    if need_corpus and not have_corpus:
        args.corpus = default_corpus
        print(f"  using cached corpus  : {args.corpus}")
    if need_haystack and not have_haystack:
        args.rniah_haystack = default_haystack
        print(f"  using cached haystack: {args.rniah_haystack}")


def _resolve_default_prompts(args) -> bool:
    """Resolve the bundled prompt set unless the user supplied a path."""
    if getattr(args, "prompts", None) is not None:
        return True
    try:
        import importlib.resources

        resource = importlib.resources.files("refract").joinpath("prompts/v0.1.jsonl")
        if not resource.is_file():
            raise FileNotFoundError(resource)
        args.prompts = Path(str(resource))
    except Exception as e:
        print(
            f"ERROR: --prompts not given and bundled prompts not found "
            f"({e}). Pass --prompts /path/to/prompts.jsonl explicitly."
        )
        return False
    return True


_PROMPTS_HELP = (
    "Path to JSONL prompts file. If omitted, REFRACT uses the bundled "
    "prompt set shipped in the wheel (30 CC0 prompts; see prompts/README.md)."
)
_CORPUS_HELP = (
    "Path to plain-text corpus for KLD axis. If omitted, REFRACT "
    "auto-downloads wikitext-2-raw (~10MB) to ~/.cache/refract/ and uses "
    "wiki.test.raw. Pass --no-auto-fetch to disable downloading."
)
_NO_AUTO_FETCH_HELP = (
    "Disable downloading wikitext-2-raw. Explicit paths and files already "
    "present in ~/.cache/refract/ are still accepted."
)


def _positive_int(value: str) -> int:
    """Argparse type for counts that must be at least one."""
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from e
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _add_score_parser(sub):
    import argparse as _ap

    p = sub.add_parser(
        "score",
        help="Score a candidate KV config vs the fp16-KV reference (4 axes).",
        description=_SCORE_DESCRIPTION,
        epilog=_SCORE_EPILOG,
        formatter_class=_ap.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        type=Path,
        help="GGUF/MLX model path or Hugging Face model ID.",
    )
    p.add_argument(
        "--reference",
        default="ctk=f16,ctv=f16",
        help="Reference KV config (default: ctk=f16,ctv=f16).",
    )
    p.add_argument("--candidate", required=True, help="Candidate KV config to score.")
    p.add_argument("--prompts", type=Path, default=None, help=_PROMPTS_HELP)
    p.add_argument("--corpus", type=Path, default=None, help=_CORPUS_HELP)
    p.add_argument("--no-auto-fetch", action="store_true", help=_NO_AUTO_FETCH_HELP)
    p.add_argument(
        "--chunks",
        type=int,
        default=32,
        help="--chunks for llama-perplexity (default: 32).",
    )
    p.add_argument(
        "-c", "--ctx", type=int, default=512, help="Context size (default: 512)."
    )
    p.add_argument(
        "-ngl", "--n-gpu-layers", type=int, default=99, help="-ngl flag (default: 99)."
    )
    p.add_argument(
        "--n-predict",
        type=int,
        default=128,
        help="Tokens to greedy-decode per GTM prompt (default: 128).",
    )
    p.add_argument("--seed", type=int, default=42, help="Greedy seed (default: 42).")
    p.add_argument(
        "--measure-floor",
        action="store_true",
        help=f"Measure REFRACT(ref, ref) and abort if < {MIN_FLOOR:.1f}.",
    )
    p.add_argument(
        "--skip-gtm",
        action="store_true",
        help="Skip Axis A. Composite uses KLD only (debug).",
    )
    p.add_argument(
        "--skip-kld",
        action="store_true",
        help="Skip Axis B. Composite uses GTM only (debug).",
    )
    p.add_argument(
        "--axis-a",
        choices=["gtm", "trajectory"],
        default="trajectory",
        help="Axis A implementation. 'trajectory' (default, v0.1.4+) "
        "captures token IDs at decode time via the patched "
        "llama-completion binary — no detokenize round-trip. "
        "'gtm' (deprecated, v0.1.x) compares retokenized text "
        "and has known unit-mismatch artifacts.",
    )
    p.add_argument(
        "--backend",
        choices=_BACKEND_CHOICES,
        default="auto",
        help="Inference backend. 'auto' (default) infers from "
        "model metadata: .gguf → llamacpp; a recognizable "
        "quantized MLX directory → mlx; otherwise → vllm. "
        "Use --backend for ambiguous directories. "
        "Override via REFRACT_BACKEND env var.",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Enable all v0.2 axes (R-NIAH + PLAD). Equivalent to "
        "passing --axis-rniah --axis-plad. R-NIAH uses the "
        "cached/auto-downloaded haystack and defaults its "
        "context ceiling to --rniah-up-to.",
    )
    p.add_argument(
        "--axis-rniah",
        action="store_true",
        help="Enable Axis C (R-NIAH, v0.2). Probes long-context "
        "retrieval degradation. Uses --rniah-haystack or "
        "auto-resolves it from cache when omitted. Cost: "
        "~10–15 min on a 7B Q8 at 16K, scaling roughly "
        "linearly with max length.",
    )
    p.add_argument(
        "--rniah-haystack",
        type=Path,
        default=None,
        help="Path to long-text haystack corpus for R-NIAH. "
        "Auto-resolved from ~/.cache/refract/ when omitted "
        "(wiki.train.raw, fits cells up to ~16K cleanly).",
    )
    p.add_argument(
        "--rniah-up-to",
        type=int,
        default=16384,
        help="R-NIAH max context length to test. Lengths are "
        "auto-generated as a doubling step-up starting at "
        "4096 (4K, 8K, 16K, 32K, ... up to this value). "
        "Default: 16384 (4K, 8K, 16K). Pass 32768 / 65536 / "
        "131072 for deeper long-context audits. Cells are "
        "always run at positions 0.10/0.50/0.90 unless "
        "overridden via --rniah-positions.",
    )
    p.add_argument(
        "--rniah-ctx-max",
        type=int,
        default=None,
        help="(Power-user) hard ceiling for R-NIAH cells. Cells "
        "longer than this are skipped. Defaults to "
        "--rniah-up-to. Useful when the model cannot "
        "actually do its nominal max context.",
    )
    p.add_argument(
        "--rniah-lengths",
        type=str,
        default=None,
        help="(Power-user) comma-separated R-NIAH lengths, "
        "overrides --rniah-up-to. Default: synthesised "
        "from --rniah-up-to.",
    )
    p.add_argument(
        "--rniah-positions",
        type=str,
        default=None,
        help="Comma-separated R-NIAH needle positions as "
        "fractions of length. Default: 0.10,0.50,0.90.",
    )
    p.add_argument(
        "--rniah-trials", type=int, default=1, help="Trials per cell. Default: 1."
    )
    p.add_argument(
        "--axis-plad",
        action="store_true",
        help="Enable Axis D (PLAD, v0.2). Probes brittleness "
        "to small prompt perturbations. Reuses --prompts. "
        "Cost: ~5–7 min on a 7B Q8 "
        "(prompts × (1 + n_perturbations) × 2 generations).",
    )
    p.add_argument(
        "--json-out", type=Path, default=None, help="Path to write the JSON report to."
    )
    p.add_argument(
        "--html-out",
        type=Path,
        default=None,
        help="Path to write a self-contained HTML report (single "
        "file with inline CSS, no JS deps). Includes hardware "
        "info, model params, and the exact repro command.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress per-prompt progress output.",
    )
    return p


def _run_score(args) -> int:
    from . import __version__
    from .backends import auto_backend, get_backend
    from .runner import set_active_backend

    if args.measure_floor and args.skip_gtm and args.skip_kld:
        print(
            "ERROR: --measure-floor requires at least one active default "
            "axis; do not combine it with both --skip-gtm and --skip-kld."
        )
        return 2

    if not _resolve_default_prompts(args):
        return 2

    ref_kv = KVConfig.parse(args.reference)
    cand_kv = KVConfig.parse(args.candidate)

    # v0.3.1: select backend per --backend flag (or auto-detect from model path).
    if args.backend == "auto":
        backend = auto_backend(args.model)
    else:
        backend = get_backend(args.backend)
    set_active_backend(backend)

    # --full is sugar for the two opt-in v0.2 axes.
    if args.full:
        args.axis_rniah = True
        args.axis_plad = True

    # v0.3.2: auto-resolve corpus + haystack from ~/.cache/refract/ when
    # the user didn't pass them. Triggers a one-time wikitext-2-raw fetch
    # if the cache is empty and --no-auto-fetch isn't set.
    _resolve_default_paths(
        args,
        need_corpus=not args.skip_kld,
        need_haystack=args.axis_rniah,
    )

    # Cost hint up front so the user knows what they're committing to.
    cost_axes = ["A (~2 min)", "B (~5 min)"]
    if args.axis_rniah:
        cost_axes.append("C R-NIAH (~10-15 min)")
    if args.axis_plad:
        cost_axes.append("D PLAD (~5-7 min)")
    cost_hint = " + ".join(cost_axes)

    print(f"REFRACT v{__version__}")
    print(f"  model     : {args.model}")
    print(f"  reference : {ref_kv.label()}")
    print(f"  candidate : {cand_kv.label()}")
    print(f"  backend   : {backend.name}")
    print(f"  axes      : {cost_hint}  (estimates for a 7B Q8 model on Apple Silicon)")

    # v0.3.1: thinking-mode auto-detect at startup. Reasoning is auto-disabled
    # via -rea off in run_completion when backend supports it; we surface the
    # detection so the user knows what we did and why.
    thinking_detected = False
    thinking_markers: list[str] = []
    try:
        thinking_detected, thinking_markers = backend.detect_thinking_mode(
            model=args.model,
            timeout=30,
        )
    except Exception as e:
        print(f"  thinking  : probe failed ({e}); assuming non-thinking")
    if thinking_detected:
        print(
            f"  thinking  : DETECTED (markers: {thinking_markers}) — "
            f"reasoning disabled, n_predict-aware fallback active"
        )
    else:
        print("  thinking  : not detected")
    print()

    # ---- Floor check ------------------------------------------------------
    floor_score = None
    if args.measure_floor:
        print("Measuring noise floor: REFRACT(ref, ref) ...")
        floor_axis_a = None
        floor_axis_a_label = args.axis_a
        if not args.skip_gtm:
            floor_axis_runner = (
                run_trajectory if args.axis_a == "trajectory" else run_gtm
            )
            floor_axis_a = floor_axis_runner(
                model=args.model,
                reference_kv=ref_kv,
                candidate_kv=ref_kv,
                prompts_path=args.prompts,
                n_predict=args.n_predict,
                ctx=args.ctx,
                n_gpu_layers=args.n_gpu_layers,
                seed=args.seed,
                progress=not args.no_progress,
            )

        floor_kld = None
        if not args.skip_kld:
            floor_kld = run_kld(
                model=args.model,
                corpus=args.corpus,
                reference_kv=ref_kv,
                candidate_kv=ref_kv,
                chunks=args.chunks,
                ctx=args.ctx,
                n_gpu_layers=args.n_gpu_layers,
                progress=not args.no_progress,
            )

        floor_composite = composite_score(
            floor_axis_a.score if floor_axis_a is not None else None,
            floor_kld.score if floor_kld is not None else None,
        )
        floor_score = floor_composite.composite
        floor_details = []
        if floor_axis_a is not None:
            floor_details.append(f"{floor_axis_a_label}={floor_axis_a.score:.2f}")
        if floor_kld is not None:
            floor_details.extend(
                [
                    f"kld={floor_kld.score:.2f}",
                    f"kld nats={floor_kld.mean_kld:.6f}",
                ]
            )
        print(f"  floor: {floor_score:.2f} ({', '.join(floor_details)})")
        if floor_score < MIN_FLOOR:
            print()
            print(f"ERROR: noise floor {floor_score:.2f} < {MIN_FLOOR}.")
            print("An active reference axis is non-deterministic on this build.")
            print("Scores against this reference cannot be trusted. Aborting.")
            return 2
        # A composite-level floor can pass when KLD is bit-exact even if Axis A
        # is broken. Require every selected ref-vs-ref trajectory to be
        # identical instead of inferring identity from aggregate lengths.
        if floor_axis_a is not None and abs(floor_axis_a.full_match_rate - 1.0) > 1e-9:
            print()
            print(
                f"ERROR: {floor_axis_a_label} ref-vs-ref full_match_rate = "
                f"{floor_axis_a.full_match_rate:.6f} (expected 1.0)."
            )
            print(
                f"{floor_axis_a_label} ref-vs-ref isn't token-identical; "
                f"the runner or backend is non-deterministic. Aborting."
            )
            return 2
        print()

    # ---- Axis A (GTM or Trajectory) ---------------------------------------
    if args.skip_gtm:
        gtm = _stub_gtm()
    else:
        if args.axis_a == "trajectory":
            print("Running Axis A (Trajectory, v0.1.4)...")
            gtm = run_trajectory(
                model=args.model,
                reference_kv=ref_kv,
                candidate_kv=cand_kv,
                prompts_path=args.prompts,
                n_predict=args.n_predict,
                ctx=args.ctx,
                n_gpu_layers=args.n_gpu_layers,
                seed=args.seed,
                progress=not args.no_progress,
            )
            print(f"  Trajectory score: {gtm.score:.2f}")
        else:
            print("Running Axis A (GTM, v0.1.x)...")
            gtm = run_gtm(
                model=args.model,
                reference_kv=ref_kv,
                candidate_kv=cand_kv,
                prompts_path=args.prompts,
                n_predict=args.n_predict,
                ctx=args.ctx,
                n_gpu_layers=args.n_gpu_layers,
                seed=args.seed,
                progress=not args.no_progress,
            )
            print(f"  GTM score: {gtm.score:.2f}")

    # ---- KLD --------------------------------------------------------------
    if args.skip_kld:
        kld = _stub_kld(args.chunks, args.ctx)
    else:
        print("Running Axis B (KLD@D, corpus proxy)...")
        kld = run_kld(
            model=args.model,
            corpus=args.corpus,
            reference_kv=ref_kv,
            candidate_kv=cand_kv,
            chunks=args.chunks,
            ctx=args.ctx,
            n_gpu_layers=args.n_gpu_layers,
            progress=not args.no_progress,
        )
        approximate = not kld.metadata.get("full_vocabulary", True)
        label = "approximate KLD" if approximate else "mean KLD"
        print(f"  KLD score: {kld.score:.2f}  ({label} = {kld.mean_kld:.6f} nats)")

    # ---- R-NIAH (v0.2 opt-in) ---------------------------------------------
    rniah = None
    if args.axis_rniah:
        if args.rniah_haystack is None:
            print(
                "ERROR: --axis-rniah requires --rniah-haystack "
                "(or run `refract fetch` first to populate the cache)."
            )
            return 2
        # v0.3.2: synthesize lengths from --rniah-up-to (doubling step-up
        # from 4K) when --rniah-lengths isn't set. Default --rniah-ctx-max
        # to --rniah-up-to so the user only has one knob to think about.
        if args.rniah_lengths:
            lengths = tuple(int(x) for x in args.rniah_lengths.split(","))
        else:
            up_to = args.rniah_up_to
            n = 4096
            synth: list[int] = []
            while n <= up_to:
                synth.append(n)
                n *= 2
            if not synth:
                synth = [up_to]
            lengths = tuple(synth)
        if args.rniah_ctx_max is None:
            args.rniah_ctx_max = max(lengths)
        positions = (
            tuple(float(x) for x in args.rniah_positions.split(","))
            if args.rniah_positions
            else None
        )
        kwargs: dict = {
            "model": args.model,
            "haystack_corpus": args.rniah_haystack,
            "reference_kv": ref_kv,
            "candidate_kv": cand_kv,
            "ctx_max": args.rniah_ctx_max,
            "n_trials": args.rniah_trials,
            "n_gpu_layers": args.n_gpu_layers,
            "seed": args.seed,
            "progress": not args.no_progress,
        }
        if lengths is not None:
            kwargs["lengths"] = lengths
        if positions is not None:
            kwargs["positions"] = positions
        print("Running Axis C (R-NIAH, v0.2)...")
        rniah = run_rniah(**kwargs)
        print(f"  R-NIAH score: {rniah.score:.2f}  ({rniah.n_cells} cells)")

    # ---- PLAD (v0.2 opt-in) -----------------------------------------------
    plad = None
    if args.axis_plad:
        print("Running Axis D (PLAD, v0.2)...")
        plad = run_plad(
            model=args.model,
            prompts_path=args.prompts,
            reference_kv=ref_kv,
            candidate_kv=cand_kv,
            n_predict=args.n_predict,
            ctx=args.ctx,
            n_gpu_layers=args.n_gpu_layers,
            seed=args.seed,
            progress=not args.no_progress,
        )
        print(f"  PLAD score: {plad.score:.2f}")

    # ---- Composite --------------------------------------------------------
    # Skipped axes contribute None to the composite (dropped before the
    # harmonic mean) rather than a stub 100, which would inflate the
    # composite and read as EXCELLENT in the report.
    gtm_for_composite = None if args.skip_gtm else gtm.score
    kld_for_composite = None if args.skip_kld else kld.score
    rniah_for_composite = None
    if rniah is not None:
        if rniah.confidence == "ok":
            rniah_for_composite = rniah.score
        else:
            print(
                "  WARNING: R-NIAH excluded from the composite because the "
                f"fp16 reference retrieved only {rniah.base_accuracy:.1%} "
                "of cells. The raw result remains in the report."
            )
    composite = composite_score(
        gtm_for_composite,
        kld_for_composite,
        rniah_score=rniah_for_composite,
        plad_score=plad.score if plad else None,
        floor_score=floor_score,
    )
    if rniah is not None and rniah_for_composite is None:
        composite.notes.append(
            "R-NIAH was excluded from the composite: fp16 reference "
            f"accuracy {rniah.base_accuracy:.1%} is below the 20% confidence floor."
        )

    print()
    print(
        text_report(
            model=str(args.model),
            reference_label=ref_kv.label(),
            candidate_label=cand_kv.label(),
            composite=composite,
            gtm=gtm,
            kld=kld,
            rniah=rniah,
            plad=plad,
        )
    )

    rep = None
    if args.json_out or args.html_out:
        rep = json_report(
            model=str(args.model),
            reference_label=ref_kv.label(),
            candidate_label=cand_kv.label(),
            composite=composite,
            gtm=gtm,
            kld=kld,
            rniah=rniah,
            plad=plad,
        )
    if args.json_out:
        assert rep is not None
        _write_text(args.json_out, to_json_string(rep))
        print(f"\nJSON report written to {args.json_out}")
    if args.html_out:
        from .report_html import html_report

        html = html_report(
            model=str(args.model),
            reference_label=ref_kv.label(),
            candidate_label=cand_kv.label(),
            composite=composite,
            gtm=gtm,
            kld=kld,
            rniah=rniah,
            plad=plad,
            raw_json=rep,
        )
        _write_text(args.html_out, html)
        print(f"HTML report written to {args.html_out}")

    return 0


# Stubs for --skip-gtm / --skip-kld dev modes (composite still computes).
def _stub_gtm():
    from .axes.gtm import GTMResult

    return GTMResult(
        score=100.0,
        full_match_rate=1.0,
        median_first_divergence=None,
        mean_prefix_agreement_length=0.0,
        mean_cand_length=0.0,
        mean_ref_length=0.0,
        n_prompts=0,
        n_tokens_each=0,
        per_prompt=[],
    )


def _stub_kld(chunks: int, ctx: int):
    from .axes.kld import KLDResult

    return KLDResult(
        score=100.0,
        mean_kld=0.0,
        ppl=None,
        rms_dp_pct=None,
        same_topp_pct=None,
        base_path="",
        chunks=chunks,
        ctx=ctx,
        is_self_reference=False,
    )


_TOP_DESCRIPTION = """\
REFRACT — REFerence-anchored Robust Acid-test for Compressed Transformers.

A benchmaxx-resistant alternative to corpus PPL for evaluating KV-cache
quantization quality on a model. Replaces "lower PPL = better" — a metric
that can invert sign on instruct-tuned models — with a 4-axis composite
anchored to the fp16-KV reference.

Subcommands:
  score   run REFRACT on a candidate KV config vs the reference

Run `refract score --help` for full options + usage examples.
"""


def _add_selftest_parser(sub):
    p = sub.add_parser(
        "selftest",
        help="Preflight check before running score/repeatability.",
        description=(
            "Verifies the environment is ready for REFRACT: binaries / "
            "imports, required CLI flags (--jinja, REFRACT_TRAJECTORY env), "
            "KV cache types compiled in, model loadable, fp16-vs-fp16 "
            "sanity generation. Bails out with a useful message on any "
            "failure so you don't burn a long run finding out your setup "
            "is broken.\n\n"
            "Tip: pass --model for a real generation probe (~10-30s "
            "depending on model size). Without --model, only static "
            "binary/import checks run (~1s)."
        ),
    )
    p.add_argument(
        "--backend",
        choices=_BACKEND_CHOICES,
        default="auto",
        help="Backend to test. 'auto' picks llamacpp for .gguf "
        "models, recognizable quantized MLX directories, and "
        "vllm otherwise. Override ambiguous paths via "
        "REFRACT_BACKEND env.",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional model to use for a real generation probe. "
        "Accepts local model paths and backend-supported model "
        "IDs. If omitted, only static checks run.",
    )
    return p


def _run_selftest(args) -> int:
    from . import __version__
    from .backends import BackendCapabilityError, auto_backend, get_backend

    print(f"REFRACT v{__version__} selftest")
    print()

    failures: list[str] = []
    warnings: list[str] = []

    # --- Backend selection ---
    if args.backend == "auto":
        if args.model:
            backend = auto_backend(args.model)
        else:
            from .backends.llamacpp import LlamaCppBackend

            backend = LlamaCppBackend()
    else:
        backend = get_backend(args.backend)
    print(f"Backend: {backend.name}")

    # --- Backend-specific binaries ---
    if backend.name == "llamacpp":
        from .runner import DEFAULT_BIN_DIR

        print(f"  bin dir : {DEFAULT_BIN_DIR}")
        for tool in (
            "llama-cli",
            "llama-completion",
            "llama-tokenize",
            "llama-perplexity",
        ):
            p = DEFAULT_BIN_DIR / tool
            if p.exists():
                print(f"  ✓ {tool}")
            else:
                print(f"  ✗ {tool}  (set LLAMA_CPP_BIN_DIR or rebuild)")
                failures.append(f"missing binary: {tool}")
        # Check for --jinja flag support. Capture stderr too — when the
        # binary can't load (e.g. Linux missing libllama.so on PATH, Windows
        # missing the DLL next to the exe), --help fails *before* it can
        # print its help text. Without checking the launch result, an empty
        # stdout would cause a misleading "--jinja missing" diagnosis.
        try:
            import subprocess as _sp

            proc = _sp.run(
                [str(DEFAULT_BIN_DIR / "llama-completion"), "--help"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            help_text = proc.stdout
            launch_failed = proc.returncode != 0 and not help_text.strip()
            stderr_tail = (proc.stderr or "")[-400:]
            if launch_failed:
                print(f"  ✗ llama-completion can't launch (rc={proc.returncode})")
                if stderr_tail:
                    print(f"      stderr: {stderr_tail.strip()}")
                # Linux + Windows shared-library hints
                hint = ""
                if (
                    "cannot open shared object" in stderr_tail
                    or "libllama" in stderr_tail.lower()
                ):
                    hint = (
                        " — Linux: libllama.so/libggml*.so not on the "
                        "loader path. Add the bin dir to LD_LIBRARY_PATH "
                        "or register it via ldconfig."
                    )
                elif (
                    "dll" in stderr_tail.lower() or "0xc0000135" in stderr_tail.lower()
                ):
                    hint = (
                        " — Windows: required DLLs (llama.dll, ggml-*.dll) "
                        "not next to the .exe. Copy them from the build "
                        "dir or add the bin dir to PATH."
                    )
                failures.append(
                    f"llama-completion failed to launch (rc={proc.returncode}){hint}"
                )
            elif "--jinja" in help_text:
                print("  ✓ --jinja chat template flag supported")
            else:
                print("  ✗ --jinja not in llama-completion help")
                failures.append("--jinja missing — chat templates won't apply")
            if "REFRACT_TRAJECTORY" in help_text or "trajectory" in help_text.lower():
                print("  ✓ REFRACT_TRAJECTORY likely supported")
            elif not launch_failed:
                # The patch is env-var triggered, no help-string evidence.
                # Run a probe instead.
                warnings.append(
                    "Could not verify REFRACT_TRAJECTORY support from --help. "
                    "Run a quick trajectory probe to confirm."
                )
        except Exception as e:
            warnings.append(f"Could not probe llama-completion --help: {e}")
    elif backend.name == "mlx":
        try:
            from .backends.mlx import _require_mlx

            _require_mlx()
            print("  ✓ mlx + mlx_lm importable")
        except BackendCapabilityError as e:
            print(f"  ✗ mlx not importable: {e}")
            failures.append(str(e))
    elif backend.name == "vllm":
        try:
            import vllm  # noqa

            print(
                f"  ✓ vllm importable (version: {getattr(vllm, '__version__', 'unknown')})"
            )
        except ImportError:
            print("  ✗ vllm not importable; pip install vllm")
            failures.append("vllm not importable")
    elif backend.name == "sglang":
        try:
            import requests  # noqa: F401
            import transformers  # noqa: F401

            print("  ✓ requests + transformers importable")
        except ImportError as e:
            print("  ✗ SGLang client dependencies missing")
            failures.append(
                f"SGLang dependencies missing ({e}); install "
                "refract-llm[refract-sglang]"
            )

    # --- Model probe (optional) ---
    if args.model:
        requires_local_model = backend.name in {"llamacpp", "mlx"}
        if requires_local_model and not args.model.exists():
            print(f"  ✗ model not found: {args.model}")
            failures.append(f"model missing: {args.model}")
        else:
            print(f"\nProbing model: {args.model.name}")
            gen_ok = False
            try:
                result = backend.run_completion(
                    model=args.model,
                    prompt="What is 2+2?",
                    kv_config_str="ctk=f16,ctv=f16",
                    n_predict=16,
                    ctx=128,
                    seed=42,
                    temperature=0.0,
                    timeout=120,
                )
                preview = (result.text or "").replace("\n", " ")[:80]
                print(f"  ✓ generation works → {preview!r}")
                gen_ok = True
            except Exception as e:
                print(f"  ✗ generation failed: {e}")
                failures.append(f"model generation: {e}")
            # Thinking probe — only run if generation succeeded. If gen
            # failed, the probe uses the same broken path and "no markers
            # detected" would be misleading.
            if gen_ok:
                try:
                    detected, markers = backend.detect_thinking_mode(model=args.model)
                    if detected:
                        print(
                            f"  ℹ thinking-mode detected (markers: {markers}). "
                            "REFRACT will handle this automatically."
                        )
                    else:
                        print("  ✓ no thinking-mode markers (faster runs)")
                except Exception as e:
                    warnings.append(f"thinking-mode probe failed: {e}")

    # --- Summary ---
    print()
    if failures:
        print(f"FAILED  {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 2
    if warnings:
        print(f"OK with {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("OK — selftest passed.")
    print()
    print("You're ready to run `refract score --model ... --candidate ... ...`.")
    return 0


def _add_compare_parser(sub):
    p = sub.add_parser(
        "compare",
        help="Side-by-side comparison of multiple report JSONs.",
    )
    p.add_argument(
        "reports",
        type=Path,
        nargs="+",
        help="Two or more report.json files to compare.",
    )
    return p


def _run_compare(args) -> int:
    import json as _json

    rows = []
    for path in args.reports:
        try:
            d = _json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        rows.append(
            {
                "label": path.stem,
                "composite": d.get("composite"),
                "band": d.get("band"),
                "summary": d.get("summary"),
                "axes": d.get("axes", {}),
                "version": d.get("framework_version"),
                "backend": d.get("environment", {}).get("backend"),
            }
        )
    if not rows:
        print("no reports parsed")
        return 1
    # Print a markdown-style comparison table
    print()
    print(
        f"{'Report':<32} {'Comp':>7} {'Band':<10} {'Traj':>7} {'KLD':>7} {'R-NIAH':>7} {'PLAD':>7}"
    )
    print("-" * 80)
    for r in rows:
        a = r["axes"]

        def fmt(d, k):
            try:
                ax = d[k]
                if ax.get("skipped") or ax.get("score") is None:
                    return "skip"
                return f"{ax['score']:.2f}"
            except Exception:
                return "—"

        comp_val = r["composite"]
        comp_str = (
            f"{comp_val:>7.2f}" if isinstance(comp_val, (int, float)) else f"{'—':>7}"
        )
        band_str = r["band"] if isinstance(r["band"], str) else "—"
        print(
            f"{r['label'][:32]:<32} {comp_str} {band_str:<10} "
            f"{fmt(a, 'gtm'):>7} {fmt(a, 'kld'):>7} "
            f"{fmt(a, 'rniah'):>7} {fmt(a, 'plad'):>7}"
        )
    print()
    return 0


def _add_fetch_parser(sub):
    p = sub.add_parser(
        "fetch",
        help="Download wikitext-2-raw corpus + haystack to ~/.cache/refract/.",
        description=(
            "Pre-fetch the wikitext-2-raw corpus (10MB zip → wiki.test.raw "
            "+ wiki.train.raw) into ~/.cache/refract/. Subsequent score / "
            "repeatability invocations will auto-find these files when "
            "--corpus / --rniah-haystack are omitted. Custom cache "
            "directories require explicit input paths.\n\n"
            "Idempotent: re-running with the cache already populated is a "
            "no-op. Skips any download if files already exist."
        ),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=_REFRACT_CACHE,
        help=f"Override cache location (default: {_REFRACT_CACHE}). "
        "Custom locations are not auto-discovered by score "
        "or repeatability.",
    )
    return p


def _run_fetch(args) -> int:
    target = _ensure_wikitext_2(cache_dir=args.cache_dir)
    print()
    print(f"  test  : {target / 'wiki.test.raw'}")
    print(f"  train : {target / 'wiki.train.raw'}")
    print()
    if args.cache_dir.resolve() == _REFRACT_CACHE.resolve():
        print(
            "`refract score` and `refract repeatability` will auto-find "
            "these when --corpus / --rniah-haystack are omitted."
        )
    else:
        print(
            "Custom cache directories are not auto-discovered. Pass the "
            "paths above via --corpus and --rniah-haystack."
        )
    return 0


def _add_repeatability_parser(sub):
    p = sub.add_parser(
        "repeatability",
        help="Run the same scoring config N times and report the spread.",
        description=(
            "Verifies REFRACT scores are reproducible on the same model + "
            "candidate. Runs ``score`` N times back-to-back, then prints "
            "min/median/max + stdev for the composite and each axis. "
            "Healthy framework: composite stdev ≤ 1.0, per-axis stdev ≤ 2.0 "
            "(R-NIAH may be noisier on n_trials=1). Uses the same bundled "
            "prompts and cached/auto-downloaded corpus defaults as score."
        ),
    )
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--candidate", required=True)
    p.add_argument("--reference", default="ctk=f16,ctv=f16")
    p.add_argument("--prompts", type=Path, default=None, help=_PROMPTS_HELP)
    p.add_argument("--corpus", type=Path, default=None, help=_CORPUS_HELP)
    p.add_argument("--no-auto-fetch", action="store_true", help=_NO_AUTO_FETCH_HELP)
    p.add_argument(
        "--runs",
        type=_positive_int,
        default=4,
        help="Number of repeat runs (default: 4).",
    )
    p.add_argument("--n-predict", type=int, default=50)
    p.add_argument("--ctx", "-c", type=int, default=512)
    p.add_argument("--chunks", type=int, default=32)
    p.add_argument("--n-gpu-layers", "-ngl", type=int, default=99)
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for ALL runs. Identical seed gives upper "
        "bound on reproducibility — variance comes purely "
        "from non-determinism in the engine, not RNG.",
    )
    p.add_argument("--axis-a", choices=["gtm", "trajectory"], default="trajectory")
    p.add_argument(
        "--full", action="store_true", help="Include R-NIAH + PLAD in each repeat run."
    )
    p.add_argument(
        "--rniah-haystack",
        type=Path,
        default=None,
        help="Long-text corpus for R-NIAH. With --full, omitted "
        "paths use cached wiki.train.raw (downloaded when "
        "needed unless --no-auto-fetch is set).",
    )
    p.add_argument("--rniah-ctx-max", type=int, default=None)
    p.add_argument(
        "--rniah-up-to",
        type=int,
        default=16384,
        help="Maximum R-NIAH context when --full is used.",
    )
    p.add_argument("--backend", choices=_BACKEND_CHOICES, default="auto")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for per-run JSON outputs. Default: tmp.",
    )
    return p


def _run_repeatability(args) -> int:
    """Run ``score`` N times and report the spread per-axis."""
    import json as _json
    import statistics as _stats
    import tempfile as _tf

    from . import __version__
    from .backends import auto_backend, get_backend
    from .runner import set_active_backend

    if args.runs < 1:
        print("ERROR: --runs must be at least 1.")
        return 2

    if not _resolve_default_prompts(args):
        return 2
    full = bool(getattr(args, "full", False))
    _resolve_default_paths(
        args,
        need_corpus=True,
        need_haystack=full,
    )

    # Set backend up front so all runs share it
    backend = (
        auto_backend(args.model)
        if args.backend == "auto"
        else get_backend(args.backend)
    )
    set_active_backend(backend)

    out_dir = args.out_dir or Path(_tf.mkdtemp(prefix="refract-repeatability-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"REFRACT v{__version__} repeatability — {args.runs} runs")
    print(f"  model     : {args.model}")
    print(f"  candidate : {args.candidate}")
    print(f"  output    : {out_dir}")
    print()

    composite_scores: list[float] = []
    axis_scores: dict[str, list[float]] = {
        "trajectory": [],
        "kld": [],
        "rniah": [],
        "plad": [],
    }

    # Build a one-shot args namespace for each run
    import argparse as _ap

    base_args = _ap.Namespace(
        model=args.model,
        reference=args.reference,
        candidate=args.candidate,
        prompts=args.prompts,
        corpus=args.corpus,
        chunks=args.chunks,
        ctx=args.ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_predict=args.n_predict,
        seed=args.seed,
        measure_floor=False,
        skip_gtm=False,
        skip_kld=False,
        axis_a=args.axis_a,
        full=full,
        axis_rniah=full,
        rniah_haystack=args.rniah_haystack,
        rniah_ctx_max=args.rniah_ctx_max,
        rniah_up_to=getattr(args, "rniah_up_to", 16384),
        rniah_lengths=None,
        rniah_positions=None,
        rniah_trials=1,
        axis_plad=full,
        json_out=None,
        html_out=None,
        no_progress=True,
        backend=args.backend,
        no_auto_fetch=bool(getattr(args, "no_auto_fetch", False)),
    )

    for i in range(1, args.runs + 1):
        json_path = out_dir / f"run-{i:02d}.json"
        run_args = _ap.Namespace(**vars(base_args))
        run_args.json_out = json_path
        print(f"=== run {i}/{args.runs} → {json_path.name} ===")
        rc = _run_score(run_args)
        if rc != 0:
            print(f"  run {i} failed (rc={rc}); aborting")
            return rc
        try:
            d = _json.loads(json_path.read_text(encoding="utf-8"))
            composite_scores.append(float(d.get("composite", 0)))
            for axis, key in (
                ("trajectory", "gtm"),
                ("kld", "kld"),
                ("rniah", "rniah"),
                ("plad", "plad"),
            ):
                ax = d.get("axes", {}).get(key, {})
                if "score" in ax and isinstance(ax["score"], (int, float)):
                    axis_scores[axis].append(float(ax["score"]))
        except Exception as e:
            print(f"  warning: could not parse {json_path}: {e}")

    # Spread report
    print()
    print("=" * 72)
    print(f"REPEATABILITY ({args.runs} runs)")
    print("=" * 72)

    def _spread(values: list[float]) -> str:
        if not values:
            return "no data"
        if len(values) == 1:
            return f"single run: {values[0]:.2f}"
        return (
            f"min={min(values):.2f}  med={_stats.median(values):.2f}  "
            f"max={max(values):.2f}  stdev={_stats.stdev(values):.2f}  "
            f"range={max(values) - min(values):.2f}"
        )

    print(f"composite  : {_spread(composite_scores)}")
    for axis, vals in axis_scores.items():
        print(f"  {axis:<10}: {_spread(vals)}")

    # Health check
    print()
    if composite_scores:
        if len(composite_scores) > 1:
            cs_stdev = _stats.stdev(composite_scores)
            if cs_stdev <= 1.0:
                print(f"✓ HEALTHY  composite stdev {cs_stdev:.2f} ≤ 1.0")
            elif cs_stdev <= 3.0:
                print(
                    f"⚠ NOISY    composite stdev {cs_stdev:.2f} (1.0-3.0): "
                    "results stable but with some run-to-run jitter"
                )
            else:
                print(
                    f"✗ UNSTABLE composite stdev {cs_stdev:.2f} > 3.0: "
                    "framework or engine is non-deterministic. Investigate."
                )
    return 0


def main(argv=None) -> int:
    _configure_utf8_stdio()
    import argparse as _ap

    parser = argparse.ArgumentParser(
        prog="refract",
        description=_TOP_DESCRIPTION,
        formatter_class=_ap.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_score_parser(sub)
    _add_selftest_parser(sub)
    _add_compare_parser(sub)
    _add_repeatability_parser(sub)
    _add_fetch_parser(sub)
    args = parser.parse_args(argv)
    if args.cmd == "score":
        return _run_score(args)
    if args.cmd == "selftest":
        return _run_selftest(args)
    if args.cmd == "compare":
        return _run_compare(args)
    if args.cmd == "repeatability":
        return _run_repeatability(args)
    if args.cmd == "fetch":
        return _run_fetch(args)
    parser.error(f"unknown subcommand: {args.cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
