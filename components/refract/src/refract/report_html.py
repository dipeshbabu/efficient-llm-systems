"""HTML report rendering for REFRACT (v0.3.3 redesign).

Single self-contained HTML page using a native system-font stack. The output
has no network dependencies and renders fully offline.

Design language: muted neutrals, semantic colour tokens per band,
pill badges, dense-but-clean tables, dark code blocks.

Layout:
  1. Masthead: brand mark + model + run metadata
  2. Stats strip: composite + 4 axes in a single 5-column row
  3. Diagnosis: summary callout + numbered findings
  4. Per-axis breakdown: dense table; R-NIAH and PLAD details inline
  5. Run details: 3-card grid (Model / Hardware / Environment)
  6. Reproduce: dark code block with syntax-highlighted flags
  7. Raw JSON in a styled `<details>`
  8. Footer: "what is this?" + GitHub link + version info
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import json as _json
import math
import os
import platform
import re as _re
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .axes.gtm import GTMResult
from .axes.kld import KLDResult
from .axes.plad import PLADResult
from .axes.rniah import RNIAHResult
from .axes.trajectory import TrajectoryResult
from .report import _sanitize_home_arg
from . import __report_schema__
from .score import CompositeScore, MIN_FLOOR, band, interpret_pattern


# Pretty band names shown to users (Excellent vs EXCELLENT).
_BAND_PRETTY = {
    "EXCELLENT": "Excellent",
    "PASS":      "Pass",
    "DEGRADED":  "Degraded",
    "FAIL":      "Fail",
}

# CSS class mapping used in the badge / fill colour.
_BAND_CLASS = {
    "EXCELLENT": "green",
    "PASS":      "green",
    "DEGRADED":  "amber",
    "FAIL":      "red",
}

_BAND_PROSE = {
    "EXCELLENT": "No material drift detected on the measured surfaces.",
    "PASS":      "Minor measured drift; validate on the target workload.",
    "DEGRADED":  "Visible drift. Audit on your workload before deploying.",
    "FAIL":      "Material quality loss. Treat as broken.",
}

_AXIS_PROSE = {
    "gtm":        "Token-level agreement with the fp16 reference (greedy decode).",
    "trajectory": "Token-level agreement with the fp16 reference (decode-time IDs).",
    "kld":        "Distribution-level divergence from the fp16 reference (corpus KLD).",
    "rniah":      "Long-context retrieval quality vs the reference (NIAH at multiple lengths).",
    "plad":       "Robustness to small prompt changes vs the reference (typo / case / punct / paraphrase).",
}

# Full-name → letter (axis A/B/C/D) and short label.
_AXIS_LETTER = {"gtm": "A", "trajectory": "A", "kld": "B",
                "rniah": "C", "plad": "D"}
_AXIS_SHORT = {"gtm": "GTM", "trajectory": "Trajectory", "kld": "KLD@D",
               "rniah": "R-NIAH", "plad": "PLAD"}
_AXIS_FULL = {
    "gtm":        "Greedy Trajectory Match",
    "trajectory": "Greedy Trajectory Match (decode-time IDs)",
    "kld":        "KL Divergence at the Decoder",
    "rniah":      "Retrieval Needle-In-A-Haystack",
    "plad":       "Perturbation-Locality Aware Drift",
}


# --------------------------------------------------------------------------
# Hardware + model metadata helpers
# --------------------------------------------------------------------------


def _hardware_metadata() -> dict:
    info: dict = {
        "system": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if info["system"] == "Darwin":
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            if chip:
                info["chip"] = chip
        except Exception:
            pass
        try:
            memsize = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip()
            if memsize:
                # Binary GiB (Apple's marketing convention) — 137438953472
                # bytes / 1024**3 = 128 GiB exact.
                info["ram_gb"] = round(int(memsize) / 1024**3, 1)
        except Exception:
            pass
        # Pretty platform string: "macOS 26.4 arm64" instead of full
        # "macOS-26.4.1-arm64-arm-64bit-Mach-O".
        try:
            mac_ver = platform.mac_ver()[0]
            if mac_ver:
                info["platform_pretty"] = f"macOS {mac_ver} {info['machine']}"
        except Exception:
            pass
    elif info["system"] == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["chip"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
        try:
            with open("/proc/meminfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["ram_gb"] = round(kb / 1024 / 1024, 1)
                        break
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip()
        if out:
            info["nvidia_gpus"] = [
                {"name": parts[0].strip(),
                 "memory_mb": int(parts[1].strip())}
                for line in out.splitlines()
                for parts in [line.split(",")] if len(parts) >= 2
            ]
    except Exception:
        pass
    return info


def _model_metadata(model_path: Path) -> dict:
    info: dict = {"path": str(model_path), "name": model_path.name}
    if not model_path.exists():
        return info
    if model_path.is_file():
        info["size_gb"] = round(model_path.stat().st_size / 1024**3, 2)
        info["format"] = "gguf" if model_path.suffix == ".gguf" else "file"
    elif model_path.is_dir():
        total = 0
        for ext in ("*.safetensors", "*.bin", "*.npz"):
            for f in model_path.glob(ext):
                total += f.stat().st_size
        info["size_gb"] = round(total / 1024**3, 2)
        info["format"] = "directory"
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                cfg = _json.loads(config_path.read_text(encoding="utf-8"))
                for k in (
                    "model_type", "architectures", "hidden_size",
                    "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "max_position_embeddings",
                    "vocab_size", "head_dim",
                ):
                    if k in cfg:
                        info[k] = cfg[k]
            except Exception:
                pass
    return info


def _repro_command(raw_json: dict | None, model: str,
                   reference_label: str, candidate_label: str,
                   has_rniah: bool, has_plad: bool) -> str:
    """Return the shell-escaped command that produced this report.

    Priority: JSON's repro_command (v0.3.2+) > sanitized sys.argv (when
    looks like a refract CLI run) > synthesized stand-in.
    """
    if raw_json and raw_json.get("repro_command"):
        return raw_json["repro_command"]
    home = os.path.expanduser("~")
    if any("refract" in a for a in sys.argv):
        return " ".join(
            shlex.quote(_sanitize_home_arg(str(a), home)) for a in sys.argv
        )
    model_short = Path(model).name
    cmd = [
        "python3", "-m", "refract.cli", "score",
        "--model", shlex.quote(model_short),
        "--reference", shlex.quote(reference_label),
        "--candidate", shlex.quote(candidate_label),
    ]
    if has_rniah or has_plad:
        cmd.append("--full")
    if has_rniah:
        cmd.extend(["--rniah-up-to", "16384"])
    cmd.extend([
        "--json-out", "report.json",
        "--html-out", "report.html",
    ])
    return " ".join(cmd)


# --------------------------------------------------------------------------
# HTML rendering helpers
# --------------------------------------------------------------------------


def _esc(s) -> str:
    return _html.escape("" if s is None else str(s))


def _badge(b: str, override_label: Optional[str] = None) -> str:
    cls = _BAND_CLASS.get(b, "gray")
    label = override_label or _BAND_PRETTY.get(b, b)
    return f'<span class="badge {cls}">{_esc(label)}</span>'


def _meter(score: float, b: str) -> str:
    pct = max(0.0, min(100.0, float(score)))
    # Empty band string ("") = explicit gray (used for low-confidence axes).
    cls = _BAND_CLASS.get(b, "gray") if b else "gray"
    return (
        f'<div class="meter"><div class="fill" '
        f'style="width:{pct:.1f}%; background:var(--{cls});"></div></div>'
    )


def _mini_meter(score: float, b: str) -> str:
    pct = max(0.0, min(100.0, float(score)))
    cls = _BAND_CLASS.get(b, "gray")
    return (
        f'<div class="mini-meter"><div class="fill" '
        f'style="width:{pct:.1f}%; background:var(--{cls});"></div></div>'
    )


def _highlight_repro(cmd: str) -> str:
    """Wrap --flags, args, and <placeholders> in spans for syntax colour."""
    parts = []
    for tok in cmd.split():
        if tok.startswith("--") or tok.startswith("-") and len(tok) > 1 and not tok[1].isdigit():
            parts.append(f'<span class="flag">{_esc(tok)}</span>')
        elif tok.startswith("<") and tok.endswith(">"):
            parts.append(f'<span class="placeholder">{_esc(tok)}</span>')
        elif "&lt;" in _esc(tok) and "&gt;" in _esc(tok):
            parts.append(f'<span class="placeholder">{_esc(tok)}</span>')
        else:
            parts.append(f'<span class="arg">{_esc(tok)}</span>')
    return " ".join(parts)


def _report_id() -> str:
    """Timestamp-based short ID, e.g. '#0430-1537'."""
    now = _dt.datetime.now()
    return f"#{now.strftime('%m%d-%H%M')}"


def _axis_letter_chip(letter: str) -> str:
    return f'<span class="letter">{_esc(letter)}</span>'


def _stat_block(name: str, score: Optional[float], *, is_composite: bool = False,
                low_confidence: bool = False) -> str:
    # When score is None the axis was skipped (--skip-gtm / --skip-kld).
    # Render explicit "n/a" instead of a fake 100/EXCELLENT.
    if score is None and not is_composite:
        letter = _AXIS_LETTER.get(name, "")
        short = _AXIS_SHORT.get(name, name)
        label_html = (
            f'<div class="label"><span class="axis-letter">{_esc(letter)}</span>'
            f'{_esc(short)}</div>'
        )
        value_html = (
            f'<div class="value-row">'
            f'<div class="value" style="color: var(--fg-faint);" '
            f'title="axis was skipped via --skip-{name} — not measured">n/a</div>'
            f'</div>'
        )
        return (
            f'<div class="stat skipped">'
            f'{label_html}{value_html}'
            f'<div class="badge-row">{_badge("", "Skipped")}</div>'
            f'</div>'
        )
    b = band(score)
    cls = "stat composite" if is_composite else "stat"
    if is_composite:
        label_html = '<div class="label">Composite score</div>'
        value_html = (
            f'<div class="value-row"><div class="value">{score:.2f}</div>'
            f'<div class="delta">/ 100</div></div>'
        )
    else:
        letter = _AXIS_LETTER.get(name, "")
        short = _AXIS_SHORT.get(name, name)
        label_html = (
            f'<div class="label"><span class="axis-letter">{_esc(letter)}</span>'
            f'{_esc(short)}</div>'
        )
        # v0.3.3: when an axis flags itself low-confidence, the score is
        # mathematically true but uninformative (e.g. R-NIAH=100 when base
        # also fails everywhere — measures "candidate matches base", not
        # real retrieval). Suppress the headline number and use "—" with
        # the real value in a tooltip to discourage misreading.
        if low_confidence:
            value_html = (
                f'<div class="value-row">'
                f'<div class="value" style="color: var(--fg-faint);" '
                f'title="raw score {score:.2f} — uninformative because base '
                f'also fails at most cells">—</div></div>'
            )
        else:
            value_html = f'<div class="value-row"><div class="value">{score:.2f}</div></div>'
    if low_confidence:
        badge = _badge("", "Low confidence")
    else:
        badge = _badge(b)
    return (
        f'<div class="{cls}">'
        f'{label_html}{value_html}'
        f'<div class="badge-row">{badge}</div>'
        f'</div>'
    )


def _findings(diagnosis: list[str]) -> str:
    if not diagnosis:
        return ""
    items = []
    for i, sentence in enumerate(diagnosis, 1):
        # Bold-up the leading clause if there's a colon / period.
        m = _re.match(r"([^.:]+[.:])\s+(.*)", sentence)
        if m:
            head, tail = m.group(1), m.group(2)
            inner = f"<strong>{_esc(head)}</strong> {_esc(tail)}"
        else:
            inner = _esc(sentence)
        items.append(
            f'<div class="finding">'
            f'<span class="num">{i:02d}</span>'
            f'<p>{inner}</p>'
            f'</div>'
        )
    return f'<div class="findings">{"".join(items)}</div>'


def _axis_row(name: str, score: Optional[float], *, low_confidence: bool = False) -> str:
    if score is None:
        # Skipped axis: explicit n/a row instead of fake 100/EXCELLENT.
        empty_meter = '<div class="meter-cell"><div class="meter"></div></div>'
        return (
            f'<div class="axis-row skipped">'
            f'<div class="key">{_axis_letter_chip(_AXIS_LETTER.get(name, ""))}'
            f'{_esc(_AXIS_SHORT.get(name, name))}</div>'
            f'<div class="name">{_esc(_AXIS_FULL.get(name, name))}'
            f'<span class="desc">skipped via --skip-{name}; not measured</span></div>'
            f'<div class="score" style="color: var(--fg-faint);">n/a</div>'
            f'<div class="badge-cell">{_badge("", "Skipped")}</div>'
            f'{empty_meter}'
            f'</div>'
        )
    b = band(score)
    if low_confidence:
        badge = _badge("", "Low confidence")
        meter_b = ""  # gray fill
        score_html = (
            f'<div class="score" style="color: var(--fg-faint);" '
            f'title="raw score {score:.2f} — uninformative">—</div>'
        )
    else:
        badge = _badge(b)
        meter_b = b
        score_html = f'<div class="score">{score:.2f}</div>'
    return (
        f'<div class="axis-row">'
        f'<div class="key">{_axis_letter_chip(_AXIS_LETTER.get(name, ""))}'
        f'{_esc(_AXIS_SHORT.get(name, name))}</div>'
        f'<div class="name">{_esc(_AXIS_FULL.get(name, name))}'
        f'<span class="desc">{_esc(_AXIS_PROSE.get(name, ""))}</span></div>'
        f'{score_html}'
        f'<div class="badge-cell">{badge}</div>'
        f'{_meter(score, meter_b)}'
        f'</div>'
    )


def _rniah_low_confidence(rniah: RNIAHResult) -> bool:
    """Mirror of the JSON `confidence: low` guard: True when base_acc
    averaged across cells is below 0.2 (model isn't engaging the task)."""
    return rniah.confidence == "low"


def _rniah_matrix_detail(rniah: RNIAHResult) -> str:
    if not rniah.cells:
        return ""
    low_conf = _rniah_low_confidence(rniah)
    warning_html = ""
    if low_conf:
        base_avg = sum(c.base_acc for c in rniah.cells) / len(rniah.cells)
        warning_html = (
            '<div class="summary-box amber" style="margin-bottom: 14px;">'
            '<div class="icon">!</div>'
            '<div class="body">'
            '<div class="title">Low confidence — score is noise floor.</div>'
            f'<div class="desc">fp16 baseline retrieves at only '
            f'{base_avg:.0%} of cells on average. With the reference '
            f'failing this often, R-NIAH = {rniah.score:.2f} measures '
            f'"candidate matches base" rather than real retrieval '
            f'capability. Try lower --rniah-up-to or a model with '
            f'longer effective context.</div>'
            '</div></div>'
        )
    lengths = sorted({c.length for c in rniah.cells})
    positions = sorted({c.position for c in rniah.cells})
    head = "<tr><th>length \\ position</th>" + "".join(
        f'<th>{p:.2f}</th>' for p in positions
    ) + "</tr>"
    rows = []
    for length in lengths:
        tds = []
        for pos in positions:
            cell = next(
                (c for c in rniah.cells
                 if c.length == length and c.position == pos), None,
            )
            if cell is None:
                tds.append('<td class="cell-na" title="cell not run">—</td>')
                continue
            base, cand = cell.base_acc, cell.cand_acc
            if base == 0.0:
                # Both base and candidate fail at this cell. Showing the
                # literal "0.00 / 0.00" reads as a real result; show a
                # cleaner "n/a" with a tooltip explaining why.
                tds.append(
                    '<td class="cell-na" '
                    'title="fp16 baseline does not retrieve at this cell — '
                    'uninformative for the candidate vs base comparison">n/a</td>'
                )
                continue
            if cell.degradation > 0:
                cls = "cell-fail"
                title = (f"candidate retrieves at {cand:.2f}, "
                         f"base at {base:.2f} — degradation {cell.degradation:.2f}")
            else:
                cls = "cell-pass"
                title = (f"candidate retrieves at {cand:.2f}, "
                         f"base at {base:.2f} — match")
            tds.append(
                f'<td class="{cls}" title="{_esc(title)}">'
                f'{cand:.2f} / {base:.2f}</td>'
            )
        rows.append(f"<tr><th>{length}</th>{''.join(tds)}</tr>")
    return (
        f'<div class="axis-detail-row">'
        f'<div class="detail-inner">'
        f'{warning_html}'
        f'<div class="detail-label">R-NIAH per-cell '
        f'<span class="legend">cand / base retrieval rate · '
        f'green = match · red = candidate worse · '
        f'n/a = base also fails at this cell</span></div>'
        f'<table class="matrix"><thead>{head}</thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        f'</div></div>'
    )


def _plad_table_detail(plad: PLADResult) -> str:
    rows = []
    for pert, score in plad.per_perturbation_score.items():
        if not isinstance(score, (int, float)) or math.isnan(score):
            rows.append(
                f'<tr>'
                f'<td class="label">{_esc(pert)}</td>'
                f'<td class="score-cell"><span class="skipped">skipped</span></td>'
                f'<td class="band-cell">{_badge("", "N/A")}</td>'
                f'<td class="meter-cell"></td>'
                f'<td class="note-cell">didn\'t apply on these prompts</td>'
                f'</tr>'
            )
            continue
        b = band(score)
        rows.append(
            f'<tr>'
            f'<td class="label">{_esc(pert)}</td>'
            f'<td class="score-cell">{score:.2f}</td>'
            f'<td class="band-cell">{_badge(b)}</td>'
            f'<td class="meter-cell">{_mini_meter(score, b)}</td>'
            f'<td class="note-cell"></td>'
            f'</tr>'
        )
    return (
        f'<div class="axis-detail-row">'
        f'<div class="detail-inner">'
        f'<div class="detail-label">PLAD per-perturbation</div>'
        f'<table class="plad-table"><thead>'
        f'<tr><th>perturbation</th><th>score</th><th>band</th>'
        f'<th>distribution</th><th></th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        f'</div></div>'
    )


def _summary_box(composite: CompositeScore) -> str:
    b = composite.band
    cls = _BAND_CLASS.get(b, "gray")
    icon_text = {"EXCELLENT": "✓", "PASS": "✓", "DEGRADED": "!", "FAIL": "!"}.get(b, "·")
    title = _BAND_PROSE.get(b, "")
    # Build a one-liner about which axes drove the band.
    bits = []
    for axis_name, score in (
        ("Trajectory", composite.gtm_score),
        ("KLD", composite.kld_score),
        ("R-NIAH", composite.rniah_score),
        ("PLAD", composite.plad_score),
    ):
        if score is None:
            continue
        ab = band(score)
        if ab in ("FAIL", "DEGRADED"):
            bits.append(f"{axis_name} {ab.lower()}")
    if bits:
        desc = (
            f"Composite of {composite.composite:.2f} falls in the "
            f"{_BAND_PRETTY.get(b, b)} band. Surfaces below threshold: "
            f"{', '.join(bits)}."
        )
    else:
        desc = (
            f"Composite of {composite.composite:.2f} falls in the "
            f"{_BAND_PRETTY.get(b, b)} band. All measured axes pass."
        )
    return (
        f'<div class="summary-box {cls}">'
        f'<div class="icon">{_esc(icon_text)}</div>'
        f'<div class="body">'
        f'<div class="title">{_esc(title)}</div>'
        f'<div class="desc">{_esc(desc)}</div>'
        f'</div></div>'
    )


def _kv_pair(label: str, value: str) -> str:
    return f"<dt>{_esc(label)}</dt><dd>{_esc(value)}</dd>"


def _run_details(model_meta: dict, hw_meta: dict,
                 reference_label: str, candidate_label: str,
                 env_meta: Optional[dict] = None) -> str:
    # Model card
    model_dl = []
    for k, label in (
        ("name", "name"), ("size_gb", "size"), ("format", "format"),
        ("model_type", "type"), ("hidden_size", "hidden"),
        ("num_hidden_layers", "layers"),
        ("num_attention_heads", "heads"),
        ("num_key_value_heads", "kv heads"),
        ("head_dim", "head dim"),
        ("vocab_size", "vocab"),
    ):
        if k not in model_meta:
            continue
        v = model_meta[k]
        if k == "size_gb":
            v = f"{v} GB"
        model_dl.append(_kv_pair(label, v))

    # Hardware card
    hw_dl = []
    if hw_meta.get("chip"):
        hw_dl.append(_kv_pair("chip", hw_meta["chip"]))
    if hw_meta.get("platform_pretty") or hw_meta.get("platform"):
        hw_dl.append(_kv_pair(
            "platform",
            hw_meta.get("platform_pretty") or hw_meta.get("platform"),
        ))
    if "ram_gb" in hw_meta:
        hw_dl.append(_kv_pair("ram", f"{hw_meta['ram_gb']} GB"))
    if hw_meta.get("machine"):
        hw_dl.append(_kv_pair("arch", hw_meta["machine"]))
    if hw_meta.get("python"):
        hw_dl.append(_kv_pair("python", hw_meta["python"]))
    if hw_meta.get("nvidia_gpus"):
        gpu_str = ", ".join(
            f"{g['name']} ({g['memory_mb']/1024:.1f} GB)"
            for g in hw_meta["nvidia_gpus"]
        )
        hw_dl.append(_kv_pair("gpu", gpu_str))

    # Environment card
    # Reference + candidate split into K/V lines for legibility.
    def _split_kv(spec: str) -> str:
        parts = [p.strip() for p in spec.split(",")]
        return "<br>".join(parts) if parts else spec
    env_dl = []
    em = env_meta or {}
    backend_name = em.get("backend")
    if backend_name:
        # Map internal backend name to a friendlier display label.
        backend_display = {
            "llamacpp": "llama.cpp",
            "mlx": "MLX (Apple Silicon)",
            "vllm": "vLLM",
            "sglang": "SGLang",
        }.get(backend_name, backend_name)
        env_dl.append(_kv_pair("backend", backend_display))
    # Engine-specific version + commit fields (whichever the active backend
    # populated). Surfaced because "which engine?" is the first question
    # readers ask when a report lands in their inbox.
    for key, label in (
        ("llama_cpp_commit", "llama.cpp commit"),
        ("llama_cpp_version", "llama.cpp version"),
        ("mlx_lm_version", "mlx-lm version"),
        ("mlx_version", "mlx version"),
        ("vllm_version", "vLLM version"),
        ("sglang_url", "SGLang URL"),
        ("served_model_id", "served model"),
    ):
        if em.get(key):
            env_dl.append(_kv_pair(label, em[key]))
    env_dl.extend([
        f"<dt>reference</dt><dd>{_split_kv(_esc(reference_label))}</dd>",
        f"<dt>candidate</dt><dd>{_split_kv(_esc(candidate_label))}</dd>",
    ])

    return (
        f'<div class="run-grid">'
        f'<div class="run-card">'
        f'<div class="card-head"><span class="dot"></span><h3>Model</h3></div>'
        f'<dl>{"".join(model_dl)}</dl>'
        f'</div>'
        f'<div class="run-card">'
        f'<div class="card-head"><span class="dot"></span><h3>Hardware</h3></div>'
        f'<dl>{"".join(hw_dl)}</dl>'
        f'</div>'
        f'<div class="run-card">'
        f'<div class="card-head"><span class="dot"></span><h3>Environment</h3></div>'
        f'<dl>{"".join(env_dl)}</dl>'
        f'</div>'
        f'</div>'
    )


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

_CSS = r"""
:root {
    color-scheme: light dark;

    /* light-dark() requires Chrome 123+ / Safari 17.5+ / Firefox 120+ (all 2024).
       Fallback browsers see the light values. */
    --bg:        light-dark(#f6f6f7, #0a0a0c);
    --paper:     light-dark(#ffffff, #131316);
    --subtle:    light-dark(#fafafa, #1c1c20);
    --hover:     light-dark(#f4f4f5, #27272a);

    --fg:        light-dark(#09090b, #fafafa);
    --fg-2:      light-dark(#27272a, #e4e4e7);
    --fg-muted:  light-dark(#52525b, #a1a1aa);
    --fg-faint:  light-dark(#a1a1aa, #71717a);

    --border:        light-dark(#e4e4e7, #27272a);
    --border-strong: light-dark(#d4d4d8, #3f3f46);

    --red:     light-dark(#b42318, #fca5a5);
    --red-bg:  light-dark(#fef3f2, #2a1517);
    --red-br:  light-dark(#fecdca, #5e2424);

    --amber:     light-dark(#b54708, #fcd34d);
    --amber-bg:  light-dark(#fffaeb, #2a1f0a);
    --amber-br:  light-dark(#fedf89, #5c4319);

    --green:     light-dark(#027a48, #86efac);
    --green-bg:  light-dark(#ecfdf3, #0e1f13);
    --green-br:  light-dark(#abefc6, #1f4a2c);

    --blue:     light-dark(#175cd3, #93c5fd);
    --blue-bg:  light-dark(#eff8ff, #0e1929);
    --blue-br:  light-dark(#b2ddff, #1f3a5c);

    --gray:     light-dark(#52525b, #a1a1aa);

    --code-bg:   light-dark(#18181b, #0a0a0c);
    --code-fg:   #e4e4e7;
    --code-dim:  #71717a;
    --code-line: light-dark(#27272a, #27272a);

    --sans: "Geist", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    --mono: "Geist Mono", ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace;
}

/* Manual override — the toggle button sets data-theme on <html>.
   Without it, color-scheme stays "light dark" and follows the OS. */
:root[data-theme="light"] { color-scheme: light; }
:root[data-theme="dark"]  { color-scheme: dark;  }
* { box-sizing: border-box; }
html { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
body {
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.5;
    color: var(--fg);
    background: var(--bg);
    margin: 0;
    padding: 32px 20px 80px;
    font-feature-settings: "ss01", "cv11";
}
.report {
    max-width: 1080px;
    margin: 0 auto;
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* HEADER */
.masthead {
    position: relative;
    padding: 24px 32px 20px;
    border-bottom: 1px solid var(--border);
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 24px;
    align-items: start;
}
.masthead .brand { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.masthead .brand-mark {
    width: 22px; height: 22px; border-radius: 5px;
    background: var(--fg); color: var(--paper);
    display: grid; place-items: center;
    font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: -0.04em;
}
.masthead .brand-text { font-family: var(--mono); font-size: 12px; font-weight: 500; }
.masthead .brand-text .ver { color: var(--fg-faint); font-weight: 400; margin-left: 6px; }
.masthead h1 {
    margin: 0; font-size: 24px; font-weight: 600; letter-spacing: -0.02em;
    line-height: 1.2; font-family: var(--mono);
}
.masthead .subtitle { margin-top: 6px; font-size: 13px; color: var(--fg-muted); }
.masthead .meta {
    display: grid; gap: 4px; text-align: right;
    font-family: var(--mono); font-size: 12px; color: var(--fg-muted);
    margin-top: 32px; /* leave room for the absolute-positioned theme toggle */
}
.masthead .meta .row { display: flex; gap: 8px; justify-content: flex-end; align-items: center; }
.masthead .meta .k { color: var(--fg-faint); }
.masthead .meta .v { color: var(--fg-2); font-weight: 500; }

/* THEME TOGGLE — sun/moon button at top-right of the masthead */
.theme-toggle {
    position: absolute;
    top: 20px; right: 24px;
    width: 30px; height: 30px;
    border-radius: 7px;
    border: 1px solid var(--border);
    background: var(--paper);
    color: var(--fg-muted);
    cursor: pointer;
    display: grid; place-items: center;
    padding: 0; z-index: 1;
    transition: background 100ms ease, border-color 100ms ease, color 100ms ease;
}
.theme-toggle:hover { background: var(--hover); border-color: var(--border-strong); color: var(--fg); }
.theme-toggle:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
.theme-toggle svg { width: 14px; height: 14px; display: block; }
.theme-toggle .icon-sun { display: none; }
@media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) .theme-toggle .icon-moon { display: none; }
    :root:not([data-theme="light"]) .theme-toggle .icon-sun  { display: block; }
}
:root[data-theme="dark"]  .theme-toggle .icon-moon { display: none; }
:root[data-theme="dark"]  .theme-toggle .icon-sun  { display: block; }
:root[data-theme="light"] .theme-toggle .icon-moon { display: block; }
:root[data-theme="light"] .theme-toggle .icon-sun  { display: none; }

/* STATS STRIP */
.stats-strip {
    display: grid;
    grid-template-columns: 1.4fr 1fr 1fr 1fr 1fr;
    border-bottom: 1px solid var(--border);
    background: var(--subtle);
}
.stat { padding: 18px 24px 20px; border-right: 1px solid var(--border); display: flex; flex-direction: column; gap: 6px; }
.stat:last-child { border-right: none; }
.stat .label { font-family: var(--mono); font-size: 11px; font-weight: 500; color: var(--fg-muted); letter-spacing: -0.01em; }
.stat .label .axis-letter { color: var(--fg-faint); margin-right: 6px; }
.stat .value-row { display: flex; align-items: baseline; gap: 8px; }
.stat .value {
    font-family: var(--mono); font-size: 28px; font-weight: 500;
    letter-spacing: -0.03em; color: var(--fg); line-height: 1;
    font-feature-settings: "tnum", "ss01";
}
.stat .delta { font-family: var(--mono); font-size: 11px; color: var(--fg-faint); font-weight: 400; }
.stat.composite .value { font-size: 36px; font-weight: 600; }
.stat .badge-row { margin-top: 2px; }

/* BADGES */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 2px 8px 2px 7px;
    border-radius: 999px;
    font-family: var(--mono); font-size: 11px; font-weight: 500;
    line-height: 1.5; letter-spacing: -0.01em;
    border: 1px solid; white-space: nowrap;
}
.badge::before { content: ""; width: 5px; height: 5px; border-radius: 50%; background: currentColor; flex-shrink: 0; }
.badge.red   { color: var(--red);   background: var(--red-bg);   border-color: var(--red-br); }
.badge.amber { color: var(--amber); background: var(--amber-bg); border-color: var(--amber-br); }
.badge.green { color: var(--green); background: var(--green-bg); border-color: var(--green-br); }
.badge.blue  { color: var(--blue);  background: var(--blue-bg);  border-color: var(--blue-br); }
.badge.gray  { color: var(--fg-muted); background: var(--subtle); border-color: var(--border); }

/* SECTION */
.section { padding: 28px 32px; border-bottom: 1px solid var(--border); }
.section:last-of-type { border-bottom: none; }
.section-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; gap: 16px; }
.section-head h2 { margin: 0; font-size: 13px; font-weight: 600; letter-spacing: -0.01em; }
.section-head .hint { font-family: var(--mono); font-size: 11px; color: var(--fg-faint); letter-spacing: -0.01em; }

/* SUMMARY BOX */
.summary-box {
    border-radius: 8px; padding: 14px 16px;
    display: flex; gap: 12px; align-items: flex-start;
    border: 1px solid;
}
.summary-box.red   { background: var(--red-bg);   border-color: var(--red-br); }
.summary-box.red   .icon { background: var(--red); }
.summary-box.red   .title { color: var(--red); }
.summary-box.amber { background: var(--amber-bg); border-color: var(--amber-br); }
.summary-box.amber .icon { background: var(--amber); }
.summary-box.amber .title { color: var(--amber); }
.summary-box.green { background: var(--green-bg); border-color: var(--green-br); }
.summary-box.green .icon { background: var(--green); }
.summary-box.green .title { color: var(--green); }
.summary-box .icon {
    width: 18px; height: 18px; border-radius: 50%;
    color: #fff; display: grid; place-items: center;
    font-family: var(--mono); font-size: 11px; font-weight: 600;
    flex-shrink: 0; margin-top: 1px;
}
.summary-box .body { flex: 1; }
.summary-box .title { font-weight: 600; font-size: 13.5px; margin-bottom: 4px; letter-spacing: -0.01em; }
.summary-box .desc { color: var(--fg-2); font-size: 13px; }

/* FINDINGS */
.findings { display: flex; flex-direction: column; margin-top: 16px; }
.finding {
    display: grid; grid-template-columns: auto 1fr; gap: 12px;
    padding: 12px 0; border-bottom: 1px solid var(--border); align-items: flex-start;
}
.finding:last-child { border-bottom: none; }
.finding:first-child { border-top: 1px solid var(--border); }
.finding .num {
    font-family: var(--mono); font-size: 11px; font-weight: 500;
    color: var(--fg-faint); background: var(--subtle);
    padding: 2px 6px; border-radius: 4px; border: 1px solid var(--border);
    flex-shrink: 0; margin-top: 1px;
}
.finding p { margin: 0; font-size: 13.5px; color: var(--fg-2); line-height: 1.5; }
.finding p strong { font-weight: 600; color: var(--fg); }

/* AXES TABLE */
.axes-table { width: 100%; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.axes-table .head-row {
    display: grid;
    grid-template-columns: 96px minmax(0, 1.4fr) 90px 110px minmax(0, 1.2fr);
    background: var(--subtle); border-bottom: 1px solid var(--border);
    padding: 10px 16px; gap: 16px; align-items: center;
    font-family: var(--mono); font-size: 11px; color: var(--fg-muted);
    font-weight: 500; letter-spacing: -0.01em;
}
.axis-row {
    display: grid;
    grid-template-columns: 96px minmax(0, 1.4fr) 90px 110px minmax(0, 1.2fr);
    padding: 14px 16px; gap: 16px; align-items: center;
    border-bottom: 1px solid var(--border);
}
.axis-row:last-of-type { border-bottom: none; }
.axis-row .key {
    font-family: var(--mono); font-size: 12px; color: var(--fg-muted);
    font-weight: 500; display: flex; align-items: center; gap: 6px;
}
.axis-row .key .letter {
    display: inline-grid; place-items: center;
    width: 18px; height: 18px; border-radius: 4px;
    background: var(--fg); color: #fff;
    font-size: 10px; letter-spacing: 0; font-weight: 500;
}
.axis-row .name { font-size: 13.5px; font-weight: 500; color: var(--fg); letter-spacing: -0.01em; }
.axis-row .name .desc {
    display: block; font-size: 12px; color: var(--fg-muted);
    font-weight: 400; margin-top: 2px; line-height: 1.4;
}
.axis-row .score {
    font-family: var(--mono); font-size: 16px; font-weight: 500;
    text-align: right; font-feature-settings: "tnum"; letter-spacing: -0.02em;
}
.axis-row .meter { position: relative; height: 6px; background: var(--hover); border-radius: 3px; overflow: hidden; }
.axis-row .meter .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; }

.axis-detail-row { padding: 0 16px 18px 16px; background: var(--subtle); border-bottom: 1px solid var(--border); }
.axis-detail-row:last-of-type { border-bottom: none; }
.detail-inner { padding: 16px 0 4px; border-top: 1px dashed var(--border); }
.detail-label { font-family: var(--mono); font-size: 11px; font-weight: 500; color: var(--fg-muted); margin-bottom: 10px; letter-spacing: -0.01em; }
.detail-label .legend { color: var(--fg-faint); margin-left: 8px; font-weight: 400; }

/* MATRIX */
.matrix {
    width: 100%; border-collapse: separate; border-spacing: 0;
    font-family: var(--mono); font-size: 12px;
    border: 1px solid var(--border); border-radius: 6px;
    overflow: hidden; background: var(--paper);
}
.matrix th, .matrix td {
    padding: 10px 14px; text-align: center;
    border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);
    font-feature-settings: "tnum";
}
.matrix th:last-child, .matrix td:last-child { border-right: none; }
.matrix tr:last-child th, .matrix tr:last-child td { border-bottom: none; }
.matrix thead th { background: var(--subtle); font-weight: 500; font-size: 11px; color: var(--fg-muted); }
.matrix tbody th { background: var(--subtle); font-weight: 500; color: var(--fg-2); font-size: 11px; }
.matrix .cell-pass { color: var(--green); background: var(--green-bg); font-weight: 500; }
.matrix .cell-fail { color: var(--red);   background: var(--red-bg);   font-weight: 500; }
.matrix .cell-na   { color: var(--fg-faint); background: var(--paper); font-style: italic; }

/* PLAD TABLE */
.plad-table {
    width: 100%; border-collapse: collapse;
    background: var(--paper); border: 1px solid var(--border);
    border-radius: 6px; overflow: hidden;
}
.plad-table th, .plad-table td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); vertical-align: middle; }
.plad-table tr:last-child th, .plad-table tr:last-child td { border-bottom: none; }
.plad-table thead th { background: var(--subtle); font-family: var(--mono); font-size: 11px; font-weight: 500; color: var(--fg-muted); }
.plad-table .label { font-family: var(--mono); font-size: 12.5px; color: var(--fg); width: 18%; font-weight: 500; }
.plad-table .score-cell { font-family: var(--mono); font-size: 14px; color: var(--fg); width: 14%; font-weight: 500; font-feature-settings: "tnum"; }
.plad-table .band-cell { width: 16%; }
.plad-table .meter-cell { width: 36%; }
.plad-table .note-cell { width: 16%; font-size: 12px; color: var(--fg-faint); font-style: italic; }
.plad-table .skipped { color: var(--fg-faint); font-style: italic; font-family: var(--mono); font-size: 12.5px; }
.mini-meter { position: relative; height: 6px; background: var(--hover); border-radius: 3px; }
.mini-meter .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; }

/* RUN DETAILS */
.run-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.run-card { border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; background: var(--paper); }
.run-card .card-head { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.run-card .card-head .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--fg-muted); }
.run-card h3 { margin: 0; font-size: 12px; font-weight: 600; letter-spacing: -0.01em; }
.run-card dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 8px 14px; }
.run-card dt { font-family: var(--mono); font-size: 11.5px; color: var(--fg-muted); align-self: baseline; }
.run-card dd { margin: 0; font-family: var(--mono); font-size: 12px; word-break: break-word; font-feature-settings: "tnum"; text-align: right; }

/* CODE */
.code-block {
    background: var(--code-bg); color: var(--code-fg);
    font-family: var(--mono); font-size: 12.5px; line-height: 1.65;
    padding: 14px 16px; border-radius: 8px; border: 1px solid var(--code-line);
    display: flex; align-items: flex-start; gap: 10px;
}
.code-block .prompt { color: var(--code-dim); user-select: none; flex-shrink: 0; }
.code-block code { color: var(--code-fg); white-space: pre-wrap; word-break: break-word; flex: 1; min-width: 0; }
.code-block code .flag { color: #93c5fd; }
.code-block code .arg { color: #fcd34d; }
.code-block code .placeholder { color: var(--code-dim); font-style: italic; }

details.json-toggle { margin-top: 16px; }
details.json-toggle summary {
    cursor: pointer; font-family: var(--mono); font-size: 12px; color: var(--fg-muted);
    list-style: none; display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px;
    background: var(--paper); user-select: none; font-weight: 500;
    transition: background 100ms ease, border-color 100ms ease;
}
details.json-toggle summary:hover { background: var(--hover); border-color: var(--border-strong); }
details.json-toggle summary::-webkit-details-marker { display: none; }
details.json-toggle summary::before {
    content: "›"; color: var(--fg-faint); font-size: 14px; line-height: 1;
    transition: transform 120ms ease; display: inline-block;
}
details.json-toggle[open] summary::before { transform: rotate(90deg); }
details.json-toggle pre {
    margin: 12px 0 0; background: var(--code-bg); color: var(--code-fg);
    padding: 16px 18px; border-radius: 8px; overflow: auto;
    font-family: var(--mono); font-size: 11.5px; line-height: 1.65;
    max-height: 480px; border: 1px solid var(--code-line);
}

/* FOOTER */
.report-foot { padding: 0; background: var(--subtle); border-top: 1px solid var(--border); }
.foot-info {
    padding: 14px 32px; border-bottom: 1px solid var(--border);
    font-size: 12.5px; color: var(--fg-muted); line-height: 1.5;
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
}
.foot-info .what { color: var(--fg-2); }
.foot-info .docs-link {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--mono); font-size: 12px; color: var(--fg);
    text-decoration: none; padding: 5px 10px; border: 1px solid var(--border);
    border-radius: 6px; background: var(--paper); font-weight: 500;
    transition: background 100ms ease, border-color 100ms ease;
    white-space: nowrap; flex-shrink: 0;
}
.foot-info .docs-link:hover { background: var(--hover); border-color: var(--border-strong); }
.foot-info .docs-link .arrow { color: var(--fg-faint); font-size: 11px; }
.foot-bar { padding: 12px 32px; display: flex; justify-content: space-between; align-items: center; font-family: var(--mono); font-size: 11px; color: var(--fg-muted); }
.foot-bar .right { display: flex; gap: 14px; }
.foot-bar .right span { color: var(--fg-faint); }

/* RESPONSIVE */
@media (max-width: 900px) {
    .stats-strip { grid-template-columns: repeat(2, 1fr); }
    .stat { border-right: 1px solid var(--border); border-bottom: 1px solid var(--border); }
    .stat:nth-child(2n) { border-right: none; }
    .stat.composite { grid-column: 1 / -1; }
    .axes-table .head-row { display: none; }
    .axis-row { grid-template-columns: auto 1fr; gap: 10px 14px; padding: 14px; }
    .axis-row .key { grid-row: 1; }
    .axis-row .name { grid-row: 1; grid-column: 2; }
    .axis-row .score { grid-row: 2; grid-column: 1 / 2; text-align: left; }
    .axis-row .badge-cell { grid-row: 2; grid-column: 2; }
    .axis-row .meter { grid-row: 3; grid-column: 1 / -1; }
    .run-grid { grid-template-columns: 1fr; }
    .masthead { grid-template-columns: 1fr; }
    .masthead .meta { text-align: left; }
    .masthead .meta .row { justify-content: flex-start; }
}
@media (max-width: 560px) {
    body { padding: 12px; }
    .section { padding: 20px 18px; }
    .masthead { padding: 18px; }
    .stats-strip { grid-template-columns: 1fr; }
    .stat { border-right: none; }
    .foot-info { flex-direction: column; align-items: flex-start; padding: 14px 18px; }
    .foot-bar { padding: 12px 18px; }
}
"""


# --------------------------------------------------------------------------
# Main entry: html_report
# --------------------------------------------------------------------------


def html_report(
    *,
    model: str,
    reference_label: str,
    candidate_label: str,
    composite: CompositeScore,
    gtm: GTMResult,
    kld: KLDResult,
    rniah: Optional[RNIAHResult] = None,
    plad: Optional[PLADResult] = None,
    raw_json: Optional[dict] = None,
) -> str:
    """Render the report as a self-contained HTML page (string)."""
    from . import __version__
    model_meta = _model_metadata(Path(model))
    hw_meta = _hardware_metadata()
    repro = _repro_command(
        raw_json=raw_json, model=model,
        reference_label=reference_label, candidate_label=candidate_label,
        has_rniah=rniah is not None,
        has_plad=(plad is not None and composite.plad_score is not None),
    )
    diag = interpret_pattern(
        gtm_score=composite.gtm_score,
        kld_score=composite.kld_score,
        rniah_score=composite.rniah_score,
        plad_score=composite.plad_score,
    )
    axis_a_key = "trajectory" if isinstance(gtm, TrajectoryResult) else "gtm"
    rid = _report_id()
    when_pretty = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Stats strip
    stats = [_stat_block(axis_a_key, composite.gtm_score, is_composite=False)]  # placeholder, replaced below
    stats = [
        _stat_block("composite", composite.composite, is_composite=True).replace(
            'class="label"><span class="axis-letter"></span>composite',
            'class="label">Composite score',
        ),
    ]
    # Actual composite block (overrides the placeholder)
    rniah_low_conf = (rniah is not None and _rniah_low_confidence(rniah))

    stats = []
    stats.append(_stat_block("composite", composite.composite, is_composite=True))
    stats.append(_stat_block(axis_a_key, composite.gtm_score))
    stats.append(_stat_block("kld", composite.kld_score))
    if rniah is not None:
        stats.append(_stat_block(
            "rniah", rniah.score,
            low_confidence=rniah_low_conf,
        ))
    if composite.plad_score is not None:
        stats.append(_stat_block("plad", composite.plad_score))

    # Axes table rows
    axis_rows = [_axis_row(axis_a_key, composite.gtm_score),
                 _axis_row("kld", composite.kld_score)]
    if rniah is not None:
        axis_rows.append(_axis_row(
            "rniah", rniah.score, low_confidence=rniah_low_conf,
        ))
        axis_rows.append(_rniah_matrix_detail(rniah))
    if composite.plad_score is not None and plad is not None:
        axis_rows.append(_axis_row("plad", composite.plad_score))
        axis_rows.append(_plad_table_detail(plad))

    n_axes = sum(1 for s in (
        composite.gtm_score, composite.kld_score,
        composite.rniah_score, composite.plad_score,
    ) if s is not None)

    raw = _json.dumps(raw_json or {}, indent=2, default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>REFRACT report — {_esc(model_meta.get("name", model))}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="report">

  <header class="masthead">
    <button class="theme-toggle" type="button" aria-label="Toggle theme" title="Toggle light / dark">
      <svg class="icon-moon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      <svg class="icon-sun" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>
    </button>
    <div>
      <div class="brand">
        <span class="brand-mark">R</span>
        <span class="brand-text">refract <span class="ver">v{_esc(__version__)}</span></span>
      </div>
      <h1>{_esc(model_meta.get("name", model))}</h1>
      <div class="subtitle">Quantization audit · {_esc(candidate_label)} against fp16 reference</div>
    </div>
    <div class="meta">
      <div class="row"><span class="k">report</span><span class="v">{_esc(rid)}</span></div>
      <div class="row"><span class="k">generated</span><span class="v">{_esc(when_pretty)}</span></div>
      <div class="row"><span class="k">scoring</span><span class="v">0–100, higher is better</span></div>
    </div>
  </header>

  <div class="stats-strip">
    {''.join(stats)}
  </div>

  <section class="section">
    <div class="section-head">
      <h2>Diagnosis</h2>
      <span class="hint">{len(diag)} finding{'s' if len(diag) != 1 else ''}</span>
    </div>
    {_summary_box(composite)}
    {_findings(diag)}
  </section>

  <section class="section">
    <div class="section-head">
      <h2>Per-axis breakdown</h2>
      <span class="hint">{n_axes} axes</span>
    </div>
    <div class="axes-table">
      <div class="head-row">
        <div>axis</div>
        <div>metric</div>
        <div style="text-align: right;">score</div>
        <div>band</div>
        <div>distribution</div>
      </div>
      {''.join(axis_rows)}
    </div>
  </section>

  <section class="section">
    <div class="section-head">
      <h2>Run details</h2>
      <span class="hint">model · hardware · environment</span>
    </div>
    {_run_details(model_meta, hw_meta, reference_label, candidate_label,
                  env_meta=(raw_json or {}).get("environment"))}
  </section>

  <section class="section">
    <div class="section-head">
      <h2>Reproduce</h2>
      <span class="hint">command</span>
    </div>
    <div class="code-block">
      <span class="prompt">$</span>
      <code>{_highlight_repro(repro)}</code>
    </div>

    <details class="json-toggle">
      <summary>Raw JSON · machine-readable</summary>
<pre>{_esc(raw)}</pre>
    </details>
  </section>

  <footer class="report-foot">
    <div class="foot-info">
      <div class="what">
        <strong style="color: var(--fg); font-weight: 600;">What is this?</strong>
        REFRACT — <em>REFerence-anchored Robust Acid-test for Compressed Transformers</em> — is a quantization audit framework that scores how faithfully a quantized KV-cache config preserves the model's own fp16 behaviour. The repository contains documentation, the motivation paper, and guidance on interpreting these scores.
      </div>
      <a class="docs-link" href="https://github.com/dipeshbabu/efficient-llm-systems/tree/main/components/refract" target="_blank" rel="noopener">
        github.com/dipeshbabu/efficient-llm-systems
        <span class="arrow">↗</span>
      </a>
    </div>
    <div class="foot-bar">
      <span>refract · v{_esc(__version__)}</span>
      <div class="right">
        <span>schema {_esc(__report_schema__)}</span>
        <span>·</span>
        <span>{_esc(rid)}</span>
      </div>
    </div>
  </footer>

</div>
<script>
(function() {{
    var btn = document.querySelector('.theme-toggle');
    if (!btn) return;
    btn.addEventListener('click', function() {{
        var html = document.documentElement;
        var current = html.getAttribute('data-theme');
        if (!current) {{
            // No override yet — read OS preference and flip to opposite.
            current = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }}
        html.setAttribute('data-theme', current === 'dark' ? 'light' : 'dark');
    }});
}})();
</script>
</body>
</html>
"""
