#!/usr/bin/env python3
"""TurboQuant Hardware Diagnostic v5 — Python rewrite.

Comprehensive benchmark + device profiling + load monitoring.
Supported on Apple Silicon/macOS and native Linux. WSL runs with degraded host
telemetry; native Windows and Intel macOS fail before benchmarking.

NO PII collected — only hardware specs, load stats, and benchmark numbers.

Usage:
    python3 turbo_hardware_diag.py /path/to/llama.cpp /path/to/model.gguf
    python3 turbo_hardware_diag.py --model /path/to/model.gguf --llama-dir /path/to/llama.cpp
    python3 turbo_hardware_diag.py --help

Output: turbo-diag-<date>.zip containing:
    turbo-diag-<date>.txt       (human-readable log, compatible with hw_replay.py)
    turbo-hwprofile-<date>.json (machine-parseable hardware profile)
    turbo-monitor-<date>.csv    (background system metrics polled every 10s)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional rich import — graceful fallback to plain text
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:  # pragma: no cover
    HAS_RICH = False  # pragma: no cover

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIAG_VERSION = 5

STRESS_DEPTHS = [2048, 3072, 4096, 6144, 8192, 12288, 16384, 20480, 24576, 28672, 32768]
PREFILL_DEPTHS = [2048, 4096, 8192, 16384, 32768]
DECODE_DEPTHS = [0, 4096, 8192, 16384, 32768]  # 0 = short context (no -d flag)
COMBINED_CONFIGS = [
    (4096, 128),
    (8192, 256),
    (16384, 512),
]

# Cache types to benchmark
CACHE_TYPES = ["q8_0", "turbo3"]
# turbo3 mode2 uses env var TURBO_LAYER_ADAPTIVE=2
MODE2_ENV = "TURBO_LAYER_ADAPTIVE=2"

MONITOR_POLL_INTERVAL = 10  # seconds
PLATFORM_SUPPORT_SCHEMA = "turbo_diag.platform_support.v1"
ALL_DIAGNOSTIC_SECTIONS = tuple(range(1, 14))


class UnsupportedDiagnosticPlatform(RuntimeError):
    """Raised when a host/backend combination cannot produce valid evidence."""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    """ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch() -> float:
    return time.monotonic()


def _safe_int(val: str) -> int:
    """Parse an integer from a string that may contain non-digit chars."""
    try:
        return int(re.sub(r"[^\d-]", "", val.strip()))
    except (ValueError, TypeError):
        return 0


def detect_storage_type(model_path: str, plat: str) -> str:
    """Detect if model file is on SSD or HDD. Returns 'ssd', 'hdd', or 'unknown'."""
    try:
        if plat == "Darwin":
            # Get the mount point for the model file
            result = subprocess.run(
                ["diskutil", "info", "-plist", "/"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "Solid State" in result.stdout or "SolidState" in result.stdout:
                return "ssd"
            # Apple Silicon Macs are always SSD
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "Apple" in cpu.stdout:
                return "ssd"
            return "unknown"
        elif plat == "Linux":
            # Find the block device for the model file's mount point
            model_real = os.path.realpath(model_path)
            result = subprocess.run(
                ["df", model_real],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    dev = lines[1].split()[0]  # /dev/sda1, /dev/nvme0n1p1, etc.
                    # NVMe devices are always SSD
                    if "nvme" in dev:
                        return "ssd"
                    # Check /sys/block for rotational flag
                    base_dev = re.sub(r"[0-9]+$", "", os.path.basename(dev))
                    base_dev = re.sub(r"p[0-9]+$", "", base_dev)  # nvme0n1p1 → nvme0n1
                    rotational_path = f"/sys/block/{base_dev}/queue/rotational"
                    if os.path.exists(rotational_path):
                        with open(rotational_path) as f:
                            return "hdd" if f.read().strip() == "1" else "ssd"
            return "unknown"
    except Exception:
        # Storage detection is best-effort; caller logs the fallback value
        return "unknown"
    return "unknown"


def _find_model(llama_dir: str) -> Optional[str]:
    """Auto-detect a .gguf model file near the llama.cpp directory."""
    search_dirs = [
        Path(llama_dir) / "models",
        Path(llama_dir).parent / "models",
    ]
    # Also check common local_llms paths on macOS
    home = Path.home()
    search_dirs.append(home / "local_llms" / "models")

    for d in search_dirs:
        if d.is_dir():
            for f in sorted(d.rglob("*.gguf")):
                return str(f)
    return None


# ---------------------------------------------------------------------------
# DiagLog — dual-writes to stdout and log file simultaneously
# ---------------------------------------------------------------------------
class DiagLog:
    """Writes every line to both stdout and a log file."""

    def __init__(self, log_path: str, verbose: bool = False):
        self._path = log_path
        self._fh = open(log_path, "w", encoding="utf-8")
        self._verbose = verbose
        self._lock = threading.Lock()

    def write(self, msg: str) -> None:
        with self._lock:
            self._fh.write(msg + "\n")
            self._fh.flush()
            print(msg, flush=True)

    def write_file_only(self, msg: str) -> None:
        """Write to log file without printing to stdout (for raw bench output)."""
        with self._lock:
            self._fh.write(msg + "\n")
            self._fh.flush()

    def verbose(self, msg: str) -> None:
        if self._verbose:
            self.write(f"  [VERBOSE] {msg}")

    def section(self, title: str) -> None:
        self.write("")
        self.write("=" * 64)
        self.write(f"  {title}")
        self.write("=" * 64)
        self.write("")

    def subsection(self, title: str) -> None:
        self.write("")
        self.write(f"--- {title} ---")

    def warning(self, msg: str) -> None:
        self.write(f"[WARNING] {msg}")

    def anomaly(self, msg: str) -> None:
        self.write(f"[ANOMALY] {msg}")

    def notable(self, msg: str) -> None:
        """Flag something noteworthy — could be good or suspicious."""
        self.write(f"[NOTABLE] {msg}")

    def investigate(self, msg: str) -> None:
        """Flag something outlandish that warrants further investigation."""
        self.write(f"[INVESTIGATE] {msg}")

    def close(self) -> None:
        self._fh.close()

    @property
    def path(self) -> str:
        return self._path


# ---------------------------------------------------------------------------
# BackgroundMonitor — polls system metrics every 10s, writes CSV
# ---------------------------------------------------------------------------
class BackgroundMonitor(threading.Thread):
    """Background thread that polls system metrics and writes to CSV."""

    def __init__(self, csv_path: str):
        super().__init__(daemon=True)
        self._csv_path = csv_path
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._sample_count = 0
        self._samples: list[dict] = []

        # Write CSV header
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "load_1m",
                    "mem_pressure_pct",
                    "swap_used_mb",
                    "gpu_temp_c",
                    "cpu_speed_limit",
                    "gpu_mem_used_mb",
                    "gpu_util_pct",
                ]
            )

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                sample = self._poll()
                with self._lock:
                    self._samples.append(sample)
                    self._sample_count += 1
                with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            sample["timestamp"],
                            sample["load_1m"],
                            sample["mem_pressure_pct"],
                            sample["swap_used_mb"],
                            sample["gpu_temp_c"],
                            sample["cpu_speed_limit"],
                            sample["gpu_mem_used_mb"],
                            sample["gpu_util_pct"],
                        ]
                    )
            except Exception as e:
                # Don't crash monitor, but track failures
                with self._lock:
                    self._samples.append({"error": str(e), "timestamp": _utc_now()})
            self._stop_event.wait(MONITOR_POLL_INTERVAL)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=5)

    @property
    def sample_count(self) -> int:
        with self._lock:
            return self._sample_count

    @property
    def samples(self) -> list[dict]:
        with self._lock:
            return list(self._samples)

    @property
    def csv_path(self) -> str:
        return self._csv_path

    def _poll(self) -> dict:
        ts = _utc_now()
        load_1m = "N/A"
        mem_pct = "0"
        swap_mb = "0"
        gpu_temp = "N/A"
        cpu_limit = "N/A"
        gpu_mem = "N/A"
        gpu_util = "N/A"

        plat = platform.system()

        try:
            getloadavg = getattr(os, "getloadavg", None)
            if getloadavg is not None:
                load_1m = str(round(getloadavg()[0], 2))
        except Exception:
            pass  # Expected probe failure on platforms without getloadavg

        if plat == "Darwin":
            mem_pct = self._macos_mem_pressure()
            swap_mb = self._macos_swap_mb()
            gpu_temp = "N/A"  # macOS doesn't expose GPU temp without sudo
            cpu_limit = self._macos_cpu_speed_limit()
            gpu_mem = "unified"
        elif plat == "Linux":
            mem_pct = self._linux_mem_pct()
            swap_mb = self._linux_swap_mb()
            gpu_temp = self._nvidia_query("temperature.gpu")
            gpu_mem = self._nvidia_query("memory.used")
            gpu_util = self._nvidia_query("utilization.gpu")

        return {
            "timestamp": ts,
            "load_1m": load_1m,
            "mem_pressure_pct": mem_pct,
            "swap_used_mb": swap_mb,
            "gpu_temp_c": gpu_temp,
            "cpu_speed_limit": cpu_limit,
            "gpu_mem_used_mb": gpu_mem,
            "gpu_util_pct": gpu_util,
        }

    @staticmethod
    def _macos_mem_pressure() -> str:
        try:
            out = subprocess.check_output(["vm_stat"], text=True, timeout=5)
            active = wired = free = 0
            for line in out.splitlines():
                if "Pages active" in line:
                    active = _safe_int(line.split(":")[1])
                elif "Pages wired" in line:
                    wired = _safe_int(line.split(":")[1])
                elif "Pages free" in line:
                    free = _safe_int(line.split(":")[1])
            total = active + wired + free
            if total > 0:
                return str(round((active + wired) * 100 / total))
            return "0"
        except Exception:
            return "0"  # Expected probe failure — vm_stat not available or parse error

    @staticmethod
    def _macos_swap_mb() -> str:
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "vm.swapusage"], text=True, timeout=5
            )
            m = re.search(r"used\s*=\s*([\d.]+)M", out)
            return m.group(1) if m else "0"
        except Exception:
            return "0"  # Expected probe failure — sysctl vm.swapusage not available

    @staticmethod
    def _macos_cpu_speed_limit() -> str:
        try:
            out = subprocess.check_output(
                ["pmset", "-g", "therm"], text=True, timeout=5
            )
            m = re.search(r"CPU_Speed_Limit\s+(\d+)", out)
            return m.group(1) if m else "100"
        except Exception:
            return "100"  # Expected probe failure — pmset not available

    @staticmethod
    def _linux_mem_pct() -> str:
        try:
            out = subprocess.check_output(["free"], text=True, timeout=5)
            for line in out.splitlines():
                if line.startswith("Mem:"):
                    parts = line.split()
                    return str(round(int(parts[2]) * 100 / int(parts[1])))
            return "0"
        except Exception:
            return "0"  # Expected probe failure — free command not available

    @staticmethod
    def _linux_swap_mb() -> str:
        try:
            out = subprocess.check_output(["free", "-m"], text=True, timeout=5)
            for line in out.splitlines():
                if line.startswith("Swap:"):
                    return line.split()[2]
            return "0"
        except Exception:
            return "0"  # Expected probe failure — free -m not available

    @staticmethod
    def _nvidia_query(field: str) -> str:
        """Query nvidia-smi. For multi-GPU, returns sum/max/first depending on field."""
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                text=True,
                timeout=5,
            )
            lines = [line.strip() for line in out.strip().split("\n") if line.strip()]
            if not lines:
                return "N/A"
            if len(lines) == 1:
                return lines[0]
            # Multi-GPU: aggregate sensibly
            try:
                vals = [float(v) for v in lines]
                if "memory" in field or "util" in field:
                    return str(int(sum(vals)))  # Sum memory/utilization across GPUs
                elif "temp" in field:
                    return str(int(max(vals)))  # Max temperature (hottest GPU)
                else:
                    return lines[0]  # Default: first GPU
            except ValueError:
                return ";".join(lines)  # Non-numeric: join all
        except Exception:
            return "N/A"  # Expected probe failure — nvidia-smi not available


# ---------------------------------------------------------------------------
# LiveDisplay — real-time terminal output
# ---------------------------------------------------------------------------
class LiveDisplay:
    """Real-time terminal display for benchmark progress.

    Uses rich.Live if available, falls back to ASCII bar charts.
    """

    def __init__(self, use_rich: bool = True):
        self._use_rich = use_rich and HAS_RICH
        self._decode_results: dict[
            str, dict[int, float]
        ] = {}  # cache_type -> {depth: tps}
        self._ratios: dict[int, float] = {}  # depth -> turbo3/q8_0 ratio
        self._live: Optional[Live] = None
        self._console: Optional[Console] = None

        if self._use_rich:
            self._console = Console()

    def start(self) -> None:
        if self._use_rich and self._console:
            self._live = Live(console=self._console, refresh_per_second=4)
            self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def update_decode(self, cache_type: str, depth: int, tps: float) -> None:
        """Record a decode result and update the display."""
        if cache_type not in self._decode_results:
            self._decode_results[cache_type] = {}
        self._decode_results[cache_type][depth] = tps
        self._recompute_ratios()
        self._refresh()

    def _recompute_ratios(self) -> None:
        q8 = self._decode_results.get("q8_0", {})
        turbo = self._decode_results.get("turbo3", {})
        self._ratios = {}
        for depth in set(q8.keys()) & set(turbo.keys()):
            if q8[depth] > 0:
                self._ratios[depth] = turbo[depth] / q8[depth]

    def _refresh(self) -> None:
        if self._use_rich and self._live:
            self._live.update(self._build_rich_table())
        # ASCII fallback prints on each section completion (see show_section_summary)

    def _build_rich_table(self) -> Table:
        """Build a rich Table showing the decode curve."""
        table = Table(title="Decode Curve (turbo3/q8_0)", show_lines=True)
        table.add_column("Depth", justify="right")
        table.add_column("q8_0 (tok/s)", justify="right")
        table.add_column("turbo3 (tok/s)", justify="right")
        table.add_column("Ratio", justify="right")
        table.add_column("", justify="left")  # bar

        q8 = self._decode_results.get("q8_0", {})
        turbo = self._decode_results.get("turbo3", {})

        for depth in sorted(set(q8.keys()) | set(turbo.keys())):
            q8_val = q8.get(depth, 0)
            t3_val = turbo.get(depth, 0)
            ratio = self._ratios.get(depth)

            depth_str = f"{depth:,}" if depth > 0 else "short"
            q8_str = f"{q8_val:.1f}" if q8_val else "..."
            t3_str = f"{t3_val:.1f}" if t3_val else "..."

            if ratio is not None:
                ratio_text = Text(f"{ratio:.3f}x")
                if ratio >= 0.9:
                    ratio_text.stylize("bold green")
                elif ratio >= 0.7:
                    ratio_text.stylize("bold yellow")
                else:
                    ratio_text.stylize("bold red")

                # Bar
                bar_len = int(ratio * 12)
                bar = "\u2588" * bar_len + "\u2591" * (12 - bar_len)
                table.add_row(depth_str, q8_str, t3_str, ratio_text, bar)
            else:
                table.add_row(depth_str, q8_str, t3_str, "...", "")

        return table

    def show_section_summary(self, section_name: str) -> None:
        """Print ASCII bar chart summary for a section (non-rich fallback)."""
        if self._use_rich:
            return  # rich Live handles display

        if not self._ratios:
            return

        print(f"\n  {section_name} turbo3/q8_0:", flush=True)
        for depth in sorted(self._ratios.keys()):
            ratio = self._ratios[depth]
            bar_len = int(ratio * 12)
            bar = "\u2588" * bar_len + "\u2591" * (12 - bar_len)
            depth_label = f"{depth // 1024}K" if depth >= 1024 else "short"
            print(f"  {depth_label:>5s} {bar} {ratio:.2f}x", flush=True)

    def show_stress_summary(self, label: str, stress_ratios: dict[int, float]) -> None:
        """Print ASCII bar chart for stress test ratios."""
        if not stress_ratios:
            return

        if self._use_rich and self._console:
            table = Table(title=f"Stress Test: {label}")
            table.add_column("Depth", justify="right")
            table.add_column("Ratio", justify="right")
            table.add_column("", justify="left")
            for depth in sorted(stress_ratios.keys()):
                r = stress_ratios[depth]
                ratio_text = Text(f"{r:.3f}x")
                if r >= 0.9:
                    ratio_text.stylize("bold green")
                elif r >= 0.7:
                    ratio_text.stylize("bold yellow")
                else:
                    ratio_text.stylize("bold red")
                bar_len = int(r * 12)
                bar = "\u2588" * bar_len + "\u2591" * (12 - bar_len)
                table.add_row(f"{depth:,}", ratio_text, bar)
            self._console.print(table)
        else:
            line_parts = []
            for depth in sorted(stress_ratios.keys()):
                r = stress_ratios[depth]
                bar_len = int(r * 12)
                bar = "\u2588" * bar_len + "\u2591" * (12 - bar_len)
                depth_label = f"{depth // 1024}K"
                line_parts.append(f"{depth_label} {bar} {r:.2f}x")
            label_short = label.replace("Decode ", "")
            print(f"  {label_short}:  {'  '.join(line_parts)}", flush=True)


# ---------------------------------------------------------------------------
# AnomalyDetector — checks results as they come in
# ---------------------------------------------------------------------------
class AnomalyDetector:
    """Checks benchmark results for anomalies in real-time."""

    def __init__(self, log: DiagLog, monitor: BackgroundMonitor):
        self._log = log
        self._monitor = monitor
        self._prev_decode_ratio: Optional[float] = None
        self._prev_depth: int = 0
        self._initial_swap_mb: float = 0.0
        self._q8_short_tps: float = 0.0
        self._q8_ppl: float = 0.0
        self._anomalies: list[str] = []
        self._notables: list[str] = []
        self._investigations: list[str] = []

    @property
    def anomalies(self) -> list[str]:
        return list(self._anomalies)

    @property
    def notables(self) -> list[str]:
        return list(self._notables)

    @property
    def investigations(self) -> list[str]:
        return list(self._investigations)

    def _flag_notable(self, msg: str) -> None:
        self._log.notable(msg)
        self._notables.append(msg)

    def _flag_investigate(self, msg: str) -> None:
        self._log.investigate(msg)
        self._investigations.append(msg)

    def set_initial_swap(self, swap_mb: float) -> None:
        self._initial_swap_mb = swap_mb

    def set_q8_short_decode(self, tps: float) -> None:
        self._q8_short_tps = tps

    def set_q8_ppl(self, ppl: float) -> None:
        self._q8_ppl = ppl

    def check_decode_ratio(self, depth: int, ratio: float) -> None:
        """Check decode ratio for anomalies in BOTH directions."""
        # Suspiciously good: turbo3 faster than q8_0 shouldn't happen
        if ratio > 1.05:
            self._flag_investigate(
                f"turbo3 FASTER than q8_0 at {depth}: {ratio:.3f}x. "
                f"Measurement error? System load changed between runs?"
            )
        # Impressively good: near-parity at long context is notable
        elif ratio > 0.98 and depth >= 8192:
            self._flag_notable(
                f"Excellent decode ratio at {depth}: {ratio:.3f}x — "
                f"near q8_0 parity at long context"
            )

        # Steep degradation between depths
        if self._prev_decode_ratio is not None and self._prev_decode_ratio > 0:
            drop = (self._prev_decode_ratio - ratio) / self._prev_decode_ratio
            if drop > 0.15:
                msg = (
                    f"Steep decode degradation: {self._prev_decode_ratio:.3f}x "
                    f"@ {self._prev_depth} -> {ratio:.3f}x @ {depth} "
                    f"({drop:.0%} drop)"
                )
                self._log.anomaly(msg)
                self._anomalies.append(msg)
            elif drop < -0.05 and depth > self._prev_depth:
                # Decode getting FASTER at deeper context — very suspicious
                self._flag_investigate(
                    f"Decode IMPROVED at deeper context: {self._prev_decode_ratio:.3f}x "
                    f"@ {self._prev_depth} -> {ratio:.3f}x @ {depth}. "
                    f"Measurement noise or system load change?"
                )

        # Absolute threshold: below 0.5x is always a red flag
        if ratio < 0.50 and depth > 0:
            self._flag_investigate(
                f"Decode ratio {ratio:.3f}x at {depth} — below 0.5x threshold. "
                f"Constant cache thrashing likely. Consider q8_0 or TURBO_LAYER_ADAPTIVE=2."
            )

        self._prev_decode_ratio = ratio
        self._prev_depth = depth

    def check_thermal(self) -> None:
        """Check if CPU speed limit has dropped below 100%."""
        samples = self._monitor.samples
        if not samples:
            return
        latest = samples[-1]
        try:
            limit = int(latest.get("cpu_speed_limit", "100"))
            if limit < 100:
                msg = f"Thermal throttling detected: CPU_Speed_Limit={limit}%"
                self._log.anomaly(msg)
                self._anomalies.append(msg)
        except (ValueError, TypeError):
            pass

    def check_swap_growth(self) -> None:
        """Check if swap has grown >100MB during benchmarks."""
        samples = self._monitor.samples
        if not samples:
            return
        try:
            current_swap = float(samples[-1].get("swap_used_mb", "0"))
            growth = current_swap - self._initial_swap_mb
            if growth > 100:
                msg = (
                    f"Memory pressure/swapping: swap grew {growth:.0f}MB "
                    f"({self._initial_swap_mb:.0f} -> {current_swap:.0f} MB)"
                )
                self._log.anomaly(msg)
                self._anomalies.append(msg)
        except (ValueError, TypeError):
            pass

    def check_q8_baseline(self, tps: float, hardware_class: str = "unknown") -> None:
        """Check if q8_0 baseline is suspiciously low for the hardware class."""
        # TODO: Build a proper hardware class -> expected baseline mapping
        # For now, use a rough heuristic: Apple Silicon should do >20 tok/s decode
        if hardware_class in ("apple_silicon", "unknown") and tps > 0 and tps < 5:
            msg = f"System under load: q8_0 short decode only {tps:.1f} tok/s"
            self._log.anomaly(msg)
            self._anomalies.append(msg)

    def check_ppl(self, cache_type: str, ppl: float, env: str = "") -> None:
        """Check PPL quality against q8_0 baseline — both directions."""
        if self._q8_ppl <= 0:
            return
        delta_pct = (ppl - self._q8_ppl) / self._q8_ppl * 100
        label = f"{cache_type}" + (f" ({env})" if env else "")

        if delta_pct > 10:
            msg = f"Quality regression: {label} PPL={ppl:.4f} is {delta_pct:.1f}% worse than q8_0 ({self._q8_ppl:.4f})"
            self._log.anomaly(msg)
            self._anomalies.append(msg)
        elif delta_pct < -1:
            # turbo3 BETTER than q8_0? That's outlandish.
            self._flag_investigate(
                f"{label} PPL={ppl:.4f} is {abs(delta_pct):.1f}% BETTER than q8_0 ({self._q8_ppl:.4f}). "
                f"Measurement noise or the quantization gods smiled on you."
            )
        elif abs(delta_pct) < 0.1 and "turbo3" in cache_type and not env:
            # turbo3 matching q8_0 within 0.1% is notable
            self._flag_notable(
                f"{label} PPL={ppl:.4f} matches q8_0 within 0.1% — excellent quality"
            )
        elif delta_pct > 2 and "turbo3" in cache_type:
            # Expected range for turbo3
            self._log.write(
                f"[INFO] {label} PPL is {delta_pct:.1f}% worse than q8_0 (expected for turbo3)"
            )

    def check_prefill_ratio(self, depth: int, ratio: float) -> None:
        """Check prefill ratio for outlandish results."""
        if ratio > 1.10:
            self._flag_investigate(
                f"turbo3 prefill {ratio:.3f}x FASTER than q8_0 at {depth}. "
                f"Unexpected — verify measurement."
            )
        elif ratio > 1.02 and depth >= 8192:
            self._flag_notable(
                f"turbo3 prefill {ratio:.3f}x at {depth} — slightly faster than q8_0 "
                f"(smaller KV cache = less memory pressure)"
            )
        elif ratio < 0.90:
            self._flag_investigate(
                f"turbo3 prefill only {ratio:.3f}x of q8_0 at {depth}. "
                f"Expected >0.95x. Possible context scaling regression?"
            )


# ---------------------------------------------------------------------------
# Platform support contract and detection
# ---------------------------------------------------------------------------
def detect_platform() -> str:
    return platform.system()


def get_platform_support(
    *,
    system: Optional[str] = None,
    release: Optional[str] = None,
    machine: Optional[str] = None,
    environ: Optional[dict[str, str]] = None,
) -> dict:
    """Return the declared support contract for a host environment.

    Optional inputs make the policy independently testable without impersonating
    platform modules. The returned dictionary is embedded in diagnostic JSON.
    """
    detected_system = system if system is not None else detect_platform()
    detected_release = release if release is not None else platform.release()
    detected_machine = machine if machine is not None else platform.machine()
    detected_environ = os.environ if environ is None else environ
    release_lower = detected_release.lower()
    machine_lower = detected_machine.lower()
    is_wsl = detected_system == "Linux" and (
        "microsoft" in release_lower
        or "wsl" in release_lower
        or bool(detected_environ.get("WSL_INTEROP"))
        or bool(detected_environ.get("WSL_DISTRO_NAME"))
    )

    def contract(
        *,
        environment: str,
        status: str,
        reason: str,
        action: str,
        monitoring: str,
        supported_backends: list[str],
        degraded_backends: list[str],
        unavailable_backends: list[str],
        supported_sections: list[int],
        degraded_sections: list[int],
        unavailable_sections: list[int],
    ) -> dict:
        return {
            "schema": PLATFORM_SUPPORT_SCHEMA,
            "platform": detected_system,
            "release": detected_release,
            "machine": detected_machine,
            "environment": environment,
            "status": status,
            "reason": reason,
            "action": action,
            "monitoring": monitoring,
            "gpu_backends": {
                "supported": supported_backends,
                "degraded": degraded_backends,
                "unavailable": unavailable_backends,
            },
            "sections": {
                "supported": supported_sections,
                "degraded": degraded_sections,
                "skipped": [],
                "unavailable": unavailable_sections,
            },
        }

    all_sections = list(ALL_DIAGNOSTIC_SECTIONS)
    if detected_system == "Darwin" and machine_lower in {"arm64", "aarch64"}:
        return contract(
            environment="macos-apple-silicon",
            status="supported",
            reason="Apple Silicon hardware and Metal telemetry are supported.",
            action="Run the Bash launcher or the Python diagnostic directly.",
            monitoring="supported",
            supported_backends=["metal"],
            degraded_backends=[],
            unavailable_backends=["cuda", "rocm", "vulkan", "cpu"],
            supported_sections=all_sections,
            degraded_sections=[],
            unavailable_sections=[],
        )
    if detected_system == "Darwin":
        return contract(
            environment="macos-intel",
            status="unsupported",
            reason="The macOS diagnostic contract covers Apple Silicon only.",
            action="Use a supported Apple Silicon or native Linux host.",
            monitoring="unavailable",
            supported_backends=[],
            degraded_backends=[],
            unavailable_backends=["metal", "cuda", "rocm", "vulkan", "cpu"],
            supported_sections=[],
            degraded_sections=[],
            unavailable_sections=all_sections,
        )
    if is_wsl:
        return contract(
            environment="wsl",
            status="degraded",
            reason=(
                "WSL can run Linux llama.cpp binaries, but host thermal, power, "
                "storage, and GPU-topology telemetry is incomplete."
            ),
            action=(
                "Use a WSL CUDA build and treat hardware/load sections as "
                "partial evidence."
            ),
            monitoring="degraded",
            supported_backends=["cuda"],
            degraded_backends=[],
            unavailable_backends=["metal", "rocm", "vulkan", "cpu"],
            supported_sections=[3, 5, 6, 7, 8, 9, 10, 11, 13],
            degraded_sections=[1, 2, 4, 12],
            unavailable_sections=[],
        )
    if detected_system == "Linux":
        return contract(
            environment="linux",
            status="supported",
            reason="Native Linux diagnostic and monitoring probes are supported.",
            action="Use a matching Linux llama.cpp build for the selected backend.",
            monitoring="supported",
            supported_backends=["cuda"],
            degraded_backends=["rocm", "vulkan", "cpu"],
            unavailable_backends=["metal"],
            supported_sections=all_sections,
            degraded_sections=[],
            unavailable_sections=[],
        )
    if detected_system == "Windows":
        return contract(
            environment="windows-native",
            status="unsupported",
            reason=(
                "Native Windows lacks the Linux/macOS telemetry contract and "
                "the supported Bash launcher."
            ),
            action=(
                "Run the diagnostic inside WSL with Linux llama.cpp binaries, "
                "or use a native Linux/macOS host."
            ),
            monitoring="unavailable",
            supported_backends=[],
            degraded_backends=[],
            unavailable_backends=["metal", "cuda", "rocm", "vulkan", "cpu"],
            supported_sections=[],
            degraded_sections=[],
            unavailable_sections=all_sections,
        )
    return contract(
        environment="unknown",
        status="unsupported",
        reason=f"No diagnostic support contract exists for {detected_system!r}.",
        action="Use a supported Apple Silicon/macOS or native Linux host.",
        monitoring="unavailable",
        supported_backends=[],
        degraded_backends=[],
        unavailable_backends=["metal", "cuda", "rocm", "vulkan", "cpu"],
        supported_sections=[],
        degraded_sections=[],
        unavailable_sections=all_sections,
    )


def apply_runtime_skips(support: dict, *, skip_stress: bool, skip_ppl: bool) -> dict:
    """Record CLI-requested section skips in a support contract copy."""
    result = copy.deepcopy(support)
    skipped: list[int] = []
    if skip_stress:
        skipped.append(8)
    if skip_ppl:
        skipped.append(10)
    for section in skipped:
        for state in ("supported", "degraded", "unavailable"):
            if section in result["sections"][state]:
                result["sections"][state].remove(section)
        if section not in result["sections"]["skipped"]:
            result["sections"]["skipped"].append(section)
    result["sections"]["skipped"].sort()
    return result


def detect_runtime_backend(support: dict, gpu_init: str, hw: dict) -> str:
    """Infer the active llama.cpp backend from initialization evidence."""
    init_lower = gpu_init.lower()
    if re.search(r"\brocm\b|\bhip\b", init_lower):
        return "rocm"
    if "cuda" in init_lower or "cublas" in init_lower:
        return "cuda"
    if "metal" in init_lower:
        return "metal"
    if "vulkan" in init_lower:
        return "vulkan"

    hardware_backend = str(hw.get("gpu_backend", "")).lower()
    if hardware_backend == "cuda":
        return "cuda"
    if hardware_backend in {"rocm", "hip"}:
        return "rocm"
    if hardware_backend == "vulkan_or_other":
        return "vulkan"
    if support["environment"] == "macos-apple-silicon":
        return "metal"
    if "cpu" in init_lower or gpu_init.strip():
        return "cpu"
    return "unknown"


def assess_runtime_backend(support: dict, gpu_init: str, hw: dict) -> dict:
    """Refine host support with the backend detected before long benchmarks."""
    result = copy.deepcopy(support)
    backend = detect_runtime_backend(result, gpu_init, hw)
    result["detected_gpu_backend"] = backend
    backend_contract = result["gpu_backends"]
    if backend in backend_contract["supported"]:
        return result
    if backend in backend_contract["degraded"]:
        result["status"] = "degraded"
        result["reason"] = (
            f"{result['reason']} Backend {backend!r} is experimental in this "
            "diagnostic and may have partial telemetry."
        )
        return result

    result["status"] = "unsupported"
    result["reason"] = (
        f"Detected backend {backend!r} is unavailable for environment "
        f"{result['environment']!r}."
    )
    available = backend_contract["supported"] + backend_contract["degraded"]
    if available:
        result["action"] = "Use one of the declared backends: " + ", ".join(available)
    for section in range(5, 14):
        for state in ("supported", "degraded", "skipped"):
            if section in result["sections"][state]:
                result["sections"][state].remove(section)
        if section not in result["sections"]["unavailable"]:
            result["sections"]["unavailable"].append(section)
    result["sections"]["unavailable"].sort()
    return result


def log_platform_support(log: DiagLog, support: dict) -> None:
    """Write stable human-readable platform support tags."""
    backend = support.get("detected_gpu_backend", "not-yet-detected")
    log.write(
        "[PLATFORM_SUPPORT] "
        f"environment={support['environment']} status={support['status']} "
        f"monitoring={support['monitoring']} backend={backend}"
    )
    log.write(f"[PLATFORM_SUPPORT] reason={support['reason']}")
    log.write(f"[PLATFORM_SUPPORT] action={support['action']}")
    for state in ("supported", "degraded", "skipped", "unavailable"):
        values = support["sections"][state]
        rendered = ",".join(str(value) for value in values) if values else "none"
        log.write(f"[PLATFORM_SECTIONS] {state}={rendered}")


def detect_hardware(log: DiagLog) -> dict:
    """Collect hardware inventory. Returns dict for JSON profile."""
    hw: dict = {}
    plat = detect_platform()

    log.write(
        f"[HW] os={plat} os_version={platform.release()} arch={platform.machine()}"
    )
    hw["platform"] = plat
    hw["os_version"] = platform.release()
    hw["arch"] = platform.machine()

    if plat == "Darwin":
        _detect_macos_hw(log, hw)
    elif plat == "Linux":
        _detect_linux_hw(log, hw)
    else:
        log.warning(f"Unsupported platform: {plat}")

    return hw


def _run_cmd(cmd: list[str] | str, timeout: int = 10, shell: bool = False) -> str:
    """Run a command, return stdout. Never raises."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=shell,
        )
        return result.stdout.strip()
    except Exception:
        # Expected probe failure — command may not exist on this platform
        return ""


def _sysctl(key: str) -> str:
    return _run_cmd(["sysctl", "-n", key])


def _detect_macos_hw(log: DiagLog, hw: dict) -> None:
    try:
        kernel = _run_cmd(["sw_vers", "-productVersion"]) or platform.release()
        log.write(f"[HW] kernel={kernel}")
    except Exception as e:
        log.warning(f"Could not get macOS version: {e}")

    try:
        cpu_brand = _sysctl("machdep.cpu.brand_string") or "unknown"
        log.write(f"[HW] cpu_brand={cpu_brand}")
        hw["cpu_brand"] = cpu_brand
    except Exception as e:
        log.warning(f"Could not get CPU brand: {e}")
        hw["cpu_brand"] = "unknown"

    try:
        cores_phys = _sysctl("hw.physicalcpu") or "unknown"
        cores_log = _sysctl("hw.logicalcpu") or "unknown"
        log.write(f"[HW] cpu_cores_physical={cores_phys}")
        log.write(f"[HW] cpu_cores_logical={cores_log}")
        hw["cpu_cores_physical"] = _safe_int(cores_phys)
        hw["cpu_cores_logical"] = _safe_int(cores_log)
    except Exception as e:
        log.warning(f"Could not get CPU core count: {e}")

    try:
        freq_max = _sysctl("hw.cpufrequency_max")
        if freq_max:
            freq_mhz = int(freq_max) // 1_000_000
            log.write(f"[HW] cpu_freq_max={freq_mhz} MHz")
    except Exception as e:
        log.warning(f"Could not get CPU max frequency: {e}")

    try:
        memsize = _sysctl("hw.memsize")
        if memsize:
            ram_bytes = int(memsize)
            ram_gb = ram_bytes // (1024**3)
            log.write(f"[HW] ram_total_bytes={ram_bytes}")
            log.write(f"[HW] ram_total_gb={ram_gb}")
            hw["ram_total_gb"] = ram_gb
        else:
            hw["ram_total_gb"] = 0
    except Exception as e:
        log.warning(f"Could not get RAM size: {e}")
        hw["ram_total_gb"] = 0

    # GPU from system_profiler
    try:
        log.subsection("GPU Details")
        gpu_out = _run_cmd(["system_profiler", "SPDisplaysDataType"], timeout=15)
        for line in gpu_out.splitlines():
            stripped = line.strip()
            if any(
                kw in stripped
                for kw in ("Chipset", "Total Number", "Metal", "Cores", "Model")
            ):
                log.write(f"[HW_GPU] {stripped}")
    except Exception as e:
        log.warning(f"Could not query system_profiler for GPU info: {e}")

    # Apple Silicon
    try:
        chip = hw.get("cpu_brand", "")
        if "Apple" in chip:
            log.write("[HW] apple_silicon=true")
            log.write(f"[HW] chip_model={chip}")
            hw["apple_silicon"] = True
            hw["chip_model"] = chip
        else:
            log.write("[HW] apple_silicon=false")
            hw["apple_silicon"] = False
    except Exception as e:
        log.warning(f"Could not determine Apple Silicon status: {e}")
        hw["apple_silicon"] = False

    # Cache hierarchy
    try:
        page_size = _sysctl("hw.pagesize")
        l1_dc = _sysctl("hw.l1dcachesize")
        l2 = _sysctl("hw.l2cachesize")
        log.write(f"[HW] page_size={page_size}")
        log.write(f"[HW] l1_dcache={l1_dc}")
        log.write(f"[HW] l2_cache={l2}")
        hw["l1_dcache"] = _safe_int(l1_dc) if l1_dc else 0
        hw["l2_cache"] = _safe_int(l2) if l2 else 0
    except Exception as e:
        log.warning(f"Could not get cache hierarchy: {e}")

    # Power state
    try:
        log.subsection("Power State")
        pmset_out = _run_cmd(["pmset", "-g", "batt"])
        for line in pmset_out.splitlines():
            if any(
                kw in line for kw in ("Now drawing", "charging", "AC Power", "Battery")
            ):
                log.write(f"[HW_POWER] {line.strip()}")
        therm_out = _run_cmd(["pmset", "-g", "therm"])
        m = re.search(r"CPU_Speed_Limit\s+(\d+)", therm_out)
        cpu_limit = m.group(1) if m else "unknown"
        log.write(f"[HW_POWER] cpu_speed_limit={cpu_limit}")
    except Exception as e:
        log.warning(f"Could not get power state: {e}")


def _detect_linux_hw(log: DiagLog, hw: dict) -> None:
    try:
        log.write(f"[HW] kernel={platform.release()}")
    except Exception as e:
        log.warning(f"Could not get Linux kernel version: {e}")

    # CPU
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        m = re.search(r"model name\s*:\s*(.*)", cpuinfo)
        cpu_brand = m.group(1).strip() if m else "unknown"
        log.write(f"[HW] cpu_brand={cpu_brand}")
        hw["cpu_brand"] = cpu_brand

        core_ids = set(re.findall(r"core id\s*:\s*(\d+)", cpuinfo))
        processors = len(re.findall(r"^processor\s*:", cpuinfo, re.MULTILINE))
        log.write(f"[HW] cpu_cores_physical={len(core_ids)}")
        log.write(f"[HW] cpu_cores_logical={processors}")
        hw["cpu_cores_physical"] = len(core_ids)
        hw["cpu_cores_logical"] = processors
    except Exception as e:
        log.warning(f"Could not read /proc/cpuinfo: {e}")
        hw["cpu_brand"] = "unknown"

    # RAM
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
        m = re.search(r"MemTotal:\s*(\d+)\s*kB", meminfo)
        if m:
            ram_kb = int(m.group(1))
            ram_gb = ram_kb // (1024 * 1024)
            log.write(f"[HW] ram_total_bytes={ram_kb * 1024}")
            log.write(f"[HW] ram_total_gb={ram_gb}")
            hw["ram_total_gb"] = ram_gb
    except Exception as e:
        log.warning(f"Could not read /proc/meminfo for RAM info: {e}")
        hw["ram_total_gb"] = 0

    # GPU
    try:
        log.subsection("GPU Details")
        nvidia = shutil.which("nvidia-smi")
        if nvidia:
            gpu_out = _run_cmd(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version,gpu_bus_id",
                    "--format=csv,noheader",
                ]
            )
            for line in gpu_out.splitlines():
                log.write(f"[HW_GPU] {line.strip()}")
            log.write("[HW] gpu_backend=cuda")
            hw["gpu_backend"] = "cuda"
        elif Path("/sys/class/drm").is_dir():
            for card_dir in sorted(Path("/sys/class/drm").glob("card*/device")):
                vendor_f = card_dir / "vendor"
                device_f = card_dir / "device"
                if vendor_f.exists():
                    vendor = vendor_f.read_text(
                        encoding="utf-8", errors="replace"
                    ).strip()
                    device = (
                        device_f.read_text(encoding="utf-8", errors="replace").strip()
                        if device_f.exists()
                        else "unknown"
                    )
                    log.write(f"[HW_GPU] pci_vendor={vendor} pci_device={device}")
            log.write("[HW] gpu_backend=vulkan_or_other")
            hw["gpu_backend"] = "vulkan_or_other"
        else:
            log.write("[HW_GPU] no GPU detected")
    except Exception as e:
        log.warning(f"Could not detect GPU: {e}")

    log.write("[HW] apple_silicon=false")
    hw["apple_silicon"] = False

    # Cache hierarchy
    try:
        for level in (1, 2, 3):
            cache_path = Path(f"/sys/devices/system/cpu/cpu0/cache/index{level}/size")
            if cache_path.exists():
                size = cache_path.read_text(encoding="utf-8", errors="replace").strip()
                log.write(f"[HW] l{level}_cache={size}")
    except Exception as e:
        log.warning(f"Could not get Linux cache hierarchy: {e}")

    # Thermal
    try:
        log.subsection("Thermal")
        thermal_base = Path("/sys/class/thermal")
        if thermal_base.is_dir():
            for tz in sorted(thermal_base.glob("thermal_zone*/temp")):
                temp = tz.read_text(encoding="utf-8", errors="replace").strip()
                zone = tz.parent.name
                log.write(f"[HW_THERMAL] {zone}={temp}m\u00b0C")
    except Exception as e:
        log.warning(f"Could not read thermal zones: {e}")

    # Power
    try:
        ps_base = Path("/sys/class/power_supply")
        if ps_base.is_dir():
            for ps_dir in sorted(ps_base.iterdir()):
                ptype_f = ps_dir / "type"
                status_f = ps_dir / "status"
                ptype = (
                    ptype_f.read_text(encoding="utf-8", errors="replace").strip()
                    if ptype_f.exists()
                    else "unknown"
                )
                status = (
                    status_f.read_text(encoding="utf-8", errors="replace").strip()
                    if status_f.exists()
                    else "unknown"
                )
                log.write(f"[HW_POWER] {ps_dir.name} type={ptype} status={status}")
    except Exception as e:
        log.warning(f"Could not read power supply info: {e}")


# ---------------------------------------------------------------------------
# Load snapshot — capture system state at a point in time
# ---------------------------------------------------------------------------
def capture_load(label: str, log: DiagLog) -> None:
    """Capture a system load snapshot. Matches bash script tag format."""
    log.write("")
    log.write(f"[LOAD_SNAPSHOT] label={label} timestamp={_utc_now()}")

    plat = detect_platform()

    # Load average
    try:
        if Path("/proc/loadavg").exists():
            parts = (
                Path("/proc/loadavg")
                .read_text(encoding="utf-8", errors="replace")
                .split()
            )
            log.write(f"[LOAD_SNAPSHOT] load_avg={parts[0]} {parts[1]} {parts[2]}")
        else:
            load_avg = _run_cmd(["sysctl", "-n", "vm.loadavg"])
            cleaned = load_avg.replace("{", "").replace("}", "").strip()
            parts = cleaned.split()
            if len(parts) >= 3:
                log.write(f"[LOAD_SNAPSHOT] load_avg={parts[0]} {parts[1]} {parts[2]}")
    except Exception as e:
        log.warning(f"Could not get load average: {e}")

    if plat == "Darwin":
        _capture_load_macos(log)
    elif plat == "Linux":
        _capture_load_linux(log)

    # Process count
    try:
        ps_out = _run_cmd(["ps", "aux"])
        count = len(ps_out.splitlines())
        log.write(f"[LOAD_SNAPSHOT] process_count={count}")
    except Exception as e:
        log.write(f"[LOAD_SNAPSHOT] process_count=unknown (error: {e})")


def _capture_load_macos(log: DiagLog) -> None:
    try:
        vm_out = _run_cmd(["vm_stat"])
        if vm_out:
            pageins = "0"
            pageouts = "0"
            free_pages = 0
            inactive_pages = 0
            for line in vm_out.splitlines():
                if "Pageins" in line:
                    pageins = re.sub(r"[^\d]", "", line.split(":")[1])
                elif "Pageouts" in line:
                    pageouts = re.sub(r"[^\d]", "", line.split(":")[1])
                elif "Pages free" in line:
                    free_pages = _safe_int(line.split(":")[1])
                elif "Pages inactive" in line:
                    inactive_pages = _safe_int(line.split(":")[1])

            swap_out = _run_cmd(["sysctl", "-n", "vm.swapusage"])
            swap_used = "unknown"
            m = re.search(r"used\s*=\s*([\d.]+M)", swap_out)
            if m:
                swap_used = m.group(1)

            log.write(
                f"[LOAD_SNAPSHOT] page_ins={pageins} page_outs={pageouts} swap_used={swap_used}"
            )
            free_mb = (free_pages + inactive_pages) * 4096 // (1024 * 1024)
            log.write(f"[LOAD_SNAPSHOT] approx_free_ram={free_mb} MB")
    except Exception as e:
        log.warning(f"Could not get vm_stat: {e}")

    try:
        mp_out = _run_cmd(["memory_pressure"], timeout=30)
        for line in mp_out.splitlines():
            if "System-wide" in line:
                log.write(f"[LOAD_SNAPSHOT] memory_pressure={line.strip()}")
                break
    except Exception as e:
        log.warning(f"Could not get memory_pressure: {e}")

    # Thermal
    try:
        therm_out = _run_cmd(["pmset", "-g", "therm"])
        cpu_limit_line = ""
        for line in therm_out.splitlines():
            if "cpu_speed_limit" in line.lower():
                cpu_limit_line = line.strip()
                break
        log.write(f"[LOAD_SNAPSHOT] thermal={cpu_limit_line or 'no thermal data'}")
    except Exception as e:
        log.warning(f"Could not get thermal data from pmset: {e}")

    # GPU utilization via ioreg
    try:
        ioreg_out = _run_cmd(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])
        for line in ioreg_out.splitlines():
            if "PerformanceState" in line:
                log.write(f"[LOAD_SNAPSHOT] gpu_ioreg={line.strip()}")
                break
        else:
            log.write("[LOAD_SNAPSHOT] gpu_ioreg=no GPU metrics")
    except Exception as e:
        log.warning(f"Could not get GPU utilization from ioreg: {e}")


def _capture_load_linux(log: DiagLog) -> None:
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
        mem_free = mem_avail = swap_total = swap_free = 0
        for line in meminfo.splitlines():
            if line.startswith("MemFree:"):
                mem_free = _safe_int(line.split(":")[1])
            elif line.startswith("MemAvailable:"):
                mem_avail = _safe_int(line.split(":")[1])
            elif line.startswith("SwapTotal:"):
                swap_total = _safe_int(line.split(":")[1])
            elif line.startswith("SwapFree:"):
                swap_free = _safe_int(line.split(":")[1])
        log.write(
            f"[LOAD_SNAPSHOT] mem_free_mb={mem_free // 1024} "
            f"mem_available_mb={mem_avail // 1024}"
        )
        log.write(
            f"[LOAD_SNAPSHOT] swap_total_mb={swap_total // 1024} "
            f"swap_free_mb={swap_free // 1024}"
        )
    except Exception as e:
        log.warning(f"Could not read /proc/meminfo for load snapshot: {e}")

    # NVIDIA GPU
    try:
        if shutil.which("nvidia-smi"):
            gpu_out = _run_cmd(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader",
                ]
            )
            log.write(f"[LOAD_SNAPSHOT] gpu_util={gpu_out}")
    except Exception as e:
        log.warning(f"Could not query nvidia-smi for GPU utilization: {e}")

    # CPU temp
    try:
        temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if temp_path.exists():
            temp = temp_path.read_text(encoding="utf-8", errors="replace").strip()
            log.write(f"[LOAD_SNAPSHOT] cpu_temp={temp}m\u00b0C")
    except Exception as e:
        log.warning(f"Could not read CPU temperature: {e}")


# ---------------------------------------------------------------------------
# Subprocess runners — run_bench, run_perpl
# ---------------------------------------------------------------------------
def _run_subprocess(
    cmd: list[str],
    log: DiagLog,
    env_extra: Optional[dict[str, str]] = None,
    timeout: int = 600,
) -> tuple[str, int]:
    """Run a subprocess, streaming output to the log.

    Returns (stdout_text, return_code).
    """
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        lines: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            log.write(line)
        proc.wait(timeout=timeout)
        return "\n".join(lines), proc.returncode
    except subprocess.TimeoutExpired:
        log.warning(f"Command timed out after {timeout}s: {' '.join(cmd[:4])}...")
        try:
            proc.kill()
        except Exception as e:
            log.warning(f"Could not kill timed-out process: {e}")
        return "", -1
    except Exception as e:
        log.warning(f"Command failed: {e}")
        return "", -1


def _parse_env_string(env_str: str) -> dict[str, str]:
    """Parse 'KEY=VALUE' or 'KEY1=V1 KEY2=V2' into a dict."""
    result: dict[str, str] = {}
    if not env_str:
        return result
    for part in env_str.split():
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


def run_bench(
    label: str,
    ctk: str,
    ctv: str,
    extra_args: str,
    log: DiagLog,
    bench_bin: str,
    model: str,
    env_prefix: str = "",
) -> tuple[str, float]:
    """Run a llama-bench invocation. Returns (stdout, wall_seconds)."""
    log.subsection(label)
    log.write(
        f'[BENCH_START] label="{label}" ctk={ctk} ctv={ctv} '
        f'args="{extra_args}" env="{env_prefix}" timestamp={_utc_now()}'
    )

    cmd = [
        bench_bin,
        "-m",
        model,
        "-ngl",
        "99",
        "-fa",
        "1",
        "-ctk",
        ctk,
        "-ctv",
        ctv,
    ]
    # Split extra_args respecting the shell-like splitting
    if extra_args:
        cmd.extend(extra_args.split())
    cmd.extend(["-r", "3"])

    env_extra = _parse_env_string(env_prefix)
    start = _epoch()
    output, rc = _run_subprocess(cmd, log, env_extra=env_extra)
    wall_sec = _epoch() - start

    if rc != 0 and rc != -1:
        log.write(f"FAILED: {label} (exit code {rc})")

    log.write(f'[BENCH_END] label="{label}" wall_sec={int(wall_sec)}')
    return output, wall_sec


def run_perpl(
    label: str,
    ctk: str,
    ctv: str,
    chunks: int,
    log: DiagLog,
    perpl_bin: str,
    model: str,
    wiki_path: str,
    env_prefix: str = "",
) -> tuple[str, float]:
    """Run llama-perplexity. Returns (stdout, wall_seconds)."""
    log.subsection(label)
    log.write(
        f'[PPL_START] label="{label}" ctk={ctk} ctv={ctv} '
        f"chunks={chunks} timestamp={_utc_now()}"
    )

    cmd = [
        perpl_bin,
        "-m",
        model,
        "-ngl",
        "99",
        "-fa",
        "on",
        "--cache-type-k",
        ctk,
        "--cache-type-v",
        ctv,
        "-f",
        wiki_path,
        "--chunks",
        str(chunks),
    ]

    env_extra = _parse_env_string(env_prefix)
    start = _epoch()
    output, rc = _run_subprocess(cmd, log, env_extra=env_extra)
    wall_sec = _epoch() - start

    if rc != 0 and rc != -1:
        log.write(f"FAILED: {label} (exit code {rc})")

    log.write(f'[PPL_END] label="{label}"')
    return output, wall_sec


# ---------------------------------------------------------------------------
# Bench output parsing helpers
# ---------------------------------------------------------------------------
def parse_bench_tps(output: str) -> list[dict]:
    """Parse llama-bench table output for tok/s values.

    Returns list of dicts with keys: mode, depth, tps, stddev, ctk
    """
    results: list[dict] = []
    for line in output.splitlines():
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 10:
            continue

        # Find test column (ppXXXX or tgXXXX)
        test_col = ""
        tps_col = ""
        for i, col in enumerate(cols):
            if re.match(r"(pp|tg)\d+", col):
                test_col = col
                if i + 1 < len(cols):
                    tps_col = cols[i + 1]
                break

        if not test_col:
            continue

        # Mode + depth
        mode = ""
        depth = 0
        if test_col.startswith("pp") and "+tg" in test_col:
            mode = "combined"
            m = re.match(r"pp(\d+)\+tg(\d+)", test_col)
            if m:
                depth = int(m.group(1))
        elif test_col.startswith("pp"):
            mode = "prefill"
            m = re.match(r"pp(\d+)", test_col)
            if m:
                depth = int(m.group(1))
        elif test_col.startswith("tg"):
            mode = "decode"
            m = re.search(r"d(\d+)", test_col)
            if m:
                depth = int(m.group(1))

        # tok/s
        tps = 0.0
        stddev = 0.0
        m = re.match(r"([\d.]+)\s*\u00b1\s*([\d.]+)", tps_col)
        if m:
            tps = float(m.group(1))
            stddev = float(m.group(2))
        else:
            m = re.match(r"[\d.]+", tps_col)
            if m:
                tps = float(m.group())

        # Extract cache type from row
        row_ctk = ""
        for col in cols:
            cs = col.strip()
            if cs in ("q8_0", "turbo3", "turbo4", "f16", "q4_0"):
                row_ctk = cs
                break

        results.append(
            {
                "mode": mode,
                "depth": depth,
                "tps": tps,
                "stddev": stddev,
                "ctk": row_ctk,
            }
        )

    return results


def parse_ppl_final(output: str) -> tuple[float, float]:
    """Extract final PPL estimate from perplexity output.

    Returns (ppl, stddev) or (0, 0) if not found.
    """
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", output)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# 13 Section Functions
# ---------------------------------------------------------------------------
def section_1_hardware_inventory(log: DiagLog) -> dict:
    """Section 1: Hardware inventory (no PII)."""
    log.section("1. HARDWARE INVENTORY (no PII)")
    return detect_hardware(log)


def section_2_system_load_pre(log: DiagLog) -> None:
    """Section 2: System load (pre-benchmark baseline)."""
    log.section("2. SYSTEM LOAD (pre-benchmark baseline)")
    log.write("Capturing system state BEFORE benchmarks to detect interference.")
    capture_load("pre_benchmark", log)

    plat = detect_platform()

    # Top CPU consumers
    log.subsection("Top CPU consumers (command name only)")
    try:
        ps_out = _run_cmd(["ps", "-eo", "pcpu,comm"])
        lines = ps_out.splitlines()
        # Sort by CPU% descending, skip header
        data_lines = []
        for line in lines[1:]:
            line = line.strip()
            if line:
                data_lines.append(line)
        data_lines.sort(
            key=lambda x: float(x.split()[0]) if x.split() else 0, reverse=True
        )
        for line in data_lines[:10]:
            log.write(f"[LOAD_TOP] {line}")
    except Exception as e:
        log.warning(f"Could not get top CPU consumers: {e}")

    # Disk I/O
    log.subsection("Disk I/O")
    if shutil.which("iostat"):
        try:
            io_out = _run_cmd(["iostat", "-c", "1"], timeout=10)
            for line in io_out.splitlines()[:5]:
                log.write(f"[LOAD_IO] {line}")
        except Exception as e:
            log.warning(f"Could not get disk I/O stats from iostat: {e}")

    # GPU-using processes
    log.subsection("GPU-using processes")
    if plat == "Darwin":
        try:
            ps_out = _run_cmd(["ps", "-eo", "pcpu,comm"])
            found = False
            for line in ps_out.splitlines():
                if re.search(
                    r"windowserver|gpu|metal|render|chrome.*helper", line, re.IGNORECASE
                ):
                    log.write(f"[LOAD_GPU_PROC] {line.strip()}")
                    found = True
            if not found:
                log.write("[LOAD_GPU_PROC] none detected")
        except Exception as e:
            log.warning(f"Could not detect GPU-using processes on macOS: {e}")
    elif plat == "Linux" and shutil.which("nvidia-smi"):
        try:
            gpu_procs = _run_cmd(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ]
            )
            if gpu_procs:
                for line in gpu_procs.splitlines():
                    log.write(f"[LOAD_GPU_PROC] {line.strip()}")
            else:
                log.write("[LOAD_GPU_PROC] none detected")
        except Exception as e:
            log.warning(f"Could not query nvidia-smi for GPU processes: {e}")

    # Model mmap check will be done in section 3


def section_3_model_info(
    log: DiagLog,
    cli_bin: str,
    model: str,
) -> None:
    """Section 3: Model info."""
    log.section("3. MODEL INFO")
    log.write("Extracting model metadata from GGUF file...")
    log.write("")

    # Run a quick CLI init to capture model metadata
    cmd = [
        cli_bin,
        "-m",
        model,
        "-ngl",
        "99",
        "-fa",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "-c",
        "512",
        "-n",
        "0",
        "-p",
        "x",
        "--jinja",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr
        keywords = [
            "general.name",
            "general.architecture",
            "general.file_type",
            "general.size_label",
            "general.quantized_by",
            "general.base_model",
            "model type",
            "model params",
            "file type",
            "file format",
            "file size",
            "n_ctx_train",
            "n_embd",
            "n_layer",
            "n_head",
            "n_head_kv",
            "n_expert",
            "n_expert_used",
            "n_embd_head",
            "vocab type",
            "n_vocab",
            "arch",
        ]
        for line in output.splitlines():
            if any(kw in line for kw in keywords):
                log.write(f"[MODEL] {line.strip()}")
    except Exception as e:
        log.warning(f"Could not extract model metadata: {e}")

    log.write("")
    model_basename = os.path.basename(model)
    log.write(f"[MODEL] filename={model_basename}")

    try:
        file_size = os.path.getsize(model)
        log.write(f"[MODEL] filesize_bytes={file_size}")
    except Exception as e:
        log.warning(f"Could not get model file size: {e}")
        log.write("[MODEL] filesize_bytes=0")

    # Model mmap + storage check
    log.subsection("Model storage and mmap check")
    log.write(
        f"[MMAP] model_file={os.path.basename(model)}"
    )  # basename only — no PII from path

    plat = detect_platform()

    # SSD detection — model on spinning disk kills mmap performance
    try:
        ssd_status = detect_storage_type(model, plat)
        log.write(f"[STORAGE] type={ssd_status}")
        if ssd_status == "hdd":
            log.warning(
                "Model is on spinning disk (HDD). mmap performance will be degraded."
            )
    except Exception as e:
        log.warning(f"Could not detect storage type: {e}")
        log.write("[STORAGE] type=unknown")

    if plat == "Darwin":
        try:
            model_size = os.path.getsize(model)
            model_mb = model_size / (1024 * 1024)
            vm_out = _run_cmd(["vm_stat"])
            free_pages = inactive_pages = 0
            for line in vm_out.splitlines():
                if "Pages free" in line:
                    free_pages = _safe_int(line.split(":")[1])
                elif "Pages inactive" in line:
                    inactive_pages = _safe_int(line.split(":")[1])
            free_mb = (free_pages + inactive_pages) * 4096 / (1024 * 1024)
            log.write(
                f"[MMAP] model_size_vs_free_ram={model_mb:.0f} MB model, "
                f"{free_mb:.0f} MB free"
            )
        except Exception as e:
            log.warning(f"Could not compute model size vs free RAM: {e}")


def section_4_gpu_capabilities(
    log: DiagLog,
    cli_bin: str,
    model: str,
) -> str:
    """Section 4: GPU device capabilities. Returns raw GPU init output."""
    log.section("4. GPU DEVICE CAPABILITIES")
    log.write("Extracting GPU info from llama.cpp init...")
    log.write("")

    gpu_init = ""
    cmd = [
        cli_bin,
        "-m",
        model,
        "-ngl",
        "99",
        "-fa",
        "on",
        "--cache-type-k",
        "turbo3",
        "--cache-type-v",
        "turbo3",
        "-c",
        "512",
        "-n",
        "1",
        "-p",
        "test",
        "--jinja",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        gpu_init = result.stdout + result.stderr
        keywords = [
            "build:",
            "GPU name",
            "GPU family",
            "simdgroup",
            "unified",
            "bfloat",
            "has tensor",
            "residency",
            "shared buffers",
            "recommendedMax",
            "system.info",
            "system_info",
            "n_threads",
            "turbo",
            "TurboQuant",
            "KV buffer",
            "rotation",
            "metal_library",
            "metal_init",
            "embed",
            "loaded in",
            "CUDA",
            "cuda",
            "VRAM",
            "cublas",
        ]
        for line in gpu_init.splitlines():
            if any(kw in line for kw in keywords):
                log.write(f"[GPU] {line.strip()}")
    except Exception as e:
        log.warning(f"Could not get GPU init: {e}")

    plat = detect_platform()
    if plat == "Darwin":
        log.subsection("Tensor API Check (M1 vs M5 decode performance)")
        has_tensor_line = "NOT FOUND"
        for line in gpu_init.splitlines():
            if "has tensor" in line:
                has_tensor_line = line.strip()
                break
        log.write(f"[METAL_TENSOR] {has_tensor_line}")
        if "false" in has_tensor_line.lower():
            log.write(
                "[METAL_TENSOR] WARNING: Tensor API disabled. This is M1/M2/M3/M4 hardware."
            )
            log.write(
                "[METAL_TENSOR] WARNING: Turbo3 decode may be significantly slower due to constant cache limitations."
            )
    elif plat == "Linux":
        log.subsection("CUDA Device Check")
        if shutil.which("nvidia-smi"):
            try:
                cuda_out = _run_cmd(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,compute_cap,memory.total,clocks.max.sm",
                        "--format=csv,noheader",
                    ]
                )
                for line in cuda_out.splitlines():
                    log.write(f"[CUDA] {line.strip()}")
            except Exception as e:
                log.write(f"[CUDA] nvidia-smi query failed: {e}")
        else:
            log.write("[CUDA] nvidia-smi not found \u2014 CUDA may not be available")

    return gpu_init


def section_5_build_validation(
    log: DiagLog,
    bench_bin: str,
    cli_bin: str,
    model: str,
    llama_dir: str,
) -> None:
    """Section 5: Build validation."""
    log.section("5. BUILD VALIDATION")

    # Verify turbo3 works
    log.subsection("turbo3 in llama-bench")
    cmd = [
        bench_bin,
        "-m",
        model,
        "-ngl",
        "99",
        "-fa",
        "1",
        "-ctk",
        "turbo3",
        "-ctv",
        "turbo3",
        "-p",
        "64",
        "-n",
        "0",
        "-r",
        "1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        lines = output.strip().splitlines()
        for line in lines[-5:]:
            log.write(line)
        if result.returncode != 0:
            log.write("FAILED: turbo3 not available in llama-bench")
    except Exception as e:
        log.write(f"FAILED: turbo3 validation error: {e}")

    # Metal library validation
    log.subsection("Metal library validation")
    cmd = [
        cli_bin,
        "-m",
        model,
        "-ngl",
        "99",
        "-fa",
        "on",
        "--cache-type-k",
        "turbo3",
        "--cache-type-v",
        "turbo3",
        "-c",
        "512",
        "-n",
        "1",
        "-p",
        "test",
        "--jinja",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        found = False
        for line in output.splitlines():
            if any(kw in line for kw in ("metal_library", "embed", "loaded in")):
                log.write(line.strip())
                found = True
                if (
                    found
                    and len(
                        [
                            output_line
                            for output_line in output.splitlines()
                            if any(
                                keyword in output_line
                                for keyword in ("metal_library", "embed", "loaded in")
                            )
                        ]
                    )
                    >= 5
                ):
                    break
        if not found:
            log.write("WARNING: Could not verify Metal library load")
    except Exception as e:
        log.write(f"WARNING: Metal library validation failed: {e}")

    # Git commit
    log.subsection("Build commit")
    try:
        commit = _run_cmd(["git", "-C", llama_dir, "log", "--oneline", "-1"])
        if commit:
            log.write(f"[BUILD] {commit}")
        else:
            log.write("[BUILD] not a git repo")
    except Exception as e:
        log.write(f"[BUILD] not a git repo (error: {e})")


def section_6_prefill(
    log: DiagLog,
    bench_bin: str,
    model: str,
    display: LiveDisplay,
) -> None:
    """Section 6: Prefill speed."""
    log.section("6. PREFILL SPEED (tok/s)")
    log.write("Prefill at 2K/4K/8K/16K/32K. Expect flat 0.98-1.00x q8_0.")
    log.write("If ratio drops >5% across depths = context scaling regression.")
    log.write("")

    capture_load("pre_prefill", log)

    depths_str = ",".join(str(d) for d in PREFILL_DEPTHS)

    run_bench(
        "q8_0 prefill (all depths)",
        "q8_0",
        "q8_0",
        f"-p {depths_str} -n 0",
        log,
        bench_bin,
        model,
    )
    run_bench(
        "turbo3 prefill (all depths)",
        "turbo3",
        "turbo3",
        f"-p {depths_str} -n 0",
        log,
        bench_bin,
        model,
    )
    run_bench(
        "turbo3 mode2 prefill (all depths)",
        "turbo3",
        "turbo3",
        f"-p {depths_str} -n 0",
        log,
        bench_bin,
        model,
        env_prefix=MODE2_ENV,
    )

    capture_load("post_prefill", log)


def section_7_decode(
    log: DiagLog,
    bench_bin: str,
    model: str,
    display: LiveDisplay,
    anomaly: AnomalyDetector,
) -> None:
    """Section 7: Decode speed — THE CRITICAL TEST."""
    log.section("7. DECODE SPEED (tok/s) \u2014 THE CRITICAL TEST")
    log.write("Decode at increasing context depths. This is where M1 fails.")
    log.write("")
    log.write("Known baselines:")
    log.write("  M5 Max: turbo3/q8_0 = 0.92x (short) \u2192 0.72x (48K)")
    log.write(
        "  M1 Max: turbo3/q8_0 = ??? (short) \u2192 0.09x (42K) \u2190 CATASTROPHIC"
    )
    log.write("")
    log.write("Healthy: ratio stays above 0.70x through 32K")
    log.write("Problem: ratio drops below 0.50x at any depth")
    log.write("")

    capture_load("pre_decode", log)

    q8_decode: dict[int, float] = {}
    t3_decode: dict[int, float] = {}

    for depth in DECODE_DEPTHS:
        if depth == 0:
            depth_label = "short"
            depth_flag = "-p 0 -n 128"
        else:
            depth_label = f"@{depth // 1024}K"
            depth_flag = f"-p 0 -n 128 -d {depth}"

        # q8_0
        output, _ = run_bench(
            f"q8_0 decode ({depth_label})",
            "q8_0",
            "q8_0",
            depth_flag,
            log,
            bench_bin,
            model,
        )
        results = parse_bench_tps(output)
        for r in results:
            if r["mode"] == "decode":
                q8_decode[depth] = r["tps"]
                display.update_decode("q8_0", depth, r["tps"])
                if depth == 0:
                    anomaly.set_q8_short_decode(r["tps"])
                    hw_class = (
                        "apple_silicon" if platform.system() == "Darwin" else "unknown"
                    )
                    anomaly.check_q8_baseline(r["tps"], hw_class)

        # turbo3
        output, _ = run_bench(
            f"turbo3 decode ({depth_label})",
            "turbo3",
            "turbo3",
            depth_flag,
            log,
            bench_bin,
            model,
        )
        results = parse_bench_tps(output)
        for r in results:
            if r["mode"] == "decode":
                t3_decode[depth] = r["tps"]
                display.update_decode("turbo3", depth, r["tps"])
                # Check ratio
                if depth in q8_decode and q8_decode[depth] > 0:
                    ratio = r["tps"] / q8_decode[depth]
                    anomaly.check_decode_ratio(depth, ratio)

        # turbo3 mode2
        run_bench(
            f"turbo3 mode2 decode ({depth_label})",
            "turbo3",
            "turbo3",
            depth_flag,
            log,
            bench_bin,
            model,
            env_prefix=MODE2_ENV,
        )

    anomaly.check_thermal()
    anomaly.check_swap_growth()

    capture_load("post_decode", log)

    display.show_section_summary("Decode")


def section_8_stress_test(
    log: DiagLog,
    bench_bin: str,
    model: str,
    display: LiveDisplay,
    anomaly: AnomalyDetector,
) -> None:
    """Section 8: Constant cache stress test."""
    log.section("8. CONSTANT CACHE STRESS TEST (fine-grained decode gradient)")
    log.write("Fine-grained decode at many depths to find the EXACT inflection point.")
    log.write("This tells us where constant cache pressure becomes dominant.")
    log.write("")
    log.write("NOTE: 1K context is unreliable (Metal async dispatch timing artifact).")
    log.write(
        "      Results at 1K may show impossibly high numbers \u2014 ignore them."
    )
    log.write("")

    t3_stress: dict[int, float] = {}
    q8_stress: dict[int, float] = {}

    # turbo3 stress
    for depth in STRESS_DEPTHS:
        output, _ = run_bench(
            f"turbo3 decode @{depth} (stress)",
            "turbo3",
            "turbo3",
            f"-p 0 -n 64 -d {depth}",
            log,
            bench_bin,
            model,
        )
        results = parse_bench_tps(output)
        for r in results:
            if r["mode"] == "decode":
                t3_stress[depth] = r["tps"]

    log.write("")
    log.write("q8_0 baseline at same depths:")

    # q8_0 stress baseline
    for depth in STRESS_DEPTHS:
        output, _ = run_bench(
            f"q8_0 decode @{depth} (stress)",
            "q8_0",
            "q8_0",
            f"-p 0 -n 64 -d {depth}",
            log,
            bench_bin,
            model,
        )
        results = parse_bench_tps(output)
        for r in results:
            if r["mode"] == "decode":
                q8_stress[depth] = r["tps"]

    # Compute ratios and check anomalies
    stress_ratios: dict[int, float] = {}
    for depth in sorted(set(t3_stress.keys()) & set(q8_stress.keys())):
        if q8_stress[depth] > 0:
            ratio = t3_stress[depth] / q8_stress[depth]
            stress_ratios[depth] = ratio
            anomaly.check_decode_ratio(depth, ratio)

    display.show_stress_summary("turbo3/q8_0 stress", stress_ratios)

    anomaly.check_thermal()
    anomaly.check_swap_growth()

    capture_load("post_stress", log)


def section_9_combined(
    log: DiagLog,
    bench_bin: str,
    model: str,
) -> None:
    """Section 9: Combined prefill+decode (realistic workload)."""
    log.section("9. COMBINED PREFILL+DECODE (realistic workload)")
    log.write("Simulates real usage: prefill a prompt, then generate response tokens.")
    log.write("")

    for pp, tg in COMBINED_CONFIGS:
        run_bench(
            f"q8_0 pp{pp // 1024}K+tg{tg}",
            "q8_0",
            "q8_0",
            f"-pg {pp},{tg}",
            log,
            bench_bin,
            model,
        )
        run_bench(
            f"turbo3 pp{pp // 1024}K+tg{tg}",
            "turbo3",
            "turbo3",
            f"-pg {pp},{tg}",
            log,
            bench_bin,
            model,
        )
        # Mode2 for all except the last one (matches bash script)
        if (pp, tg) != COMBINED_CONFIGS[-1]:
            run_bench(
                f"turbo3 mode2 pp{pp // 1024}K+tg{tg}",
                "turbo3",
                "turbo3",
                f"-pg {pp},{tg}",
                log,
                bench_bin,
                model,
                env_prefix=MODE2_ENV,
            )


def section_10_perplexity(
    log: DiagLog,
    perpl_bin: str,
    model: str,
    wiki_path: str,
    anomaly: AnomalyDetector,
    skip_ppl: bool = False,
) -> None:
    """Section 10: Perplexity (quality validation)."""
    log.section("10. PERPLEXITY (quality validation)")
    log.write(
        "PPL must be within 2% of q8_0. If not, something is wrong with the build."
    )
    log.write(
        "CRITICAL: A PPL >2% delta means quality is broken \u2014 speed numbers are meaningless."
    )
    log.write("")

    if skip_ppl:
        log.write("SKIPPED: --skip-ppl flag was set.")
        return

    if not os.path.isfile(wiki_path):
        log.write(f"SKIPPED: wikitext-2-raw not found at {wiki_path}")
        log.write("")
        log.write("To enable PPL testing, download wikitext-2:")
        log.write(
            "  wget https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1.zip"
        )
        llama_dir = str(Path(perpl_bin).parent.parent.parent)
        log.write(f"  unzip wikitext-2-raw-v1.zip -d {llama_dir}/wikitext-2-raw")
        log.write("")
        log.write(
            "PPL is the single most important quality check. Without it, speed numbers"
        )
        log.write("could be from a broken build that outputs garbage at high speed.")
        return

    # q8_0 baseline
    output, _ = run_perpl(
        "q8_0 PPL (8 chunks)",
        "q8_0",
        "q8_0",
        8,
        log,
        perpl_bin,
        model,
        wiki_path,
    )
    ppl, stddev = parse_ppl_final(output)
    if ppl > 0:
        anomaly.set_q8_ppl(ppl)

    # turbo3
    output, _ = run_perpl(
        "turbo3 PPL (8 chunks)",
        "turbo3",
        "turbo3",
        8,
        log,
        perpl_bin,
        model,
        wiki_path,
    )
    ppl_t3, _ = parse_ppl_final(output)
    if ppl_t3 > 0:
        anomaly.check_ppl("turbo3", ppl_t3)

    # turbo3 mode2
    output, _ = run_perpl(
        "turbo3 mode2 PPL (8 chunks)",
        "turbo3",
        "turbo3",
        8,
        log,
        perpl_bin,
        model,
        wiki_path,
        env_prefix=MODE2_ENV,
    )
    ppl_m2, _ = parse_ppl_final(output)
    if ppl_m2 > 0:
        anomaly.check_ppl("turbo3 mode2", ppl_m2, env=MODE2_ENV)


def section_11_memory(
    log: DiagLog,
    cli_bin: str,
    model: str,
) -> None:
    """Section 11: Memory breakdown."""
    log.section("11. MEMORY BREAKDOWN")

    configs = [
        ("KV cache at 32K context", "turbo3", 32768, "MEM_32K"),
        ("KV cache at 131K context", "turbo3", 131072, "MEM_131K"),
        ("q8_0 KV at 32K (for size comparison)", "q8_0", 32768, "MEM_Q8_32K"),
    ]

    mem_keywords = [
        "KV buffer",
        "KV",
        "size",
        "memory_breakdown",
        "compute buffer",
        "RS buffer",
        "model buffer",
        "recommendedMax",
        "load_tensors",
        "offload",
    ]

    for title, ctk, ctx, tag in configs:
        log.subsection(title)
        cmd = [
            cli_bin,
            "-m",
            model,
            "-ngl",
            "99",
            "-fa",
            "on",
            "--cache-type-k",
            ctk,
            "--cache-type-v",
            ctk,
            "-c",
            str(ctx),
            "-n",
            "1",
            "-p",
            "test",
            "--jinja",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            output = result.stdout + result.stderr
            for line in output.splitlines():
                if any(kw in line for kw in mem_keywords):
                    log.write(f"[{tag}] {line.strip()}")
        except Exception as e:
            log.warning(f"Memory check failed for {title}: {e}")


def section_12_post_load(log: DiagLog) -> None:
    """Section 12: System load (post-benchmark)."""
    log.section("12. SYSTEM LOAD (post-benchmark)")
    log.write(
        "Final load snapshot \u2014 compare with pre-benchmark to detect thermal throttling."
    )
    capture_load("post_all_benchmarks", log)

    # Thermal throttling check
    log.subsection("Thermal throttling check")
    plat = detect_platform()
    if plat == "Darwin":
        try:
            therm_out = _run_cmd(["pmset", "-g", "therm"])
            m = re.search(r"CPU_Speed_Limit\s+(\d+)", therm_out)
            limit = m.group(1) if m else "100"
            log.write(f"[THERMAL] final_cpu_speed_limit={limit}")
            if int(limit) < 100:
                log.write(
                    f"[THERMAL] WARNING: CPU speed limited to {limit}%. Results may be throttled."
                )
                log.write(
                    "[THERMAL] WARNING: Run benchmarks again after cooling down for accurate results."
                )
        except Exception as e:
            log.warning(f"Could not check post-benchmark thermal state on macOS: {e}")
    elif plat == "Linux":
        try:
            temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
            if temp_path.exists():
                temp = temp_path.read_text(encoding="utf-8", errors="replace").strip()
                log.write(f"[THERMAL] final_cpu_temp={temp}m\u00b0C")
                if _safe_int(temp) > 90000:
                    log.write(
                        "[THERMAL] WARNING: CPU temperature above 90\u00b0C. Results may be thermally throttled."
                    )
        except Exception as e:
            log.warning(f"Could not check post-benchmark thermal state on Linux: {e}")


def section_13_summary(
    log: DiagLog,
    anomaly_detector: AnomalyDetector,
    platform_support: Optional[dict] = None,
) -> None:
    """Section 13: Diagnostic summary."""
    log.section("13. DIAGNOSTIC SUMMARY")

    if platform_support is not None:
        log_platform_support(log, platform_support)
        log.write("")

    log.write("TURBO_DIAG_COMPLETE=true")
    log.write(f"TURBO_DIAG_END_TIMESTAMP={_utc_now()}")
    log.write("")

    # Print notable findings (good results worth highlighting)
    notables = anomaly_detector.notables
    if notables:
        log.write(f"NOTABLE FINDINGS: {len(notables)}")
        for n in notables:
            log.write(f"  [NOTABLE] {n}")
        log.write("")

    # Print items needing investigation (outlandish — suspiciously good or bad)
    investigations = anomaly_detector.investigations
    if investigations:
        log.write(f"INVESTIGATE FURTHER: {len(investigations)}")
        for inv in investigations:
            log.write(f"  [INVESTIGATE] {inv}")
        log.write("")

    # Print anomalies (confirmed problems)
    anomalies = anomaly_detector.anomalies
    if anomalies:
        log.write(f"ANOMALIES DETECTED: {len(anomalies)}")
        for a in anomalies:
            log.write(f"  [ANOMALY] {a}")
        log.write("")

    if not notables and not investigations and not anomalies:
        log.write("No anomalies or notable findings. Results look clean.")
        log.write("")

    log.write("=" * 44)
    log.write("  HOW TO READ THESE RESULTS")
    log.write("=" * 44)
    log.write("")
    log.write(
        "1. MODEL: Section 3 confirms which model and quantization you're running."
    )
    log.write("")
    log.write("2. QUALITY: Section 10 PPL should be within 2% of q8_0.")
    log.write("   If PPL is broken, ALL speed numbers are meaningless.")
    log.write("")
    log.write("3. PREFILL: Section 6 turbo3/q8_0 ratio should be flat 0.95-1.00x")
    log.write("   across all depths. If it degrades = context scaling regression.")
    log.write("")
    log.write("4. DECODE: Section 7 is the critical metric.")
    log.write(
        "   Healthy (M5):  0.92x (short) -> 0.72x (48K) \u2014 gradual degradation"
    )
    log.write(
        "   Problem (M1):  0.90x (short) -> 0.09x (42K) \u2014 constant cache thrashing"
    )
    log.write("")
    log.write("5. INFLECTION POINT: Section 8 stress test shows the exact depth")
    log.write("   where turbo3 decode starts falling off a cliff.")
    log.write("")
    log.write("6. THERMAL: Compare Sections 2 and 12. If CPU_Speed_Limit dropped")
    log.write("   during the test, results may be artificially low.")
    log.write("")
    log.write("7. MEMORY: Section 11 shows KV cache sizing. If the system is near")
    log.write("   its recommendedMaxWorkingSetSize, swap pressure kills decode.")
    log.write("")
    log.write("NEXT STEPS:")
    log.write(
        "  Open a GitHub issue: github.com/dipeshbabu/efficient-llm-systems/issues"
    )
    log.write("     Title: 'Diagnostic: [your hardware]', attach the zip file.")
    log.write("  If decode ratio < 0.50x at any depth, use TURBO_LAYER_ADAPTIVE=2")
    log.write("  or q8_0 cache until the M1 constant cache fix is available.")
    log.write("")
    log.write("END OF DIAGNOSTIC")


# ---------------------------------------------------------------------------
# JSON Profile Builder
# ---------------------------------------------------------------------------
def build_json_profile(
    hw: dict,
    model: str,
    gpu_init: str,
    date_str: str,
    platform_support: Optional[dict] = None,
) -> dict:
    """Build the machine-readable hardware profile JSON."""
    support = platform_support or get_platform_support()
    plat = support["platform"]

    # Extract GPU family from init output
    gpu_family = "N/A"
    has_tensor = False
    for line in gpu_init.splitlines():
        if "GPU family:" in line:
            gpu_family = line.split("GPU family:")[-1].strip()
        if "has tensor" in line and "true" in line.lower():
            has_tensor = True

    profile = {
        "diag_version": DIAG_VERSION,
        "timestamp": _utc_now(),
        "platform": plat,
        "os_version": platform.release(),
        "arch": platform.machine(),
        "model_file": os.path.basename(model),
        "model_size_bytes": 0,
        "platform_support": support,
        "hardware": {
            "cpu_brand": hw.get("cpu_brand", "unknown"),
            "ram_gb": hw.get("ram_total_gb", 0),
            "gpu_family": gpu_family,
            "has_tensor": has_tensor,
            "apple_silicon": hw.get("apple_silicon", False),
        },
    }

    try:
        profile["model_size_bytes"] = os.path.getsize(model)
    except Exception:
        # No log available here; model_size_bytes stays 0 (set in profile init)
        pass

    return profile


# ---------------------------------------------------------------------------
# ZIP Packaging
# ---------------------------------------------------------------------------
def package_results(
    log: DiagLog,
    monitor: BackgroundMonitor,
    profile_json: dict,
    date_str: str,
    output_dir: str,
) -> str:
    """Package all artifacts into a zip file. Returns zip path."""
    log.write("")
    log.write("Packaging results...")

    zip_name = f"turbo-diag-{date_str}.zip"
    zip_path = os.path.join(output_dir, zip_name)

    profile_name = f"turbo-hwprofile-{date_str}.json"
    profile_path = os.path.join(output_dir, profile_name)

    # Write JSON profile
    try:
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_json, f, indent=2)
    except Exception as e:
        log.warning(f"Could not write JSON profile: {e}")

    # Close log so it's fully flushed before zipping
    log_path = log.path

    # Scrub PII: replace home directory with ~ in all output files
    home_dir = str(Path.home())
    for scrub_path in [log_path, profile_path, monitor.csv_path]:
        if os.path.isfile(scrub_path):
            try:
                with open(scrub_path, encoding="utf-8") as f:
                    content = f.read()
                content = content.replace(home_dir, "~")
                with open(scrub_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception:
                pass  # best-effort scrub

    # Create zip
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Log file
            if os.path.isfile(log_path):
                zf.write(log_path, os.path.basename(log_path))

            # Monitor CSV
            if os.path.isfile(monitor.csv_path):
                zf.write(monitor.csv_path, os.path.basename(monitor.csv_path))

            # JSON profile
            if os.path.isfile(profile_path):
                zf.write(profile_path, os.path.basename(profile_path))
    except Exception as e:
        log.warning(f"Could not create zip: {e}")
        log.write(f"Raw artifacts preserved at: {output_dir}")
        return ""  # Don't cleanup if zip failed

    # Cleanup temp files only if zip was created successfully
    try:
        if os.path.isfile(profile_path):
            os.remove(profile_path)
        if os.path.isfile(monitor.csv_path):
            os.remove(monitor.csv_path)
    except Exception as e:
        log.warning(f"Could not clean up temp files: {e}")

    log.write("")
    log.write("=" * 44)
    log.write("  DIAGNOSTIC PACKAGE READY")
    log.write("=" * 44)
    log.write("")
    log.write(f"  Zip file: {zip_path}")
    log.write("  Contents:")
    log.write(f"    {os.path.basename(log_path)}")
    log.write(f"    {os.path.basename(monitor.csv_path)}")
    log.write(f"    {profile_name}")
    log.write("")
    log.write("  Share: github.com/dipeshbabu/efficient-llm-systems/issues")
    log.write("")

    return zip_path


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="TurboQuant Hardware Diagnostic v5 \u2014 comprehensive benchmark + device profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python3 turbo_hardware_diag.py /path/to/llama.cpp /path/to/model.gguf
              python3 turbo_hardware_diag.py --model model.gguf --llama-dir /path/to/llama.cpp
              python3 turbo_hardware_diag.py --skip-ppl --verbose

            Output: turbo-diag-<date>.zip
              Share via: github.com/dipeshbabu/efficient-llm-systems/issues
              Contains .txt log, .json profile, .csv background monitor data.

            NO PII is collected. Only hardware specs, load stats, and benchmark numbers.
            Estimated runtime: 20-40 minutes depending on hardware.
        """),
    )
    parser.add_argument(
        "llama_dir",
        nargs="?",
        default=None,
        help="Path to llama.cpp directory (default: current directory)",
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default=None,
        help="Path to .gguf model file (auto-detected if not specified)",
    )
    parser.add_argument(
        "--model",
        dest="model_flag",
        default=None,
        help="Path to .gguf model file (alternative to positional arg)",
    )
    parser.add_argument(
        "--llama-dir",
        dest="llama_dir_flag",
        default=None,
        help="Path to llama.cpp directory (alternative to positional arg)",
    )
    parser.add_argument(
        "--skip-ppl",
        action="store_true",
        help="Skip perplexity tests (saves ~10 minutes)",
    )
    parser.add_argument(
        "--skip-stress",
        action="store_true",
        help="Skip the fine-grained stress test (section 8)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (debug-level logging)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Output directory for diagnostic files (default: current directory)",
    )
    parser.add_argument(
        "--platform-support",
        action="store_true",
        help="Print the current platform support contract as JSON and exit",
    )

    args = parser.parse_args()

    platform_support = get_platform_support()
    if args.platform_support:
        print(json.dumps(platform_support, indent=2))
        return 0
    if platform_support["status"] == "unsupported":
        print(
            "ERROR: TurboQuant diagnostic is not supported on "
            f"{platform_support['environment']}."
        )
        print(f"Reason: {platform_support['reason']}")
        print(f"Next step: {platform_support['action']}")
        print("Inspect the contract with: turbo_hardware_diag.py --platform-support")
        return 2
    platform_support = apply_runtime_skips(
        platform_support,
        skip_stress=args.skip_stress,
        skip_ppl=args.skip_ppl,
    )

    # Resolve paths
    llama_dir = args.llama_dir_flag or args.llama_dir or os.getcwd()
    model = args.model_flag or args.model_path
    llama_dir = os.path.abspath(llama_dir)

    # Auto-find model if not specified
    if not model:
        model = _find_model(llama_dir)
        if not model:
            print("ERROR: No .gguf model found. Pass model path as argument.")
            print(
                "Usage: python3 turbo_hardware_diag.py /path/to/llama.cpp /path/to/model.gguf"
            )
            return 1
    model = os.path.abspath(model)

    # Validate tools
    bench_bin = os.path.join(llama_dir, "build", "bin", "llama-bench")
    perpl_bin = os.path.join(llama_dir, "build", "bin", "llama-perplexity")
    cli_bin = os.path.join(llama_dir, "build", "bin", "llama-cli")
    wiki_path = os.path.join(llama_dir, "wikitext-2-raw", "wiki.test.raw")

    # Required tools — perplexity only needed if PPL not skipped
    required_tools = [bench_bin, cli_bin]
    if not args.skip_ppl:
        required_tools.append(perpl_bin)

    for tool in required_tools:
        if not os.path.isfile(tool):
            print(f"ERROR: {tool} not found. Build llama.cpp first:")
            print(f"  cd {llama_dir}")
            plat = platform.system()
            if plat == "Darwin":
                print(
                    "  cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release"
                )
            else:
                print("  cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release")
            print("  cmake --build build -j")
            return 1

    if not os.path.isfile(model):
        print(f"ERROR: Model not found: {model}")
        return 1

    # Create output directory
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Setup
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(output_dir, f"turbo-diag-{date_str}.txt")
    monitor_path = os.path.join(output_dir, f"turbo-monitor-{date_str}.csv")

    log = DiagLog(log_path, verbose=args.verbose)
    monitor = BackgroundMonitor(monitor_path)
    display = LiveDisplay(use_rich=HAS_RICH)
    anomaly_detector: Optional[AnomalyDetector] = None

    # Graceful shutdown on Ctrl+C
    _interrupted = False

    def _signal_handler(signum: int, frame: object) -> None:  # pragma: no cover
        nonlocal _interrupted
        _interrupted = True
        try:
            log.write("")
            log.write("[WARNING] Interrupted by user (Ctrl+C)")
        except Exception:
            pass  # Log file may already be closed during shutdown
        monitor.stop()
        display.stop()
        sys.exit(130)

    signal.signal(signal.SIGINT, _signal_handler)

    # --- Banner ---
    log.write(f"TurboQuant Hardware Diagnostic v{DIAG_VERSION}")
    log.write(f"Output: {log_path} (human-readable log, zipped at end)")
    log.write(f"Model: {model}")
    log.write("")
    log.write("NO PII is collected. Only hardware specs, load stats, and benchmarks.")
    log.write("Estimated runtime: 20-40 minutes.")
    log.write("")
    log.write("Press Ctrl+C to abort at any time.")
    log.write("")

    # --- Header tags (must match parse_diag_output expectations) ---
    log.write(f"TURBO_DIAG_VERSION={DIAG_VERSION}")
    log.write(f"TURBO_DIAG_TIMESTAMP={_utc_now()}")
    log.write(f"TURBO_DIAG_MODEL={os.path.basename(model)}")
    log.write(f"TURBO_DIAG_PLATFORM_ENVIRONMENT={platform_support['environment']}")
    log.write(f"TURBO_DIAG_PLATFORM_STATUS={platform_support['status']}")
    log_platform_support(log, platform_support)
    try:
        model_size = os.path.getsize(model)
        # Human-readable size
        if model_size >= 1024**3:
            size_str = f"{model_size / (1024**3):.1f}G"
        elif model_size >= 1024**2:
            size_str = f"{model_size / (1024**2):.1f}M"
        else:
            size_str = f"{model_size}"
        log.write(f"TURBO_DIAG_MODEL_SIZE={size_str}")
    except Exception as e:
        log.warning(f"Could not determine model file size: {e}")
        log.write("TURBO_DIAG_MODEL_SIZE=unknown")

    # Start background monitor
    monitor.start()

    # Record initial swap for anomaly detection
    try:
        initial_swap = float(monitor._poll().get("swap_used_mb", "0"))
    except Exception as e:
        log.warning(f"Could not read initial swap usage: {e}")
        initial_swap = 0.0

    # Start live display
    display.start()

    hw: dict = {}
    gpu_init: str = ""
    _unsupported = False

    try:
        # Section 1: Hardware inventory
        hw = section_1_hardware_inventory(log)

        # Create anomaly detector now that we have monitor running
        anomaly_detector = AnomalyDetector(log, monitor)
        anomaly_detector.set_initial_swap(initial_swap)

        # Section 2: System load (pre-benchmark)
        section_2_system_load_pre(log)

        # Section 3: Model info
        section_3_model_info(log, cli_bin, model)

        # Section 4: GPU capabilities
        gpu_init = section_4_gpu_capabilities(log, cli_bin, model)
        platform_support = assess_runtime_backend(platform_support, gpu_init, hw)
        if platform_support["status"] == "unsupported":
            raise UnsupportedDiagnosticPlatform(platform_support["reason"])

        # Section 5: Build validation
        section_5_build_validation(log, bench_bin, cli_bin, model, llama_dir)

        # Section 6: Prefill speed
        section_6_prefill(log, bench_bin, model, display)

        # Section 7: Decode speed (critical)
        section_7_decode(log, bench_bin, model, display, anomaly_detector)

        # Section 8: Stress test
        if args.skip_stress:
            log.section("8. CONSTANT CACHE STRESS TEST (fine-grained decode gradient)")
            log.write("SKIPPED: --skip-stress flag was set.")
        else:
            section_8_stress_test(log, bench_bin, model, display, anomaly_detector)

        # Section 9: Combined
        section_9_combined(log, bench_bin, model)

        # Section 10: Perplexity
        section_10_perplexity(
            log,
            perpl_bin,
            model,
            wiki_path,
            anomaly_detector,
            skip_ppl=args.skip_ppl,
        )

        # Section 11: Memory breakdown
        section_11_memory(log, cli_bin, model)

        # Section 12: Post-benchmark load
        section_12_post_load(log)

        # Section 13: Summary
        section_13_summary(log, anomaly_detector, platform_support)

    except UnsupportedDiagnosticPlatform as e:
        _had_error = True
        _unsupported = True
        log.section("DIAGNOSTIC STOPPED: UNSUPPORTED BACKEND")
        log.write(f"[ERROR] {e}")
        log_platform_support(log, platform_support)
        log.write("TURBO_DIAG_COMPLETE=false")
    except Exception as e:
        _had_error = True
        log.write(f"\n[ERROR] Unhandled exception: {e}")
        import traceback

        log.write(traceback.format_exc())
    else:
        _had_error = False
    finally:
        # Stop display + monitor (guard against interrupted state)
        display.stop()
        monitor.stop()
        if not _interrupted:
            try:
                log.write(
                    f"[MONITOR] Captured {monitor.sample_count} samples in {monitor.csv_path}"
                )
            except Exception as e:
                log.warning(f"Could not write monitor summary: {e}")

    # Build JSON profile
    profile_json = build_json_profile(
        hw,
        model,
        gpu_init,
        date_str,
        platform_support,
    )

    # Package results
    package_results(log, monitor, profile_json, date_str, output_dir)

    log.close()

    if _unsupported:
        return 2
    return 1 if _had_error else 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
