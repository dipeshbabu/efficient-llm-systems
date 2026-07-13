# Tools

Hardware profiling, benchmarking, quality validation, and conversion utilities
for Efficient LLM Systems. Tools are grouped by purpose:

- `diagnostics/` — portable hardware and runtime diagnostics
- `validation/` — quality gates, NIAH, MLX validation, and skip-rate analysis
- `benchmarks/` — quick and real-world benchmark drivers
- `conversion/` — model-conversion helpers
- `maintenance/` — repository integrity checks

---

## Quality & Speed At A Glance

**Quality (Qwen3.5-35B-A3B MoE Q8_0, M5 Max 128GB):**

| Config | PPL vs q8_0 | NIAH Single (4K-32K) | NIAH Multi-Key (4K-32K) |
|--------|-------------|---------------------|------------------------|
| Uniform turbo3 | +1.1% | 18/20 (q8_0: 17/20) | 4/4 perfect |
| Mode 2 (layer-adaptive) | **+0.14%** | — | — |

turbo3 matches q8_0 on needle retrieval. The few single-needle misses at 32K overlap with positions where q8_0 also fails.

**Speed (turbo3/q8_0 ratio):**

| Hardware | Prefill | Decode (short) | Decode (32K) | Notes |
|----------|---------|---------------|-------------|-------|
| M5 Max 128GB | 0.95-1.00x | 0.92x | 0.75x | Healthy gradient |
| M2 Pro 32GB | **1.32-1.73x** | ~0.73x | pending | Prefill wins big on lower bandwidth |
| M1 Max 64GB | 0.93-0.98x | 0.83x | 0.39x | Constant memory LUT regression |
| Dual 4090 (CUDA) | 0.98x | 0.37x (MoE, 42K) | — | buun investigating, fix in progress |

**Pre-M5 decode regression** is a known hardware limitation, not a quality issue. Fully documented in [decode-speed-hardware-analysis.md](../research/investigations/decode-speed-hardware-analysis.md). 12 kernel approaches tested, root cause is constant memory LUT divergence on Apple7/Apple8 GPU families.

**Compression:**

| Config | Effective ratio | Bits per value |
|--------|----------------|---------------|
| Uniform turbo3 | 4.6x | 3.5 |
| Mode 2 (q8_0 last 8 layers) | 3.5x | ~4.6 |
| + Temporal decay | +30-34% savings at 32K+ | — |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/dipeshbabu/efficient-llm-systems.git
cd efficient-llm-systems

# 2. Run the diagnostic
bash tools/diagnostics/turbo-diag /path/to/llama.cpp /path/to/model.gguf
```

That's it. The launcher automatically:
1. Finds or creates an isolated Python venv (`.turbo-diag-venv/`)
2. Installs `rich` for a color-coded terminal UI (falls back to ASCII if it fails)
3. Runs the full diagnostic suite

**Estimated runtime:** 20–40 minutes depending on hardware. Use `--skip-ppl` and `--skip-stress` to cut that roughly in half.

**Requirements:** Python 3.10+, a built llama.cpp with turbo3 support, and a `.gguf` model file.

### Which fork do I build?

| Hardware | Fork | Branch |
|----------|------|--------|
| **Apple Silicon (M1-M5)** | [dipeshbabu/llama-cpp-turboquant](https://github.com/dipeshbabu/llama-cpp-turboquant) | `feature/turboquant-kv-cache` |
| **NVIDIA (CUDA)** | [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) | `feature/turboquant-kv-cache` |
| **AMD (ROCm)** | spiritbuun's fork (has RDNA `v_dot2` path) | `feature/turboquant-kv-cache` |

> **AMD users:** ROCm support is untested and experimental. Flash attention is reportedly broken on RDNA 2 (6800 XT) with ROCm > 7.0.1, and turbo3 requires flash attention. RDNA 3/4 may work but nobody has confirmed yet. Expect rough edges and please report what you find.

```bash
# Apple Silicon
git clone https://github.com/dipeshbabu/llama-cpp-turboquant.git
cd llama-cpp-turboquant && git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# NVIDIA CUDA
git clone https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git
cd llama-cpp-turboquant-cuda && git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# AMD ROCm
git clone https://github.com/spiritbuun/llama-cpp-turboquant-cuda.git
cd llama-cpp-turboquant-cuda && git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**Important:** Flash attention must be enabled. Add `-fa on` (server) or `-fa 1` (bench/CLI) to all commands. Without it, turbo3 will silently fail on pre-M5 hardware.

## How It Runs

```
                        turbo-diag
                            |
              +-------------+-------------+
              |                           |
         Creates venv              Installs rich
      (.turbo-diag-venv/)          (optional UI)
              |                           |
              +-------------+-------------+
                            |
                 turbo_hardware_diag.py
                            |
         +------------------+------------------+
         |                  |                  |
   Detect Hardware    Start Monitor      Start Display
   (CPU/GPU/RAM)     (10s CSV poll)    (rich or ASCII)
         |                  |                  |
         +------------------+------------------+
                            |
              Run 13 Diagnostic Sections
              +---------------------------------+
              |  1. Hardware Inventory           |
              |  2. System Load (pre)            |
              |  3. Model Info                   |
              |  4. GPU Capabilities             |
              |  5. Build Validation             |
              |  6. Prefill Speed (5 depths)     |  <-- llama-bench
              |  7. Decode Speed (5 depths)      |  <-- llama-bench
              |  8. Stress Test (11 depths)      |  <-- llama-bench
              |  9. Combined Workload            |  <-- llama-bench
              | 10. Perplexity (quality gate)    |  <-- llama-perplexity
              | 11. Memory Breakdown             |  <-- llama-cli
              | 12. System Load (post)           |
              | 13. Summary + Anomaly Report     |
              +---------------------------------+
                            |
              +-------------+-------------+
              |             |             |
         .txt log     .json profile  .csv monitor
              |             |             |
              +-------------+-------------+
                            |
                     turbo-diag-*.zip
                    (send this to us!)
```

---

## What It Generates

The diagnostic produces a single zip file you send back to the team:

```
turbo-diag-20260326-143022.zip
├── turbo-diag-20260326-143022.txt       # Human-readable log (all sections)
├── turbo-hwprofile-20260326-143022.json  # Machine-parseable hardware profile
└── turbo-monitor-20260326-143022.csv     # Background system metrics (polled every 10s)
```

### Output Files

| File | Format | Purpose |
|------|--------|---------|
| `turbo-diag-*.txt` | Plain text | Full diagnostic log — hardware inventory, benchmark tables, anomaly flags, and how-to-read guide. Compatible with `hw_replay.py` for programmatic parsing. |
| `turbo-hwprofile-*.json` | JSON | Structured hardware profile: CPU, RAM, GPU family, model metadata, and build info. Used for cross-machine comparison. |
| `turbo-monitor-*.csv` | CSV | Background metrics captured every 10 seconds during the entire run. |

### Monitor CSV Columns

| Column | Description |
|--------|-------------|
| `timestamp` | UTC ISO 8601 timestamp |
| `load_1m` | 1-minute load average |
| `mem_pressure_pct` | Memory pressure percentage (active+wired pages) |
| `swap_used_mb` | Swap usage in MB |
| `gpu_temp_c` | GPU temperature in Celsius (NVIDIA only; `N/A` on macOS) |
| `cpu_speed_limit` | macOS CPU speed limit percentage (100 = no throttling) |
| `gpu_mem_used_mb` | GPU memory used in MB (`unified` on Apple Silicon) |
| `gpu_util_pct` | GPU utilization percentage (NVIDIA only) |

---

## How to Read Results

The `.txt` log is organized into 13 numbered sections. Here's what to look at, in order of importance:

### Check quality first (Section 10)

Perplexity (PPL) must be within 2% of q8_0. **If PPL is broken, all speed numbers are meaningless.**

### Decode speed — the critical metric (Section 7)

This is the test that matters most. Look at the turbo3/q8_0 ratio at each context depth:

- **Healthy (M5 Max):** `0.92x (short) → 0.72x (48K)` — gradual, predictable degradation
- **Problem (M1):** `0.90x (short) → 0.09x (42K)` — constant cache thrashing, falls off a cliff

### Prefill speed (Section 6)

turbo3/q8_0 ratio should be flat at 0.95–1.00x across all depths. If it degrades with context length, that's a context scaling regression.

### Stress test inflection point (Section 8)

Fine-grained decode scaling gradient. Shows the exact context depth where turbo3 decode starts falling apart. This is the most useful data for diagnosing constant cache issues on specific hardware.

### Thermal comparison (Sections 2 vs 12)

Compare pre-benchmark and post-benchmark system load. If `CPU_Speed_Limit` dropped below 100 during the test, results may be artificially low due to thermal throttling.

### Memory breakdown (Section 11)

KV cache sizing at multiple context lengths. If the system is near its `recommendedMaxWorkingSetSize`, swap pressure will kill decode performance.

### Section-by-Section Reference

| Section | Title | What It Tells You |
|---------|-------|-------------------|
| 1 | Hardware Inventory | CPU, RAM, GPU, cache hierarchy, and power state (no PII) |
| 2 | System Load (pre-benchmark) | Baseline load, memory pressure, top CPU consumers, and disk I/O |
| 3 | Model Info | Model architecture, params, layer count, expert count, and file type |
| 4 | GPU Device Capabilities | GPU family, Metal version, Tensor API support, and unified memory |
| 5 | Build Validation | Confirms turbo3 type is available, Metal library loads, and build commit |
| 6 | Prefill Speed | Prefill tok/s at 2K/4K/8K/16K/32K for q8_0, turbo3, and mode 2 |
| 7 | Decode Speed | Decode tok/s at short/4K/8K/16K/32K — **the critical test** |
| 8 | Constant Cache Stress Test | Fine-grained decode gradient at 2K–32K in ~2K increments |
| 9 | Combined Prefill+Decode | Realistic workload simulation (prefill then decode) |
| 10 | Perplexity | Quality validation — PPL for q8_0, turbo3, and turbo3 mode 2 |
| 11 | Memory Breakdown | KV cache sizing at multiple context lengths |
| 12 | System Load (post-benchmark) | Post-test load, thermal throttling check |
| 13 | Diagnostic Summary | All anomalies, how-to-read guide, and next steps |

---

## Known Baselines

### M5 Max Reference Numbers (Qwen 35B MoE Q8_0)

| Metric | Expected | Notes |
|--------|----------|-------|
| Prefill ratio (turbo3/q8_0) | 0.95–1.00x | Flat across all depths |
| Decode ratio (short context) | ~0.92x | Small overhead from WHT transform |
| Decode ratio (32K context) | ~0.75x | Gradual degradation, still usable |
| PPL delta vs q8_0 | < 2% | Typically < 1% |

### M1 Max 64GB Community Results (Qwen 35B MoE Q8_0)

#### Tester 1: @Ambisphaeric

*2026-03-26. GPU: Apple M1 Max, 32 cores, 64GB unified.*

**Prefill (tok/s):**

| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| 2K | 855.7 | 836.1 | 0.977x |
| 4K | 805.2 | 784.0 | 0.974x |
| 8K | 724.6 | 696.3 | 0.961x |
| 16K | 600.5 | 569.6 | 0.949x |
| 32K | 444.9 | 414.0 | 0.931x |

**Decode (tok/s, tg128):**

| Context | q8_0 | turbo3 | Ratio | Flag |
|---------|------|--------|-------|------|
| short | 41.2 | 34.5 | 0.837x | |
| 4K | 31.9 | 20.4 | 0.640x | |
| 8K | 27.5 | 14.8 | 0.538x | |
| 16K | 20.7 | 9.4 | 0.454x | ANOMALY |
| 32K | 13.7 | 5.3 | 0.387x | ANOMALY |

#### Tester 2: @mariotomich

*2026-03-26. GPU: Apple M1 Max, 32 cores, 64GB unified. Clean system (94% free memory, low load). Full diagnostic run with mode 2.*

**Prefill (tok/s):**

| Context | q8_0 | turbo3 | mode2 | turbo3 ratio | mode2 ratio |
|---------|------|--------|-------|-------------|-------------|
| 2K | 859.2 | 844.5 | 848.2 | 0.983x | 0.987x |
| 4K | 810.5 | 789.6 | 795.6 | 0.974x | 0.982x |
| 8K | 731.2 | 698.9 | 707.1 | 0.956x | 0.967x |
| 16K | 599.9 | 570.9 | 575.9 | 0.952x | 0.960x |
| 32K | 450.0 | 416.4 | 423.5 | 0.925x | 0.941x |

**Decode (tok/s, tg128):**

| Context | q8_0 | turbo3 | mode2 | turbo3 ratio | mode2 ratio | Flag |
|---------|------|--------|-------|-------------|-------------|------|
| short | 42.9 | 35.6 | 36.9 | 0.830x | 0.860x | |
| 4K | 34.3 | 20.8 | 22.1 | 0.606x | 0.645x | |
| 8K | 28.1 | 14.8 | 16.3 | 0.526x | 0.580x | ANOMALY |
| 16K | 21.0 | 8.7 | 9.7 | 0.414x | 0.462x | ANOMALY |
| 32K | 13.8 | 5.4 | 6.1 | 0.391x | 0.442x | ANOMALY |

**Cross-validation:** Both M1 Max testers show nearly identical results, confirming the M1 decode regression is hardware-dependent, not user-specific. Mode 2 (layer-adaptive) consistently adds ~5% decode improvement over uniform turbo3.

Decode degrades significantly at long context. Known M1 issue: constant memory LUT contention in the flash attention dequant path. M5 Max (with tensor API) does not have this problem.

### M2 Pro 32GB (Mac Mini) Results (Qwen2.5-7B Q4_K_M) — Partial

*Internal test (2026-03-26). GPU: Apple M2 Pro, 19-core GPU, Apple8 family (1008), has_tensor=false, 32GB unified, Metal 4. Build: 02268fc (norm correction). Clean run after killing competing workloads.*

**Prefill (tok/s):**

| Context | q8_0 | turbo3 | Ratio |
|---------|------|--------|-------|
| 2K | 155.7 | 263.6 | **1.69x** |
| 4K | 143.2 | 245.3 | **1.71x** |
| 8K | 119.5 | 206.3 | **1.73x** |
| 16K | 93.5 | 144.3 | **1.54x** |
| 32K | 63.3 | 83.7 | **1.32x** |

turbo3 prefill is 32-73% **faster** than q8_0 on M2 Pro. Much bigger gap than M5 Max (~parity). Hypothesis: M2 Pro's lower memory bandwidth (200 GB/s vs M5 Max 546 GB/s) makes the compressed cache win harder since turbo3 moves less data per token.

**Decode:** Test hung at 32K context, no decode data collected. Rerun pending with timeout fix. Expect M1-like decode regression (has_tensor=false, same constant memory LUT issue).

**Mode 2 prefill:** Hung at pp16384. High variance at 8K+ suggests memory pressure beginning (32GB with 4GB model + KV cache + system overhead).

---

### What "Good" vs "Bad" Looks Like

| Indicator | Good | Bad |
|-----------|------|-----|
| Decode ratio curve | Gradual slope, stays above 0.70x at 32K | Cliff drop below 0.50x at any depth |
| Prefill ratio | Flat 0.95–1.00x | Drops with context depth |
| PPL delta | < 2% | > 5% (quality regression) |
| CPU Speed Limit | 100 before and after | Drops below 90 during test |
| Swap growth | < 50 MB | > 100 MB (memory pressure) |

### Anomaly Flags

The diagnostic auto-detects these issues and tags them `[ANOMALY]` in the log:

- **Steep decode degradation** — ratio drops >15% between consecutive depths
- **Thermal throttling** — `CPU_Speed_Limit` dropped below 100%
- **Memory pressure/swapping** — swap grew >100 MB during benchmarks
- **Low q8_0 baseline** — system was under load (q8_0 decode < 5 tok/s)
- **Quality regression** — turbo3 PPL >10% worse than q8_0
- **Unreliable 1K measurement** — Metal async dispatch artifact (>10,000 tok/s at 1K context)

---

## Hardware Replay System

The `hw_replay.py` module (`components/turboquant-reference/src/turboquant/hw_replay.py`) parses diagnostic output into structured Python objects for programmatic analysis.

### Load a diagnostic profile

```python
from turboquant.hw_replay import HardwareProfile

profile = HardwareProfile.from_diag_file("turbo-diag-20260326.txt")

# Or from the JSON profile
profile = HardwareProfile.from_json("turbo-hwprofile-20260326.json")
```

### Extract performance curves

```python
# Decode speed vs context depth
decode_curve = profile.get_decode_curve("turbo3")
# {0: 42.1, 4096: 38.7, 8192: 35.2, ...}

# turbo3/q8_0 ratio at each depth
ratio_curve = profile.get_ratio_curve("turbo3", "q8_0", "decode")
# {0: 0.92, 4096: 0.89, 8192: 0.85, ...}

# Find the inflection point (where decode falls off a cliff)
inflection = profile.find_decode_inflection("turbo3")
# 16384  (or None if the curve is smooth)
```

### Compare two profiles

```python
from turboquant.hw_replay import compare_profiles

m5 = HardwareProfile.from_diag_file("diag-m5-max.txt")
m1 = HardwareProfile.from_diag_file("diag-m1-ultra.txt")

report = compare_profiles(m5, m1)
print(report.to_markdown())
```

### Predict performance on untested hardware

```python
from turboquant.hw_replay import predict_decode_from_baseline

# Predict M1 (family 1007, no tensor API) from M5 baseline
predicted = predict_decode_from_baseline(
    baseline=m5,
    target_gpu_family_id=1007,
    target_has_tensor=False,
)
# {0: 0.92, 4096: 0.85, 8192: 0.71, 16384: 0.34, ...}
```

---

## Command Reference

```
usage: turbo_hardware_diag.py [-h] [--model MODEL] [--llama-dir LLAMA_DIR]
                              [--skip-ppl] [--skip-stress] [--verbose]
                              [--output-dir OUTPUT_DIR]
                              [llama_dir] [model_path]
```

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `llama_dir` | Path to llama.cpp directory (default: current directory) |
| `model_path` | Path to `.gguf` model file (auto-detected if not specified) |

### Optional Flags

| Flag | Description |
|------|-------------|
| `--model MODEL` | Path to `.gguf` model file (alternative to positional arg) |
| `--llama-dir DIR` | Path to llama.cpp directory (alternative to positional arg) |
| `--skip-ppl` | Skip perplexity tests — saves ~10 minutes |
| `--skip-stress` | Skip the fine-grained stress test (Section 8) |
| `--verbose`, `-v` | Enable verbose/debug-level logging |
| `--output-dir DIR`, `-o DIR` | Output directory for diagnostic files (default: current directory) |

### Examples

```bash
# Positional args (most common)
bash tools/diagnostics/turbo-diag ~/llama.cpp ~/models/Qwen3.5-35B-A3B-Q8_0.gguf

# Named flags
bash tools/diagnostics/turbo-diag --llama-dir ~/llama.cpp --model ~/models/model.gguf

# Quick run — skip slow tests
bash tools/diagnostics/turbo-diag ~/llama.cpp ~/models/model.gguf --skip-ppl --skip-stress

# Verbose output to a specific directory
bash tools/diagnostics/turbo-diag ~/llama.cpp ~/models/model.gguf -v -o ~/diag-results/
```

---

## Before You Run

**Plug in your charger and close heavy apps.** On Apple Silicon Macs, battery power triggers CPU/GPU throttling that produces artificially low speed numbers — we observed 2-3x slower benchmarks on an M5 Max running on battery vs plugged in. The diagnostic will detect throttling (`CPU_Speed_Limit` in Section 12), but the damage is already done to earlier results. NVIDIA laptops have similar power management behavior.

Close Chrome, video calls, and anything GPU-heavy before running. The diagnostic captures top CPU/GPU consumers (Section 2), so interference will be visible in the log, but clean results are always better.

---

## Troubleshooting

### "turbo3 not recognized" or cache type errors

Your llama.cpp build doesn't have turbo types compiled in. Rebuild with the TurboQuant patches applied:

```bash
cd /path/to/llama.cpp
cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### "No .gguf model found"

The auto-detection searches `llama.cpp/models/`, `../models/`, and `~/local_llms/models/`. If your model lives somewhere else, pass the path explicitly:

```bash
bash tools/diagnostics/turbo-diag ~/llama.cpp /path/to/your/model.gguf
```

### Benchmarks are painfully slow

Use the skip flags to cut runtime roughly in half:

```bash
bash tools/diagnostics/turbo-diag ~/llama.cpp model.gguf --skip-ppl --skip-stress
```

PPL tests take ~10 minutes, and the stress test adds another ~10 minutes of fine-grained decode sweeps.

### Script crashes on startup

- **Python version:** Requires 3.10+. The launcher checks this automatically, but if you're running `turbo_hardware_diag.py` directly, verify with `python3 --version`.
- **Missing binaries:** The script needs `llama-bench`, `llama-perplexity`, and `llama-cli` in `<llama-dir>/build/bin/`. If any are missing, rebuild llama.cpp.
- **Permissions:** On macOS, `system_profiler` and `pmset` may require user consent the first time. If hardware detection fails, the diagnostic still continues — it just logs warnings.

### Rich terminal UI not showing

If you see plain ASCII instead of color tables, the `rich` package didn't install. This is cosmetic — all results are identical. To fix:

```bash
pip install rich
# Or delete the venv and let turbo-diag recreate it:
rm -rf tools/diagnostics/.turbo-diag-venv
```

---

## Other Scripts

### turbo-quality-gate.sh

Pre-push quality and speed gate. Run before pushing any TurboQuant changes.

**What it checks:**
1. **Perplexity** — turbo3 PPL must be within 5% of q8_0 baseline (8 chunks of wikitext-2)
2. **Context scaling** — turbo3/q8_0 speed ratio at 4K context must be > 0.95x

**Usage:**
```bash
bash tools/validation/turbo-quality-gate.sh
# Exit 0 = PASS, Exit 1 = FAIL
```

Configure paths via environment variables: `LLAMA`, `MODEL`, `WIKI`.

### turbo-realworld-bench.sh

Real-world decode benchmark using a long PDF document (70+ pages). Compares turbo3 vs q8_0 at realistic context depths by spinning up `llama-server` instances, sending the full PDF as a prompt, and measuring prefill + decode speed.

**Usage:**
```bash
bash tools/benchmarks/turbo-realworld-bench.sh [path-to-pdf]
```

**Requirements:** A long PDF file (not included in repo), `pdftotext` (from poppler), and a built llama.cpp with turbo3 support. Configure via `LLAMA`, `MODEL`, `PORT_BASE`, `THREADS`, and `MAX_TOKENS` env vars.

---

## Sharing Results

After the diagnostic completes, you'll see:

```
============================================
  DIAGNOSTIC PACKAGE READY
============================================

  Zip file: /path/to/turbo-diag-20260326-143022.zip
  Contents:
    turbo-diag-20260326-143022.txt
    turbo-hwprofile-20260326-143022.json
    turbo-monitor-20260326-143022.csv

  Send this zip file to the TurboQuant team.
```

**How to share:**
1. **GitHub issue** (preferred): Open an issue at [Efficient LLM Systems](https://github.com/dipeshbabu/efficient-llm-systems/issues) with title "Diagnostic: [your hardware]" and attach the zip. GitHub supports attachments up to 25MB — the zip is typically under 100KB.
2. **DM on X/Twitter**: Send the zip file directly to [@dipeshbabu](https://x.com/dipeshbabu)

The zip is self-contained — everything we need is inside. No need to copy-paste logs or screenshot terminal output. The `.txt` log is human-readable if you want to look at it yourself first.

---

## Privacy

**No PII is collected.** The diagnostic is designed to be safe to share publicly.

### What IS collected

- Hardware specs: CPU model, core count, RAM size, GPU family, cache hierarchy
- GPU capabilities: Metal/CUDA version, Tensor API support, VRAM/unified memory size
- Benchmark numbers: tok/s for prefill, decode, combined, and perplexity
- System load: load average, memory pressure, swap usage, CPU speed limit
- Build info: llama.cpp commit hash
- Model metadata: filename, file size, architecture, layer/head/expert counts

### What is NOT collected

- Usernames, home directory paths, or hostnames
- Network configuration, IP addresses, or MAC addresses
- File system contents or directory listings
- Environment variables (except `TURBO_LAYER_ADAPTIVE` when explicitly set)
- Model weights or prompt content
- Any data that could identify you or your machine beyond hardware specs
