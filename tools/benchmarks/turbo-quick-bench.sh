#!/bin/bash
# TurboQuant Quick Benchmark — rapid warm fuzzy / red flag test
#
# Runs q8_0, turbo3, turbo4 against a model and reports:
#   - PPL (8-chunk, c=512) — ~30 sec per type
#   - Decode speed (tg128) — ~10 sec per type
#   - NIAH single needle at 3 positions — ~60 sec per type
#
# Total: ~5 min per model. Enough to catch regressions or format issues.
#
# Usage: bash tools/benchmarks/turbo-quick-bench.sh [--no-ref] <model.gguf> [llama_dir]
#
# Options:
#   --no-ref    Report numbers only, skip pass/fail judgment.
#               Use this for models without known reference values.
#
# Example:
#   bash tools/benchmarks/turbo-quick-bench.sh ~/models/Qwen3.5-35B-A3B-Q8_0.gguf
#   bash tools/benchmarks/turbo-quick-bench.sh --no-ref ~/models/Phi-3-mini-Q4_K_M.gguf

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NIAH_SCRIPT="${REPO_ROOT}/tools/validation/niah_test.py"

# Parse args
NO_REF=0
POSITIONAL=()
for arg in "$@"; do
    case $arg in
        --no-ref) NO_REF=1 ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done
set -- "${POSITIONAL[@]}"

MODEL="${1:?Usage: turbo-quick-bench.sh [--no-ref] <model.gguf> [llama_dir]}"
LLAMA_DIR="${2:-$HOME/local_llms/llama.cpp}"
BUILD="${LLAMA_DIR}/build-turbo/bin"
WIKI="${LLAMA_DIR}/wikitext-2-raw/wiki.test.raw"

# Validate
[ -f "$MODEL" ] || { echo "ERROR: Model not found: $MODEL"; exit 1; }
[ -f "$BUILD/llama-perplexity" ] || { echo "ERROR: llama-perplexity not found in $BUILD"; exit 1; }
[ -f "$WIKI" ] || { echo "WARNING: wikitext-2 not found, skipping PPL"; SKIP_PPL=1; }

MODEL_NAME=$(basename "$MODEL" .gguf)
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

echo "========================================"
echo "  TurboQuant Quick Benchmark"
echo "  Model: $MODEL_NAME"
echo "  Date:  $(date -u +%Y-%m-%d\ %H:%M\ UTC)"
echo "========================================"
echo ""

# Known good values (Qwen3.5-35B-A3B Q8_0 on M5 Max)
# Update these if you change reference model
REF_PPL_Q8="6.11"
REF_PPL_T3="6.18"
REF_PPL_T4="6.13"
REF_DEC_Q8="86"
REF_DEC_T3="77"
REF_DEC_T4="80"

PASS=0
WARN=0
FAIL=0

check_ppl() {
    local label=$1 ppl=$2 ref=$3 max_delta=${4:-5.0}
    # Validate numeric input
    if ! python3 -c "float('$ppl')" 2>/dev/null; then
        echo "  $label: PPL parse error ($ppl) 🔴"
        FAIL=$((FAIL + 1))
        return
    fi
    if [ "$NO_REF" -eq 1 ]; then
        echo "  $label: PPL $ppl"
        return
    fi
    local delta=$(python3 -c "print(f'{(($ppl - $ref) / $ref * 100):.2f}')")
    local abs_delta=$(python3 -c "print(abs(($ppl - $ref) / $ref * 100))")
    local status="✅"
    if python3 -c "exit(0 if $abs_delta > $max_delta else 1)"; then
        status="🔴 INVESTIGATE"
        FAIL=$((FAIL + 1))
    elif python3 -c "exit(0 if $abs_delta > 2.0 else 1)"; then
        status="⚠️  WARNING"
        WARN=$((WARN + 1))
    else
        PASS=$((PASS + 1))
    fi
    echo "  $label: PPL $ppl (${delta}% vs ref) $status"
}

check_speed() {
    local label=$1 speed=$2 min_speed=$3
    # Validate numeric input
    if ! python3 -c "float('$speed')" 2>/dev/null; then
        echo "  $label: speed parse error ($speed) 🔴"
        FAIL=$((FAIL + 1))
        return
    fi
    if [ "$NO_REF" -eq 1 ]; then
        echo "  $label: ${speed} tok/s"
        return
    fi
    local status="✅"
    if python3 -c "exit(0 if $speed < $min_speed * 0.5 else 1)"; then
        status="🔴 INVESTIGATE"
        FAIL=$((FAIL + 1))
    elif python3 -c "exit(0 if $speed < $min_speed * 0.8 else 1)"; then
        status="⚠️  WARNING"
        WARN=$((WARN + 1))
    else
        PASS=$((PASS + 1))
    fi
    echo "  $label: ${speed} tok/s $status"
}

# ==========================================
# 1. PPL (fast: 8 chunks, c=512)
# ==========================================
if [ -z "$SKIP_PPL" ]; then
    echo "--- PPL (c=512, 8 chunks) ---"
    for ct in q8_0 turbo3 turbo4; do
        PPL=$($BUILD/llama-perplexity -m "$MODEL" -f "$WIKI" -c 512 \
            -ctk $ct -ctv $ct -fa on --chunks 8 -ngl 99 2>&1 | \
            grep "Final estimate" | awk '{print $5}')

        if [ -z "$PPL" ]; then
            echo "  $ct: FAILED TO RUN 🔴"
            FAIL=$((FAIL + 1))
            continue
        fi

        case $ct in
            q8_0)   check_ppl "$ct" "$PPL" "$REF_PPL_Q8" 10.0 ;;
            turbo3) check_ppl "$ct" "$PPL" "$REF_PPL_T3" 5.0 ;;
            turbo4) check_ppl "$ct" "$PPL" "$REF_PPL_T4" 5.0 ;;
        esac
    done
    echo ""
fi

# ==========================================
# 2. Decode Speed (tg128, short context)
# ==========================================
echo "--- Decode Speed (tg128) ---"
for ct in q8_0 turbo3 turbo4; do
    SPEED=$($BUILD/llama-bench -m "$MODEL" -ctk $ct -ctv $ct \
        -fa 1 -ngl 99 -p 0 -n 128 2>&1 | \
        grep "tg128" | awk -F'|' '{print $(NF-1)}' | awk '{print $1}')

    if [ -z "$SPEED" ]; then
        echo "  $ct: FAILED TO RUN 🔴"
        FAIL=$((FAIL + 1))
        continue
    fi

    case $ct in
        q8_0)   check_speed "$ct" "$SPEED" "10" ;;
        turbo3) check_speed "$ct" "$SPEED" "10" ;;
        turbo4) check_speed "$ct" "$SPEED" "10" ;;
    esac
done
echo ""

# ==========================================
# 3. Quick NIAH (3 positions, 8K only)
# ==========================================
echo "--- NIAH Quick (8K, 3 positions) ---"
if [ -f "$NIAH_SCRIPT" ]; then
    for ct in q8_0 turbo3 turbo4; do
        RESULT=$(python3 "$NIAH_SCRIPT" "$LLAMA_DIR" "$MODEL" \
            --mode single --cache-types $ct --depths 8192 \
            --depths-sweep 0,50,100 \
            --server-bin "$BUILD/llama-server" \
            --output-dir "/tmp/turbo-quick-niah-${ct}" \
            2>&1 | grep "^|" | grep -c "✅" || true)

        TOTAL=3
        MISSED=$((TOTAL - RESULT))
        if [ "$MISSED" -eq 0 ]; then
            echo "  $ct: ${RESULT}/${TOTAL} ✅"
            PASS=$((PASS + 1))
        elif [ "$MISSED" -le 1 ]; then
            echo "  $ct: ${RESULT}/${TOTAL} ⚠️  WARNING"
            WARN=$((WARN + 1))
        else
            echo "  $ct: ${RESULT}/${TOTAL} 🔴 INVESTIGATE"
            FAIL=$((FAIL + 1))
        fi
    done
else
    echo "  (niah_test.py not found, skipping)"
fi
echo ""

# ==========================================
# Summary
# ==========================================
echo "========================================"
echo "  SUMMARY: $MODEL_NAME"
if [ "$NO_REF" -eq 1 ]; then
    echo "  MODE: report only (--no-ref)"
    echo "  STATUS: review numbers manually"
else
    echo "  PASS: $PASS  WARN: $WARN  FAIL: $FAIL"
    if [ "$FAIL" -gt 0 ]; then
        echo "  STATUS: 🔴 ISSUES FOUND — investigate before using"
    elif [ "$WARN" -gt 0 ]; then
        echo "  STATUS: ⚠️  WARNINGS — review results"
    else
        echo "  STATUS: ✅ ALL CLEAR"
    fi
fi
echo "========================================"
