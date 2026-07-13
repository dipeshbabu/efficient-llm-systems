#!/bin/bash
# Benchmark TurboQuant vs q8_0/q4_0 on M5 Max
# Usage: bash components/turboquant-reference/benchmarks/runners/benchmark_llama.sh
#
# Copyright 2026 Dipesh Tharu Mahato. Licensed under Apache 2.0.

LLAMA_DIR="$HOME/local_llms/llama.cpp"
MODEL="$HOME/local_llms/models/Qwen3.5-35B-A3B-Q8_0.gguf"
CLI="$LLAMA_DIR/build-turbo/bin/llama-cli"
RESULTS_FILE="${RESULTS_FILE:-turboquant-benchmark-results.md}"

echo "=============================================="
echo "TurboQuant Benchmark — M5 Max 128GB"
echo "Model: Qwen 3.5 35B-A3B MoE Q8_0"
echo "=============================================="
echo ""

echo "| Cache Type | Bits/val | Prompt tok/s | Gen tok/s | KV Compression |" > "$RESULTS_FILE"
echo "|------------|----------|-------------|-----------|----------------|" >> "$RESULTS_FILE"

for CACHE_TYPE in "q8_0" "q4_0" "turbo3" "turbo4"; do
    echo "--- Testing $CACHE_TYPE ---"

    # Run with --perf and capture timing lines
    OUTPUT=$($CLI \
        -m "$MODEL" \
        -ngl 99 -fa on \
        -c 2048 \
        -ctk "$CACHE_TYPE" -ctv "$CACHE_TYPE" \
        -n 50 -p "Explain quantum computing in simple terms." \
        --no-display-prompt --jinja \
        2>&1)

    # Extract timing info
    PP=$(echo "$OUTPUT" | grep "prompt eval" | grep -oE '[0-9]+\.[0-9]+ tokens per second' | grep -oE '[0-9]+\.[0-9]+')
    TG=$(echo "$OUTPUT" | grep "eval time\|generation" | grep -oE '[0-9]+\.[0-9]+ tokens per second' | grep -oE '[0-9]+\.[0-9]+')

    # Compression ratio
    case $CACHE_TYPE in
        q8_0)   RATIO="2.0×" ; BITS="8" ;;
        q4_0)   RATIO="4.0×" ; BITS="4" ;;
        turbo3) RATIO="4.9×" ; BITS="3.25" ;;
        turbo4) RATIO="3.8×" ; BITS="4.25" ;;
    esac

    echo "  Prompt: ${PP:-N/A} tok/s, Gen: ${TG:-N/A} tok/s"
    echo "| $CACHE_TYPE | $BITS | ${PP:-N/A} | ${TG:-N/A} | $RATIO |" >> "$RESULTS_FILE"
done

echo ""
echo "Results saved to $RESULTS_FILE"
cat "$RESULTS_FILE"
