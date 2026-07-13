#!/bin/bash
# TurboQuant real-world benchmark — tests decode speed with Mario's 70-page PDF
# Compares turbo3 vs q8_0 at realistic context depths
#
# Usage: bash tools/benchmarks/turbo-realworld-bench.sh [path-to-pdf]
# Default PDF: ~/Downloads/70_Pages_of_Fitness_Notes_-_A_Collection.pdf
#
# Requirements:
# - llama.cpp built with turbo3 support
# - PDF file (not included in repo — get from Mario or use your own long document)
# - python3 with no extra deps

set -e

LLAMA=${LLAMA:-~/local_llms/llama.cpp/build-turbo/bin}
MODEL=${MODEL:-~/local_llms/models/Qwen3.5-35B-A3B-Q8_0.gguf}
PDF=${1:-~/Downloads/70_Pages_of_Fitness_Notes_-_A_Collection.pdf}
PORT_BASE=${PORT_BASE:-18100}
THREADS=${THREADS:-6}
MAX_TOKENS=${MAX_TOKENS:-64}

if [ ! -f "$PDF" ]; then
    echo "ERROR: PDF not found at $PDF"
    echo "Usage: bash tools/benchmarks/turbo-realworld-bench.sh [path-to-pdf]"
    echo ""
    echo "Download the test PDF or provide your own long document."
    echo "The PDF is NOT included in the repo."
    exit 1
fi

# Extract text from PDF (crude but works for most PDFs)
echo "Extracting text from PDF..."
PDF_TEXT=$(python3 -c "
import subprocess, sys
try:
    # Try pdftotext first (poppler)
    result = subprocess.run(['pdftotext', '-layout', '$PDF', '-'], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout[:100000])  # Cap at ~100K chars
        sys.exit(0)
except FileNotFoundError:
    pass

# Fallback: read raw bytes and extract printable ASCII
with open('$PDF', 'rb') as f:
    data = f.read()
text = data.decode('latin-1', errors='ignore')
# Extract text between BT/ET markers (crude PDF text extraction)
import re
chunks = re.findall(r'\(([^)]+)\)', text)
print(' '.join(chunks)[:100000])
" 2>/dev/null)

if [ -z "$PDF_TEXT" ]; then
    echo "ERROR: Could not extract text from PDF. Install poppler: brew install poppler"
    exit 1
fi

CHAR_COUNT=${#PDF_TEXT}
echo "Extracted ~$CHAR_COUNT characters from PDF"
echo ""

PROMPT="Here is a document:\n\n${PDF_TEXT}\n\nWhat are the top 10 lessons from this document?"

echo "========================================"
echo "  TurboQuant Real-World Benchmark"
echo "  PDF: $(basename $PDF)"
echo "  Model: $(basename $MODEL)"
echo "========================================"
echo ""

for CACHE_TYPE in turbo3 q8_0; do
    PORT=$((PORT_BASE))
    PORT_BASE=$((PORT_BASE + 1))

    echo "--- $CACHE_TYPE ---"
    $LLAMA/llama-server \
        -m $MODEL -ngl 99 -c 65536 -fa on \
        --cache-type-k $CACHE_TYPE --cache-type-v $CACHE_TYPE \
        --host 127.0.0.1 --port $PORT \
        -t $THREADS --no-warmup -np 1 --cache-ram 0 2>/dev/null &
    SERVER_PID=$!
    sleep 15

    # Send the PDF prompt
    RESULT=$(curl -s http://127.0.0.1:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"$(echo "$PROMPT" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read())[1:-1])")\"}],\"max_tokens\":$MAX_TOKENS,\"temperature\":0}" 2>/dev/null)

    python3 -c "
import json, sys
d = json.loads('''$RESULT''')
t = d.get('timings', {})
u = d.get('usage', {})
print(f'  Prompt tokens: {u.get(\"prompt_tokens\", \"?\")}')
print(f'  Prefill: {t.get(\"prompt_per_second\", 0):.1f} tok/s')
print(f'  Decode:  {t.get(\"predicted_per_second\", 0):.1f} tok/s')
print(f'  Completion tokens: {u.get(\"completion_tokens\", \"?\")}')
" 2>/dev/null || echo "  ERROR: Could not parse server response"

    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo ""
done

echo "========================================"
echo "  Benchmark complete"
echo "========================================"
