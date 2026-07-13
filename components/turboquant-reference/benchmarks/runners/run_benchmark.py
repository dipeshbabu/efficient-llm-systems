#!/usr/bin/env python3
"""
TurboQuant+ Benchmark Script

Runs llama-server with different KV cache types and measures performance
via the HTTP API. More reliable than llama-cli for benchmarking.

Usage: python components/turboquant-reference/benchmarks/runners/run_benchmark.py

Copyright 2026 Dipesh Tharu Mahato. Licensed under Apache 2.0.
"""

import json
import os
import subprocess
import time
from pathlib import Path

# Try to import requests, fall back to urllib
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    import urllib.error
    import urllib.request

    HAS_REQUESTS = False


LLAMA_DIR = os.path.expanduser("~/local_llms/llama.cpp")
SERVER_BIN = f"{LLAMA_DIR}/build-turbo/bin/llama-server"
MODELS = {
    "Qwen3.5-35B-A3B-MoE": os.path.expanduser(
        "~/local_llms/models/Qwen3.5-35B-A3B-Q8_0.gguf"
    ),
    "Qwopus-v2-27B": os.path.expanduser("~/local_llms/models/Qwen3.5-27B.Q8_0.gguf"),
}
CACHE_TYPES = ["q8_0", "q4_0", "turbo3", "turbo4"]
PORT = 8090  # use non-default to avoid conflicts
PROMPT = "Explain the concept of KV cache compression for large language models in exactly three sentences."
N_PREDICT = 100


def api_request(endpoint, data=None):
    """Make HTTP request to llama-server."""
    url = f"http://localhost:{PORT}{endpoint}"
    if HAS_REQUESTS:
        if data:
            r = requests.post(url, json=data, timeout=300)
        else:
            r = requests.get(url, timeout=30)
        return r.json()
    else:
        if data:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
            )
        else:
            req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())


def wait_for_server(timeout=120):
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            api_request("/health")
            return True
        except Exception:
            time.sleep(2)
    return False


def run_benchmark(model_name, model_path, cache_type):
    """Start server, run completion, measure performance, stop server."""
    print(f"  Starting server with -ctk {cache_type} -ctv {cache_type}...")

    # Start server
    cmd = [
        SERVER_BIN,
        "-m",
        model_path,
        "-ngl",
        "99",
        "-c",
        "4096",
        "-fa",
        "on",
        "-ctk",
        cache_type,
        "-ctv",
        cache_type,
        "-np",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        # Wait for ready
        if not wait_for_server(timeout=180):
            print("    TIMEOUT: server didn't start in 180s")
            proc.kill()
            return None

        print("    Server ready. Running completion...")

        # Run completion
        t0 = time.time()
        try:
            result = api_request(
                "/v1/chat/completions",
                {
                    "model": "test",
                    "messages": [{"role": "user", "content": PROMPT}],
                    "max_tokens": N_PREDICT,
                    "temperature": 0.0,
                },
            )
        except Exception:
            # Try non-chat endpoint
            try:
                result = api_request(
                    "/completion",
                    {
                        "prompt": PROMPT,
                        "n_predict": N_PREDICT,
                        "temperature": 0.0,
                    },
                )
            except Exception as e2:
                print(f"    ERROR: {e2}")
                return None
        elapsed = time.time() - t0

        # Extract timing from /metrics
        try:
            _metrics = api_request("/metrics")
        except Exception:
            _metrics = None

        # Extract timing from /slots
        try:
            _slots = api_request("/slots")
        except Exception:
            _slots = None

        # Parse results
        output = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not output:
            output = result.get("content", "")

        tokens_generated = len(output.split()) * 1.3  # rough estimate
        tok_s = tokens_generated / elapsed if elapsed > 0 else 0

        # Get actual usage if available
        usage = result.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        if completion_tokens > 0:
            tok_s = completion_tokens / elapsed

        return {
            "cache_type": cache_type,
            "model": model_name,
            "elapsed_s": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tok_s": tok_s,
            "output_preview": output[:100] + "..." if len(output) > 100 else output,
        }

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)  # let port release


def main():
    print("=" * 70)
    print("TURBOQUANT+ BENCHMARK")
    print("=" * 70)

    results = []

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_name}: {model_path} not found")
            continue

        print(f"\n{'─' * 70}")
        print(f"Model: {model_name}")
        print(f"Path: {model_path}")
        print(f"{'─' * 70}")

        for cache_type in CACHE_TYPES:
            result = run_benchmark(model_name, model_path, cache_type)
            if result:
                results.append(result)
                print(
                    f"    {cache_type}: {result['tok_s']:.1f} tok/s, "
                    f"{result['completion_tokens']} tokens in {result['elapsed_s']:.1f}s"
                )
                print(f"    Preview: {result['output_preview'][:80]}")
            else:
                print(f"    {cache_type}: FAILED")
            print()

    # Write results
    if results:
        report_path = Path(
            os.environ.get(
                "TURBOQUANT_BENCH_RESULTS", "turboquant-benchmark-results.md"
            )
        ).expanduser()
        with report_path.open("w", encoding="utf-8") as f:
            f.write("# TurboQuant+ Benchmark Results\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write("Hardware: Apple M5 Max 128GB\n")
            f.write(f'Prompt: "{PROMPT}"\n')
            f.write(f"Max tokens: {N_PREDICT}\n\n")

            f.write(
                "| Model | Cache Type | Bits/val | tok/s | Tokens | Time (s) | Compression |\n"
            )
            f.write(
                "|-------|-----------|----------|-------|--------|----------|-------------|\n"
            )

            for r in results:
                bits = {
                    "q8_0": "8",
                    "q4_0": "4",
                    "turbo3": "3.25",
                    "turbo4": "4.25",
                }.get(r["cache_type"], "?")
                compression = {
                    "q8_0": "2.0×",
                    "q4_0": "4.0×",
                    "turbo3": "4.9×",
                    "turbo4": "3.8×",
                }.get(r["cache_type"], "?")
                f.write(
                    f"| {r['model']} | {r['cache_type']} | {bits} | "
                    f"{r['tok_s']:.1f} | {r['completion_tokens']} | "
                    f"{r['elapsed_s']:.1f} | {compression} |\n"
                )

            f.write("\n## Output Samples\n\n")
            for r in results:
                f.write(f"### {r['model']} / {r['cache_type']}\n")
                f.write(f"```\n{r['output_preview']}\n```\n\n")

        print(f"\nResults saved to {report_path}")
        print("\nSummary:")
        print(f"{'Model':<25} {'Cache':<10} {'tok/s':>8} {'Compression':>12}")
        print(f"{'─' * 60}")
        for r in results:
            compression = {
                "q8_0": "2.0×",
                "q4_0": "4.0×",
                "turbo3": "4.9×",
                "turbo4": "3.8×",
            }.get(r["cache_type"], "?")
            print(
                f"{r['model']:<25} {r['cache_type']:<10} {r['tok_s']:>8.1f} {compression:>12}"
            )


if __name__ == "__main__":
    main()
