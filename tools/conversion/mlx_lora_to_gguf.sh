#!/bin/bash
# mlx_lora_to_gguf.sh — Merge MLX LoRA adapter into HF base model and convert to GGUF
#
# Instead of using mlx_lm.fuse (which outputs MLX-sanitized tensor names that
# convert_hf_to_gguf.py can't handle for MoE/hybrid architectures), this script
# applies LoRA deltas directly to the original HF base weights. The result is a
# clean HF-format model that converts to GGUF without any rename/unsanitize hacks.
#
# Supports: MoE expert layers (SwitchGLU gate_up fusion), linear attention (Mamba),
#           standard attention, shared experts, embeddings
#
# Tested on: Qwen3.5-35B-A3B (256-expert MoE + GatedDeltaNet linear attention)
#
# Requires: python3, mlx, safetensors

set -e

usage() {
    cat << 'EOF'
Usage: $(basename "$0") [OPTIONS] --base <model> --adapter <path> --output <name>

Merge an MLX LoRA adapter into a HuggingFace base model and convert to GGUF.

Required:
  --base <path>         Path to base HuggingFace model (safetensors, BF16)
  --adapter <path>      Path to MLX LoRA adapter directory
  --output <name>       Output name (creates <name>-Q8_0.gguf)

Optional:
  --llama-cpp <path>    Path to llama.cpp directory (default: auto-detect)
  --output-dir <path>   Output directory (default: current directory)
  --outtype <type>      GGUF output type: bf16, f16, q8_0 (default: q8_0)
  --compress            Also compress with TQ4_1S Config I
  --keep-merged         Keep the intermediate merged HF safetensors
  --help                Show this help

How it works:
  1. Loads the LoRA adapter weights (lora_a/lora_b pairs)
  2. Computes deltas using the exact MLX LoRA fuse formulas:
     - LoRALinear:       delta = scale * lora_b.T @ lora_a.T
     - LoRASwitchLinear: delta = scale * lora_b @ lora_a
  3. Maps MLX tensor names back to HF names:
     - language_model.model.X → model.language_model.X
     - switch_mlp.{gate,up}_proj → experts.gate_up_proj (fused)
     - switch_mlp.down_proj → experts.down_proj
  4. Adds deltas directly to base HF weight tensors
  5. Converts to GGUF using llama.cpp's convert_hf_to_gguf.py

Examples:
  $(basename "$0") --base ~/models/Qwen3.5-35B-A3B-BF16 \
      --adapter ~/models/my-adapter --output MyModel

  $(basename "$0") --base ~/models/Qwen3.5-35B-A3B-BF16 \
      --adapter ~/models/my-adapter --output MyModel \
      --compress --output-dir ~/models
EOF
    exit 0
}

# Parse args
BASE_MODEL=""
ADAPTER_PATH=""
OUTPUT_NAME=""
LLAMA_CPP=""
OUTPUT_DIR="."
OUTTYPE="q8_0"
COMPRESS=false
KEEP_MERGED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --base) BASE_MODEL="$2"; shift 2 ;;
        --adapter) ADAPTER_PATH="$2"; shift 2 ;;
        --output) OUTPUT_NAME="$2"; shift 2 ;;
        --llama-cpp) LLAMA_CPP="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --outtype) OUTTYPE="$2"; shift 2 ;;
        --compress) COMPRESS=true; shift ;;
        --keep-merged) KEEP_MERGED=true; shift ;;
        --help|-h) usage ;;
        *) echo "Error: unknown option: $1"; echo ""; usage ;;
    esac
done

if [[ -z "$BASE_MODEL" || -z "$ADAPTER_PATH" || -z "$OUTPUT_NAME" ]]; then
    echo "Error: --base, --adapter, and --output are required."
    echo ""
    usage
fi

# Validate inputs
if [[ ! -d "$BASE_MODEL" ]]; then
    echo "Error: base model directory not found: $BASE_MODEL"
    exit 1
fi
if [[ ! -d "$ADAPTER_PATH" ]]; then
    echo "Error: adapter directory not found: $ADAPTER_PATH"
    exit 1
fi
if [[ ! -f "$ADAPTER_PATH/adapters.safetensors" ]]; then
    echo "Error: adapters.safetensors not found in $ADAPTER_PATH"
    exit 1
fi
if [[ ! -f "$ADAPTER_PATH/adapter_config.json" ]]; then
    echo "Error: adapter_config.json not found in $ADAPTER_PATH"
    exit 1
fi

# Auto-detect llama.cpp
if [[ -z "$LLAMA_CPP" ]]; then
    for candidate in "." ".." "$HOME/local_llms/llama.cpp" "$HOME/llama.cpp"; do
        if [[ -f "$candidate/convert_hf_to_gguf.py" ]]; then
            LLAMA_CPP="$candidate"
            break
        fi
    done
    if [[ -z "$LLAMA_CPP" ]]; then
        echo "Error: cannot find convert_hf_to_gguf.py. Use --llama-cpp to specify llama.cpp path."
        exit 1
    fi
fi

if $COMPRESS && [[ ! -f "${LLAMA_CPP}/build/bin/llama-quantize" ]]; then
    echo "Error: llama-quantize not found at ${LLAMA_CPP}/build/bin/llama-quantize"
    echo "Build llama.cpp first: cmake -B build && cmake --build build -t llama-quantize"
    exit 1
fi

MERGED_PATH="/tmp/${OUTPUT_NAME}-merged-$$"
mkdir -p "$OUTPUT_DIR"

echo "=== MLX LoRA → GGUF ==="
echo "Base model:  $BASE_MODEL"
echo "Adapter:     $ADAPTER_PATH"
OUTTYPE_UPPER=$(echo "$OUTTYPE" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
echo "Output type: $OUTTYPE"
echo "Output:      $OUTPUT_DIR/${OUTPUT_NAME}-${OUTTYPE_UPPER}.gguf"
if $COMPRESS; then
    echo "Compress:    TQ4_1S Config I → $OUTPUT_DIR/${OUTPUT_NAME}-ConfigI.gguf"
fi
echo ""

# Step 1: Merge LoRA deltas into base HF weights
echo "[1/3] Merging LoRA adapter into base model weights..."
export BASE_MODEL ADAPTER_PATH MERGED_PATH
python3 << 'PYEOF'
import json, os, shutil, gc
import mlx.core as mx

base_dir = os.environ["BASE_MODEL"]
adapter_path = os.environ["ADAPTER_PATH"]
out_dir = os.environ["MERGED_PATH"]

# Read LoRA scale from adapter config
with open(os.path.join(adapter_path, "adapter_config.json")) as f:
    adapter_cfg = json.load(f)
lora_scale = adapter_cfg.get("lora_parameters", {}).get("scale", 20.0)
print(f"  LoRA scale: {lora_scale}")

# Load adapter
adapter = dict(mx.load(os.path.join(adapter_path, "adapters.safetensors")))

# Build target map: stub → {lora_a, lora_b}
targets = {}
for k in adapter:
    if k.endswith(".lora_a"):
        stub = k[:-len(".lora_a")]
        targets[stub] = {
            "lora_a": adapter[stub + ".lora_a"],
            "lora_b": adapter[stub + ".lora_b"],
        }
print(f"  LoRA targets: {len(targets)} layers")

# Map adapter keys → HF base keys + transform type
hf_map = {}  # stub → (hf_key, type)
for stub in targets:
    hf_stub = stub.replace("language_model.model.", "model.language_model.")
    if "switch_mlp.gate_proj" in hf_stub:
        hf_key = hf_stub.replace("switch_mlp.gate_proj", "experts.gate_up_proj")
        hf_map[stub] = (hf_key, "expert_gate")
    elif "switch_mlp.up_proj" in hf_stub:
        hf_key = hf_stub.replace("switch_mlp.up_proj", "experts.gate_up_proj")
        hf_map[stub] = (hf_key, "expert_up")
    elif "switch_mlp.down_proj" in hf_stub:
        hf_key = hf_stub.replace("switch_mlp.down_proj", "experts.down_proj")
        hf_map[stub] = (hf_key, "expert_down")
    else:
        hf_key = hf_stub + ".weight"
        hf_map[stub] = (hf_key, "linear")

# Load base index and validate all keys exist
with open(os.path.join(base_dir, "model.safetensors.index.json")) as f:
    idx = json.load(f)

missing = []
for stub, (hf_key, _) in hf_map.items():
    if hf_key not in idx["weight_map"]:
        missing.append(hf_key)
if missing:
    print(f"  ERROR: {len(missing)} adapter keys not found in base model:")
    for k in missing[:5]:
        print(f"    {k}")
    raise SystemExit(1)

# Group operations by shard
shard_ops = {}
for stub, (hf_key, ttype) in hf_map.items():
    shard = idx["weight_map"][hf_key]
    shard_ops.setdefault(shard, []).append((hf_key, ttype, stub))

# Create output and copy metadata files
os.makedirs(out_dir, exist_ok=True)
for f in os.listdir(base_dir):
    if not f.endswith(".safetensors"):
        src = os.path.join(base_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, f))

# Process each shard
all_shards = sorted(f for f in os.listdir(base_dir) if f.endswith(".safetensors"))
total = len(all_shards)

for i, shard_name in enumerate(all_shards):
    src_path = os.path.join(base_dir, shard_name)
    dst_path = os.path.join(out_dir, shard_name)

    if shard_name not in shard_ops:
        shutil.copy2(src_path, dst_path)
        print(f"  [{i+1}/{total}] {shard_name}: copied (no changes)")
        continue

    ops = shard_ops[shard_name]
    print(f"  [{i+1}/{total}] {shard_name}: {len(ops)} modifications...", end=" ", flush=True)

    weights = dict(mx.load(src_path))
    expert_pairs = {}  # hf_key → {"gate": stub, "up": stub}

    for hf_key, ttype, stub in ops:
        la = targets[stub]["lora_a"]
        lb = targets[stub]["lora_b"]

        if ttype == "linear":
            # LoRALinear fuse formula: delta = (scale * lora_b.T) @ lora_a.T
            delta = (lora_scale * lb.T) @ la.T
            w = weights[hf_key]
            weights[hf_key] = w + delta.astype(w.dtype)

        elif ttype == "expert_gate":
            expert_pairs.setdefault(hf_key, {})["gate"] = stub
        elif ttype == "expert_up":
            expert_pairs.setdefault(hf_key, {})["up"] = stub

        elif ttype == "expert_down":
            # LoRASwitchLinear fuse formula: delta = scale * lora_b @ lora_a
            delta = lora_scale * lb @ la
            w = weights[hf_key]
            weights[hf_key] = w + delta.astype(w.dtype)

    # Process fused expert gate+up pairs
    for hf_key, pair in expert_pairs.items():
        w = weights[hf_key]  # shape: (num_experts, hidden*2, input)
        hidden_dim = w.shape[1] // 2
        w_gate = w[:, :hidden_dim, :]
        w_up = w[:, hidden_dim:, :]

        if "gate" in pair:
            la = targets[pair["gate"]]["lora_a"]
            lb = targets[pair["gate"]]["lora_b"]
            w_gate = w_gate + (lora_scale * lb @ la).astype(w.dtype)
        if "up" in pair:
            la = targets[pair["up"]]["lora_a"]
            lb = targets[pair["up"]]["lora_b"]
            w_up = w_up + (lora_scale * lb @ la).astype(w.dtype)

        weights[hf_key] = mx.concatenate([w_gate, w_up], axis=1)

    mx.eval(*weights.values())
    mx.save_safetensors(dst_path, weights)
    del weights
    mx.clear_cache()
    gc.collect()
    print("done")

print(f"  Merged model: {out_dir}")
PYEOF

# Step 2: Convert to GGUF
echo ""
echo "[2/3] Converting to GGUF (${OUTTYPE})..."

# Determine outtype flag for convert_hf_to_gguf.py
GGUF_OUTFILE="${OUTPUT_DIR}/${OUTPUT_NAME}-${OUTTYPE_UPPER}.gguf"

python3 "${LLAMA_CPP}/convert_hf_to_gguf.py" "$MERGED_PATH" \
    --outfile "$GGUF_OUTFILE" --outtype "$OUTTYPE"

# Step 3: Optional TQ4_1S compression
if $COMPRESS; then
    echo ""
    echo "[3/3] Compressing with TQ4_1S Config I..."

    CONFIG_I_FILE="/tmp/${OUTPUT_NAME}-config-i-$$.txt"
    COMPRESSED_GGUF="${OUTPUT_DIR}/${OUTPUT_NAME}-ConfigI.gguf"

    python3 << PYEOF
import json

with open("${BASE_MODEL}/config.json") as f:
    config = json.load(f)

tc = config.get("text_config", config)
layers = tc.get("layer_types", [])
n = len(layers)
boundary = 2  # protect first/last 2 layers

# Identify full attention layers (vs linear/mamba layers)
attn_layers = set()
for i, t in enumerate(layers):
    if t == "full_attention":
        attn_layers.add(i)

lines = []
for i in range(boundary, n - boundary):
    # Attention weights (only on full_attention layers)
    if i in attn_layers:
        for t in ["attn_q", "attn_k", "attn_v", "attn_output"]:
            lines.append(f"blk.{i}.{t}.weight=tq4_1s")
    # FFN expert weights (gate/up → TQ4_1S, down → Q4_K)
    for t in ["ffn_gate_exps", "ffn_up_exps"]:
        lines.append(f"blk.{i}.{t}.weight=tq4_1s")
    lines.append(f"blk.{i}.ffn_down_exps.weight=q4_k")

with open("${CONFIG_I_FILE}", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"  Config I: {len(lines)} tensor type overrides")
PYEOF

    "${LLAMA_CPP}/build/bin/llama-quantize" --allow-requantize \
        --tensor-type-file "$CONFIG_I_FILE" \
        "$GGUF_OUTFILE" "$COMPRESSED_GGUF" Q8_0

    rm -f "$CONFIG_I_FILE"
else
    echo ""
    echo "[3/3] Skipping compression (use --compress to enable)"
fi

# Cleanup
if ! $KEEP_MERGED; then
    rm -rf "$MERGED_PATH"
fi

echo ""
echo "=== Done ==="
ls -lh "${OUTPUT_DIR}/${OUTPUT_NAME}"*.gguf 2>/dev/null
echo ""
echo "Launch with:"
echo "  llama-server -m $GGUF_OUTFILE -ngl 99 -fa 1 --jinja --reasoning off -c 8192"
if $COMPRESS; then
    echo "  llama-server -m $COMPRESSED_GGUF -ngl 99 -fa 1 -ctk q8_0 -ctv turbo3 --jinja --reasoning off -c 8192"
fi
