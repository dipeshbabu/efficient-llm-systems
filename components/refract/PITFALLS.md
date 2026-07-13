# REFRACT — known pitfalls

Things that have actually bitten us, in priority order. Keep this
discoverable so first-time users don't repeat the discovery.

## Setup

### CUDA OOM on consumer GPUs running large MoE models

REFRACT defaults to `-ngl 99` (all layers on GPU). Consumer cards (12
GB 3060, 16 GB 4060 Ti, etc.) running large MoE models like
Qwen3.6-35B-A3B will OOM out of the box. The fix is to pass extra
llama.cpp flags via `REFRACT_LLAMA_EXTRA_FLAGS`:

```bash
# 12 GB GPU + Qwen3.6-35B-A3B
export REFRACT_LLAMA_EXTRA_FLAGS="-ngl 28 -ncmoe 32"
python3 -m refract.cli score --backend llamacpp ...
```

The env var is appended to every `llama-cli` / `llama-completion` /
`llama-perplexity` invocation. llama.cpp uses last-wins on repeated
flags, so `-ngl 28` in `REFRACT_LLAMA_EXTRA_FLAGS` overrides the
default `-ngl 99`.

Reported by AJ on a 3060 (2026-05-02). Added in v0.3.2.1.

### Wrong default `--axis-a` in v0.1.x / v0.2.x

The `gtm` axis used `llama-tokenize` to retokenize the candidate's
decoded text and compare it against the reference's retokenized text.
Detokenize→retokenize is lossy: a 50-token greedy generation can come
back as 60–137 tokens after retokenization (gemma-4 31B v0.1.2 hit a
2.87× inflation). v0.1.4 added the `trajectory` axis (decode-time
token IDs via patched llama-completion); **v0.3.1 made `trajectory`
the default**. If you're seeing weird Axis A scores, confirm
`--axis-a trajectory` is in your invocation.

### `head_dim % block_size != 0` blocks q4/q8 KV

phi-2 and a few other models have `head_dim = 80` which doesn't divide
the q4_0 / q8_0 cache block size of 32. llama.cpp will fail to create
the context with `K cache type q4_0 with block size 32 does not divide
n_embd_head_k=80`. Use a different model OR pick a KV type compatible
with the head dim (e.g., q4_K with block 256 if your build has it).

### llama.cpp lacking turbo support

If you're testing TurboQuant configs (`ctv=turbo4`), your llama.cpp
build needs the turbo branches compiled in. Check with:

```
llama-cli --help | grep -A2 'cache-type-k'
# look for 'turbo2, turbo3, turbo4' in the allowed values
```

If absent, build from dipeshbabu/llama.cpp `feature/turboquant-kv-cache`
or set `--candidate "ctk=q8_0,ctv=q8_0"` (standard quant) instead.

### MLX RotatingKVCache + quantization

Gemma family + a few other sliding-window-attention models use
`RotatingKVCache` in mlx-lm. As of mlx-lm 0.31.2, RotatingKVCache
quantization is NotImplementedYet. Symptom:

```
NotImplementedError: RotatingKVCache Quantization NYI
```

If you hit this on MLX, either:
  - Switch to `--backend llamacpp` for that model, OR
  - Test with `ctk=f16,ctv=f16` only on the MLX side

## Result interpretation traps

### R-NIAH = 100 with `confidence: low`

Means base_acc averaged across cells is below 0.2 — the model isn't
engaging retrieval at all under either config. R-NIAH = 100 is then a
noise-floor reading rather than real signal. Look at the
`confidence` field in the JSON before posting the score.

Common causes (today, all v0.3+ should be fixed):
  - Older versions used "secret password" needle wording → triggered
    safety refusals on instruct models. v0.2.1 changed to neutral
    "rare paint color" framing.
  - Raw "Q: A:" prompt format didn't engage instruct models → they
    continued the wikitext stylistically. v0.3.0 applies chat
    templates via `--jinja`.
  - `n_predict = 32` with thinking-mode models → answer truncated
    inside the thinking trace. v0.2.1 bumped default to 256.

### PLAD `paraphrase = NaN`

The built-in synonym table is small. If your prompts have no words in
the table, paraphrase never fires and the per-perturbation slot is
NaN. The cell is listed in `skipped_perturbations` (v0.3.1+) and the
overall PLAD score still uses the perturbations that DID fire. Don't
read NaN as FAIL — read it as "didn't apply".

### Trajectory + KLD low, R-NIAH + PLAD high

Means per-token distribution drift but high-level reasoning intact.
Common for aggressive turbo configs on certain models. The candidate
generates different text but still retrieves facts and resists
perturbations. **Useful triage signal.** If your workload doesn't
require text-level reproducibility (just task success), this may be
safe. Otherwise revert to a less aggressive quant.

### KLD low but Trajectory high

Means distributions look fine on the corpus but the model decodes
different text. Two common explanations:
  1. The corpus distribution doesn't match the prompts the trajectory
     axis is using. Try a corpus that better reflects your workload.
  2. There's a chat-template engagement difference between the two
     code paths (KLD is corpus-driven and never touches chat
     templates; Trajectory does).

### Composite shifts across versions

REFRACT v0.2.0 → v0.3.0 chat-template fix produced ±2 composite
deltas on the same models / candidates. The number isn't a bug — the
new methodology is correct, but cross-version comparisons require
same version on both sides. Always include `framework_version` from
the JSON when sharing scores.

## Discovered during the v0.2/v0.3 matrix runs (2026-04-30)

  - **"secret password" needle = safety refusal** on RLHF'd instruct
    models. Fixed in v0.2.1 with neutral "rare paint color" framing.
  - **Thinking-mode models burn n_predict** before answering. v0.2.1
    bumped R-NIAH n_predict to 256. v0.3.0 added auto-detect at
    startup + `-rea off` on llama-cli.
  - **Symmetric turbo (`ctk=turbo4,ctv=turbo4`) is empirically
    worse** than asymmetric on every distribution-level surface.
    This is the framework's negative-control config — if REFRACT
    doesn't FAIL it, the framework has problems. v0.2.0 result on
    gemma-4-26B-A4B: composite 19 FAIL, KLD 11.84 (2.13 nats),
    Trajectory 9.74. Use this as a sanity check.
  - **R-NIAH substring matching is coarse** on broken-distribution
    models — survives drift if the keyword shows up anywhere in the
    response (including thinking traces echoing the prompt).
  - **`--measure-floor` is highly recommended** for first runs — it
    verifies the reference itself is deterministic on your build.
    Without it, KLD deltas are theoretical.
