# Sample REFRACT reports

Real reports from the 2026-04-30 v0.3.0 matrix run. Each cell ships in
two formats:

- **`<name>.json`** — machine-readable and consumable by `refract compare`
  or any JSON-aware tool. The first three reports retain their original
  `refract.report.v0.2.0` schema; the catastrophic control is v0.3.1. They are
  historical artifacts, not regenerated data disguised as a newer schema.
- **`<name>.html`** — self-contained HTML page. Open in any browser
  (no required CDN or JS framework). Pasteable as a single file in Discord/X
  for visual sharing.

Use these to sanity-check your own first runs: if your composite is in
the same ballpark as the matching reference for a similar model size +
KV config, your framework is wired up correctly.

| File | Composite | Band | What it shows |
|------|-----------|------|---------------|
| `clean-q8q8-mistral24b` | ~91 | EXCELLENT | Mistral-Small-24B Q4_K_M with q8/turbo4 KV. Faithful quant on a robust model. Use as a "what success looks like" anchor. |
| `degraded-qwen7b` | ~76 | DEGRADED | Qwen2.5-7B Q8 with q8/turbo4 KV. Trajectory low (55) and PLAD low (76); KLD high (98). Per-token decode drift + brittleness, retrieval intact. |
| `distribution-broken-gemma26b` | ~29 | FAIL | gemma-4-26B-A4B Q8 with q8/turbo4 KV. Trajectory + KLD both ~17 (1.74 nats), R-NIAH 100, PLAD 78. The "distribution wrecked but reasoning intact" pattern. |
| `catastrophic-symturbo` | ~11 | FAIL | gemma-4-26B-A4B Q8 with **symmetric** turbo4/turbo4 (the deliberately-broken negative control). Trajectory 3.93, KLD 11.84 (2.13 nats). If you run this and don't get FAIL, your framework setup has a problem. |

## Sanity reading

```bash
# View a JSON
python3 -m json.tool examples/clean-q8q8-mistral24b.json | less

# Open an HTML report
open examples/clean-q8q8-mistral24b.html

# Side-by-side via the compare subcommand
python3 -m refract.cli compare examples/*.json
```

## Reproducing on your hardware

The v0.3.0+ matrix-runner script that produced these is reproducible
via `refract score` directly. The shapes you should see if running the
same model + same candidate:

  - Trajectory + KLD reproducible to ~1 point on the same llama.cpp commit
  - R-NIAH and PLAD reproducible to ~5 points (single-trial cells, some
    stochasticity in retrieval cell-by-cell)
  - Composite reproducible to ~2 points

If your numbers diverge by more than that, suspect:
  - Different llama.cpp commit (capture from your JSON's
    `environment.llama_cpp_commit`)
  - Different candidate KV config than you think
  - Different chat-template handling (v0.3+ vs older REFRACT)

The HTML reports embed the raw JSON in a `<details>` block at the
bottom — one artifact carries everything, no "did the JSON come from
the same run as the screenshot?" ambiguity.
