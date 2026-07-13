# NIAH: Boundary V (LA-V7) Retrieval Test

**Date:** 2026-03-29
**Model:** Qwen2.5-7B-Instruct-Q4_K_M
**Hardware:** M5 Max
**Config:** TURBO_LAYER_ADAPTIVE=7, -ctk q8_0 -ctv turbo3

## Prompt

```
Facts: The capital of Australia is Canberra. The secret password is turbo-rainbow-42. Mount Everest is 8849m. Gold is Au. What is the secret password mentioned above?
```

## Result

```
The secret password mentioned above is "turbo-rainbow-42."
```

**PASS** — correct retrieval.

## Context

Boundary V (LA-V7) uses q8_0-V for first 2 + last 2 layers, turbo2-V for remaining 24 layers. This test confirms retrieval works correctly with the mixed V precision policy.
