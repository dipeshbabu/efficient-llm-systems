# Turbo4 & Related Next Steps

## Immediate: Fix turbo4 SET_ROWS (make it functional)

Priority 1: Split SET_ROWS kernel so turbo4 uses correct 3-bit packing
- This alone should bring PPL from 679 → ~6.3-6.4 range
- Sean's PR #4 partially did this but introduced turbo3 regression
- Need to do it without breaking turbo3

Priority 2: Evaluate if QJL actually helps
- scos-lab (issue #45) found QJL (Prod) is worse than MSE for attention
- QJL variance amplified by softmax → may not improve turbo4 over turbo3
- Test: turbo4 with QJL disabled (just 3-bit PolarQuant in 128-block format)
- If no improvement, turbo4 should be "turbo3 in 128-block format" not "turbo3 + QJL"

Priority 3: Fix QJL bugs (if QJL is worth keeping)
- Bug 1: Residual in wrong basis
- Bug 2: QJL matrix missing in dequant
- Bug 7: Wrong rotation matrices
- Only worth fixing if Priority 2 shows QJL helps

## Backlog: RotorQuant Review

johndpope posted RotorQuant — Clifford algebra variant claiming faster rotation via Cl(3,0) rotors.
- Repo: https://github.com/scrya-com/rotorquant (issue #30 on our repo)
- Key concern: rotation isn't our bottleneck (<1% of inference)
- No PPL numbers provided
- Worth reviewing the math to see if the Clifford approach has other advantages
- See also: [[RotorQuant - Clifford Algebra KV Cache Compression]] in Obsidian

## Backlog: Attention-Gated Optimizations

From docs/attention-gated-optimizations.md:
- #3 O rescaling skip — TODO
- #4 Softmax exp() skip — TODO (most promising remaining)

## Backlog: Community Issues

- sabotage3d CUDA bugs (llama-cpp-turboquant #14) — follow up by Apr 2
- Vulkan device support (turboquant_plus #46)
- M4 crash during KV cache init (turboquant_plus #35)
- scos-lab findings on K/V norm disparity (turboquant_plus #45) — worth investigating for asymmetric K/V
