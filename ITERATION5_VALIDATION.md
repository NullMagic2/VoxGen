# VoxGen iteration 5 validation notes

## Upstream inference contract

The implementation targets OpenBMB's current `UnifiedCFM` inference behavior:

1. `z = randn * temperature`.
2. `linspace(1, 0, n_timesteps + 1)`.
3. Sway transform `t + sway * (cos(pi/2*t) - 1 + t)`.
4. `zero_init_steps = max(1, int(len(t_span) * 0.04))`.
5. CFG-Zero* initial steps use zero velocity.
6. Effective steps run the estimator with identical `x`, `cond`, and `t` for conditional/unconditional passes; only `mu` becomes zero in the unconditional pass.
7. VoxCPM2 `dit_mean_mode=false`, therefore estimator `dt` is zero.
8. CFG-Zero* computes `st* = dot(pos,neg)/(||neg||^2+1e-8)`.
9. Guided velocity is `neg*st* + cfg*(pos-neg*st*)`.
10. Euler update is `x = x - dt * velocity`.

The main VoxCPM2 loop passes concatenated projected BaseLM/ResidualLM state as `mu`, the previous latent patch as `cond`, and requests `n_timesteps` / `cfg_value` from the generation call.

## VoxGen GPU mapping

- Initial noise: `cfm_noise.comp`.
- Positive/unconditional LocDiT estimator calls: existing step-4 local transformer kernels.
- Positive/negative velocity snapshots: device-local buffers.
- `mu=0` unconditional branch: Vulkan buffer fill, no CPU upload/readback.
- optimized-scale reduction + CFG blend + Euler update: `cfm_cfg_euler.comp`.
- Effective Euler step: one Vulkan command submission.
- Final public CFM result: one 256-float readback.

## Explicit numerical parity mode

VoxGen's GPU RNG is deterministic but is not PyTorch RNG-compatible. For parity testing, supply `--cfm-initial-x-f32`, plus the same explicit `mu`, `cond`, steps, CFG and sway values to both implementations. This isolates solver/LocDiT numerical differences from random-number generation.

## Deliberate boundary

Iteration 5 produces VoxCPM2 acoustic latent patches. AudioVAE encoding/decoding and waveform output are not implemented here.

## Packaging-host limitation

The packaging host does not provide Cargo/Rust or glslc/glslangValidator. `validate_iteration5.py` therefore checks source/shader parity, deterministic input sizes, project isolation, solver formulas, and version/status contracts. Native compilation and numerical execution must be performed on the target Vulkan system.
