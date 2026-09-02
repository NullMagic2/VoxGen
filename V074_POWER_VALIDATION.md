# VoxGen v0.7.34 power-efficiency validation

This release is a source-level optimization pass on v0.7.33. It does not alter model weights, sampling settings, CFM step count, CFG math, playback DSP, GPU clocks, voltage, or power limits.

## Implemented

- One-dispatch Q/K/V projection in BaseLM and ResidualLM, with portable and RX 7900 XTX subgroup variants.
- One-dispatch sequence Q/K/V projection in LocEnc/LocDiT when cooperative matrices are not explicitly enabled.
- The explicit XTX cooperative-matrix Q/K/V path remains available and takes precedence when enabled.
- Buffer-scoped compute dependencies replace broad shader-memory barriers inside BaseLM, ResidualLM, LocEnc, and LocDiT transformer hot loops.
- Positive CFG mu vectors remain resident for an entire CFM solve; the negative pass logically inserts zero mu tokens in `pack_locdit` rather than clearing buffers.
- The old per-step mu save/restore copies and the buffer fills that required them are removed.
- 256-thread workgroup sizes are intentionally retained until target-hardware profiling demonstrates a no-regression alternative.

## Static validation

Run:

```text
python validate_power_efficiency_v074.py
```

The release also runs every current v0.7.x validator. Historical iteration-5/iteration-6 validators remain pinned to their original milestone contracts and are not release validators for v0.7.34.

The build environment used to package this archive does not contain Rust/Cargo or the Vulkan GLSL compiler, so final SPIR-V compilation and runtime benchmarking must be performed on a machine with the Vulkan SDK and Rust toolchain.
