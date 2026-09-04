# V094 — JSON macro recursion build validation

VoxGen v0.7.58 enlarged the `/health` capability response enough that `serde_json::json!` exceeded Rust's default macro recursion limit while compiling the `voxgen` binary.

V0.7.59 is a build-only fix. The crate roots for the binary, shared library, and wxDragon demo set `#![recursion_limit = "512"]`. This changes only compiler macro-expansion headroom; runtime synthesis, transitions, playback DSP, gain, CFG, reference selection, and API semantics are unchanged.

`validate_json_recursion_limit.py` verifies the crate-root attributes, package versions, and the continued presence of the large health capability object that originally triggered the regression.
