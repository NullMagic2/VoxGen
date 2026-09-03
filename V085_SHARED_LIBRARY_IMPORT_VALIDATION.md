# VoxGen v0.7.47 — Shared Library Import Build Fix

This build fixes the binary/library crate boundary for the managed prosody compiler.

`src/runtime.rs` is compiled as part of the `voxgen` binary crate, while `prosody_control` is exported by the `voxgen` library crate in `src/lib.rs`. The correct import is therefore:

```rust
use voxgen::prosody_control::refine_control_instruction;
```

The invalid `crate::prosody_control::...` import has been removed. The unused `DSP_PULL_CHUNK` constant was also removed from `src/playback_dsp.rs`.

`validate_shared_library_imports.py` now guards both shared modules (`playback_dsp` and `prosody_control`) against this class of regression.
