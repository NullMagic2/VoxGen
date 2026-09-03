# VoxGen v0.7.39 — clean-source scripts

v0.7.39 adds project-level source-cleaning entry points for Windows and Linux plus wrappers in `demo/`.

## Cleanup contract

The cleaner removes:

- engine Cargo intermediates under `target/`;
- demo Cargo intermediates under `demo/target/`;
- project-local `models/`, `downloads/`, and `.cache/` trees (including demo-local equivalents);
- generated `Cargo.lock` files for the two standalone crates;
- output files produced by the bundled smoke tests.

The cleaner preserves final engine/demo executables when they exist in debug or release target directories:

- `voxgen` / `voxgen.exe`;
- `voxgen-demo` / `voxgen-demo.exe`.

The scripts intentionally do **not** clean the global Cargo registry/cache or any model directory outside the VoxGen source tree. Windows reparse-point handling removes a project-local junction/symlink itself rather than recursing into its external target.

## Validation

`validate_clean_source.py` performs static contract checks for both platforms and runs the Linux cleaner against an isolated synthetic build/download tree. The test verifies that final binaries and checked-in test input fixtures survive while Cargo intermediates, local model/download/cache data, generated lockfiles, and smoke-test outputs are removed.
