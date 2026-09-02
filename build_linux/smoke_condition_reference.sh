#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
voxgen_require_bin
voxgen_require_file "$VOXGEN_BASE_Q8"; voxgen_require_file "$VOXGEN_ACOUSTIC"
[[ -f test_reference_latents.f32 ]] || "$VOXGEN_PYTHON" make_test_local_inputs.py
# Token IDs 1,2,3 are deterministic plumbing inputs, not meaningful speech text.
"$VOXGEN_BIN" --base-lm "$VOXGEN_BASE_Q8" --acoustic "$VOXGEN_ACOUSTIC" --base-format q8_0 --max-context 256 --conditioning-text-tokens 1,2,3 --reference-latents-f32 test_reference_latents.f32
