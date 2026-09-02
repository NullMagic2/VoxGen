#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
voxgen_require_bin
voxgen_require_file "$VOXGEN_BASE_Q8"; voxgen_require_file "$VOXGEN_ACOUSTIC"
[[ -f test_base_hidden.f32 ]] || "$VOXGEN_PYTHON" make_test_embeddings.py
"$VOXGEN_BIN" --base-lm "$VOXGEN_BASE_Q8" --acoustic "$VOXGEN_ACOUSTIC" --base-format q8_0 --max-context 256 --fsq-input-f32 test_base_hidden.f32
