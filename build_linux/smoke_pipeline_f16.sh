#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
voxgen_require_bin
voxgen_require_file "$VOXGEN_BASE_F16"; voxgen_require_file "$VOXGEN_ACOUSTIC"
[[ -f test_current_embed.f32 ]] || "$VOXGEN_PYTHON" make_test_embeddings.py
"$VOXGEN_BIN" --base-lm "$VOXGEN_BASE_F16" --acoustic "$VOXGEN_ACOUSTIC" --base-format f16 --max-context 256 --base-residual-embedding-f32 test_current_embed.f32
