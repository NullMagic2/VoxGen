#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
voxgen_require_bin
voxgen_require_file "$VOXGEN_BASE_Q8"; voxgen_require_file "$VOXGEN_ACOUSTIC"
"$VOXGEN_BIN" --base-lm "$VOXGEN_BASE_Q8" --acoustic "$VOXGEN_ACOUSTIC" --max-context 256 --base-residual-text-prefill 1,2,3,4
