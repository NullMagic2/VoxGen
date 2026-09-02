#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
voxgen_require_bin
voxgen_require_file "$VOXGEN_BASE_Q8"
"$VOXGEN_BIN" --base-lm "$VOXGEN_BASE_Q8" --base-format q8_0 --max-context 4096 --baselm-token 1 --top-k 8
