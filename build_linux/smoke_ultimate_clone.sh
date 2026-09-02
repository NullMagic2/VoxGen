#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
voxgen_require_bin
if [[ $# -lt 2 ]]; then echo "Usage: $0 expressive-reference.wav 'exact reference transcript'" >&2; exit 2; fi
voxgen_require_file "$VOXGEN_BASE_Q8"; voxgen_require_file "$VOXGEN_ACOUSTIC"; voxgen_require_file "$1"
"$VOXGEN_BIN" --base-lm "$VOXGEN_BASE_Q8" --acoustic "$VOXGEN_ACOUSTIC" --clone-mode ultimate --reference-wav "$1" --prompt-text "$2" --text "VoxGen Ultimate cloning test." --max-steps 12 --output-wav test_ultimate.wav
