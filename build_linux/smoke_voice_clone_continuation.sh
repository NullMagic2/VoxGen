#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
voxgen_require_bin
if [[ $# -lt 2 ]]; then echo "Usage: $0 prompt.wav 'Exact prompt transcript. '" >&2; exit 2; fi
voxgen_require_file "$VOXGEN_BASE_Q8"; voxgen_require_file "$VOXGEN_ACOUSTIC"; voxgen_require_file "$1"
"$VOXGEN_BIN" --base-lm "$VOXGEN_BASE_Q8" --acoustic "$VOXGEN_ACOUSTIC" --prompt-wav "$1" --prompt-text "$2" --text "This is the continuation." --max-steps 12 --output-wav test_continuation.wav
