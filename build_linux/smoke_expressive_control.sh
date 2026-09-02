#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
voxgen_require_bin
if [[ $# -lt 1 ]]; then echo "Usage: $0 speaker.wav" >&2; exit 2; fi
voxgen_require_file "$VOXGEN_BASE_Q8"; voxgen_require_file "$VOXGEN_ACOUSTIC"; voxgen_require_file "$1"
"$VOXGEN_BIN" --base-lm "$VOXGEN_BASE_Q8" --acoustic "$VOXGEN_ACOUSTIC" --clone-mode reference --reference-wav "$1" --control "warm and cheerful, conversational, with natural changes in emphasis" --text "VoxGen expressive speech control test." --max-steps 12 --output-wav test_expressive.wav
