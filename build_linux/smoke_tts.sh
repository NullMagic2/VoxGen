#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
voxgen_require_bin
voxgen_require_file "$VOXGEN_BASE_Q8"; voxgen_require_file "$VOXGEN_ACOUSTIC"
"$VOXGEN_BIN" --base-lm "$VOXGEN_BASE_Q8" --acoustic "$VOXGEN_ACOUSTIC" --text "VoxGen end to end speech test." --max-steps 12 --output-wav test_tts.wav
