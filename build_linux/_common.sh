#!/usr/bin/env bash
# Shared path/model setup for VoxGen Linux scripts.
set -euo pipefail

VOXGEN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VOXGEN_ROOT"

: "${VOXGEN_MODEL_DIR:=$VOXGEN_ROOT/models}"
: "${VOXGEN_BASE_Q8:=$VOXGEN_MODEL_DIR/VoxCPM2-BaseLM-Q8_0.gguf}"
: "${VOXGEN_BASE_F16:=$VOXGEN_MODEL_DIR/VoxCPM2-BaseLM-F16.gguf}"
: "${VOXGEN_ACOUSTIC:=$VOXGEN_MODEL_DIR/VoxCPM2-Acoustic-F16.gguf}"
: "${VOXGEN_BIN:=$VOXGEN_ROOT/target/release/voxgen}"
: "${VOXGEN_PYTHON:=python3}"
export VOXGEN_ROOT VOXGEN_MODEL_DIR VOXGEN_BASE_Q8 VOXGEN_BASE_F16 VOXGEN_ACOUSTIC VOXGEN_BIN VOXGEN_PYTHON

voxgen_require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: required file not found: $1" >&2
    return 1
  fi
}

voxgen_require_bin() {
  if [[ ! -x "$VOXGEN_BIN" ]]; then
    echo "ERROR: VoxGen binary not found: $VOXGEN_BIN" >&2
    echo "Run ./build_voxgen.sh first." >&2
    return 1
  fi
}
