#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

mode="${1:-release}"
probe=1
if [[ "${2:-}" == "--no-probe" ]]; then probe=0; fi

command -v cargo >/dev/null 2>&1 || {
  echo "ERROR: cargo not found. Install Rust 1.78+ from https://rustup.rs/" >&2
  exit 1
}

if [[ -n "${VOXGEN_GLSLC:-}" ]]; then
  [[ -x "$VOXGEN_GLSLC" ]] || { echo "ERROR: VOXGEN_GLSLC is not executable: $VOXGEN_GLSLC" >&2; exit 1; }
elif [[ -n "${VULKAN_SDK:-}" && -x "$VULKAN_SDK/bin/glslc" ]]; then
  export VOXGEN_GLSLC="$VULKAN_SDK/bin/glslc"
elif command -v glslc >/dev/null 2>&1; then
  export VOXGEN_GLSLC="$(command -v glslc)"
else
  echo "ERROR: glslc not found. Install the Vulkan SDK / glslc package, or set VOXGEN_GLSLC." >&2
  exit 1
fi

case "$mode" in
  release)
    echo "[VoxGen] Building release binary and Vulkan shaders..."
    cargo build --release
    built="$VOXGEN_ROOT/target/release/voxgen"
    ;;
  debug)
    echo "[VoxGen] Building debug binary and Vulkan shaders..."
    cargo build
    built="$VOXGEN_ROOT/target/debug/voxgen"
    ;;
  check)
    echo "[VoxGen] cargo check + Vulkan shader compilation"
    cargo check
    exit 0
    ;;
  clean)
    echo "[VoxGen] cargo clean"
    cargo clean
    exit 0
    ;;
  *)
    echo "Usage: ./build_voxgen.sh [release|debug|check|clean] [--no-probe]" >&2
    exit 2
    ;;
esac

echo
echo "[VoxGen] Build succeeded: $built"
if (( probe )); then
  echo "[VoxGen] Probing Vulkan devices..."
  "$built" --list-devices
fi
