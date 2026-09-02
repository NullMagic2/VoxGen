#!/usr/bin/env bash
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VOXGEN_ROOT="${VOXGEN_ROOT:-$(cd "$DEMO_DIR/.." && pwd)}"
if [[ ! -x "$DEMO_DIR/target/release/voxgen-demo" ]]; then
  "$DEMO_DIR/build_demo.sh"
fi
cd "$VOXGEN_ROOT"
exec "$DEMO_DIR/target/release/voxgen-demo"
