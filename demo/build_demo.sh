#!/usr/bin/env bash
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[VoxGen Demo] Building wxDragon demo..."
cargo build --manifest-path "$DEMO_DIR/Cargo.toml" --release
echo "[VoxGen Demo] Built: $DEMO_DIR/target/release/voxgen-demo"
