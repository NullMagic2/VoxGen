#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

say() {
  printf '[VoxGen clean] %s\n' "$*"
}

clean_target_keep_binary() {
  local target="$1"
  shift
  local keep_names=("$@")

  [[ -d "$target" ]] || return 0

  say "Cleaning ${target#$ROOT/}/ while preserving final binaries..."

  # Remove every regular file/symlink except the explicitly preserved final
  # executables. Cargo dependency artifacts, shader outputs, incremental state,
  # metadata, PDBs, rlibs, etc. are intentionally removed.
  while IFS= read -r -d '' path; do
    local rel="${path#$target/}"
    local keep=0
    local wanted
    for wanted in "${keep_names[@]}"; do
      if [[ "$rel" == "$wanted" ]]; then
        keep=1
        break
      fi
    done
    if (( ! keep )); then
      rm -f -- "$path"
    fi
  done < <(find "$target" -mindepth 1 \( -type f -o -type l \) -print0)

  # Remove now-empty directories from the bottom up. Directories containing a
  # preserved executable remain automatically.
  find "$target" -depth -mindepth 1 -type d -empty -delete
}

remove_tree() {
  local path="$1"
  if [[ -e "$path" || -L "$path" ]]; then
    say "Removing ${path#$ROOT/}"
    rm -rf -- "$path"
  fi
}

remove_file() {
  local path="$1"
  if [[ -f "$path" || -L "$path" ]]; then
    say "Removing ${path#$ROOT/}"
    rm -f -- "$path"
  fi
}

say "Project root: $ROOT"

# Engine build tree. Keep only the final debug/release executable, whether the
# tree was produced on Linux or Windows/WSL.
clean_target_keep_binary \
  "$ROOT/target" \
  "release/voxgen" \
  "debug/voxgen" \
  "release/voxgen.exe" \
  "debug/voxgen.exe"

# Demo build tree.
clean_target_keep_binary \
  "$ROOT/demo/target" \
  "release/voxgen-demo" \
  "debug/voxgen-demo" \
  "release/voxgen-demo.exe" \
  "debug/voxgen-demo.exe"

# Project-local downloads only. Never touch ~/.cargo, ~/.cache, or model paths
# outside the VoxGen source tree.
remove_tree "$ROOT/models"
remove_tree "$ROOT/downloads"
remove_tree "$ROOT/.cache"
remove_tree "$ROOT/demo/models"
remove_tree "$ROOT/demo/downloads"
remove_tree "$ROOT/demo/.cache"

# Cargo creates these for the two standalone binary crates when they are not
# already part of the source package.
remove_file "$ROOT/Cargo.lock"
remove_file "$ROOT/demo/Cargo.lock"

# Outputs produced by the bundled smoke/validation commands. The checked-in
# deterministic test input fixtures are intentionally retained.
for generated in \
  test_cfm_output.f32 \
  test_conditioned_cfm_output.f32 \
  test_clone.wav \
  test_continuation.wav \
  test_expressive.wav \
  test_tts.wav \
  test_tts_stream.wav \
  test_ultimate.wav \
  test_vae_decode.wav \
  test_vae_decode_pcm.f32 \
  test_vae_encoded.f32 \
  test_vae_pcm_encoded.f32 \
  test_vae_roundtrip.f32 \
  test_vae_roundtrip.wav; do
  remove_file "$ROOT/$generated"
done

say "Done. Source/build/download artifacts were cleaned; final VoxGen and demo binaries were preserved when present."
