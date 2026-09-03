# VoxGen v0.7.45 CLI playback flags

VoxGen now exposes the native playback DSP with the concise command-line flags:

- `--speed <percent>` — 50 through 200, default 100.
- `--pitch <semitones>` — -12 through +12, default 0.

The previous automatically derived Clap names `--speed-percent` and `--pitch-semitones` are intentionally not aliases and are not accepted by the CLI. Internal Rust fields and HTTP JSON fields keep their explicit unit-bearing names (`speed_percent`, `pitch_semitones`).

Validation: `python validate_cli_playback_flags.py`.
