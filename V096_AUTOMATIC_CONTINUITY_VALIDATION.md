# VoxGen v0.7.60 Automatic Continuity Validation

This release intentionally removes the client-facing explicit transition API introduced in earlier releases. The public request model is destination-only: `style`, `intensity`, `pace_percent`, `continuity_id`, and `boundary`. Unknown fields are rejected by `SpeechRequest`.

## State guarantees

- 30-minute continuity-session expiry.
- Maximum 256 active continuity IDs.
- State commits only after successful synthesis.
- Cancellation or synthesis failure does not advance state.
- Per-ID reset endpoint: `POST /v1/audio/continuity/reset`.
- Model load, unload, and pipeline reset clear all continuity state.
- Global/per-ID generation epochs prevent an in-flight request from recreating reset state.
- Continuity state is scoped to the speaker-conditioning source.
- Response headers report previous/effective style, intensity, and pace.

## Pace policy

- Changes under 5 percentage points are suppressed.
- Meaningful changes are realized by single-pass VoxCPM2 prosody conditioning.
- A request more than 45 points away advances at most 45 points per completed phrase.
- `speed_percent` must stay at 100 while `continuity_id` is active; managed pace uses `pace_percent`.

## Demo policy

The demo exposes Style, Intensity, Managed pace %, and Continuous/Hard cut. Managed style requests keep a stable Neutral reference anchor. Default CFG 2.0 is omitted so VoxGen can choose endpoint-aware guidance; non-default CFG values remain explicit overrides.

## Regression validation

`validate_automatic_continuity_v0760.py` checks state safety and diagnostics. `validate_breaking_transition_api_v0760.py` checks that the old transition object, endpoint fields, and CLI/demo surface are absent while the pair compiler remains crate-private. Historical v0.5.0/v0.6.0 milestone validators explicitly skip on this release.

## Packaging validation status

The packaged source passed all 54 Python regression validators (with the historical v0.5.0 and v0.6.0 milestone validators exiting successfully as explicit non-applicable skips), TOML parsing, shell syntax checks, and a Rust source lexical integrity audit. A Rust toolchain (`cargo`/`rustc`/`rustfmt`) is not installed in the packaging environment, so this archive does not claim an in-environment Cargo compilation.
