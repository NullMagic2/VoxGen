# VoxGen v0.7.40 native playback DSP validation

## Architecture

VoxGen now owns one speed/pitch implementation in `src/playback_dsp.rs`.

- Pitch transposition: 24-tap Lanczos-windowed sinc resampler with anti-alias cutoff for upward transposition.
- Tempo: VoxGen-owned `SpeechWsola` with normalized waveform correlation, 30 ms windows, 15 ms overlap, and 7.5 ms search half-range.
- Independent controls:
  - `pitch_factor = 2^(pitch_semitones / 12)`
  - sinc resampler factor = `pitch_factor`
  - WSOLA tempo = `(speed_percent / 100) / pitch_factor`
- Neutral `100% / 0 st` takes a dry bypass so ordinary VoxGen PCM is not colored by DSP.
- The processor is stateful and can accept live control changes. WSOLA history is reset when speed/pitch changes; sinc history is reset only when pitch changes.

The engine crate exports `StreamingPlaybackDsp` and `PlaybackControls`. Both the HTTP server and wxDragon demo consume this module. The demo no longer owns a second sinc or WSOLA implementation.

## HTTP API

`POST /v1/audio/speech` and `/v1/audio/speech/stream` accept:

```json
{
  "speed_percent": 100.0,
  "pitch_semitones": 0.0
}
```

Supported ranges are 50–200% and -12..+12 semitones. `/health` advertises `native_playback_dsp: true` and the ranges. Streaming and completed audio are processed before gain/PCM serialization.

## Dynamic Dictionary integration

Dynamic Dictionary v84 sends speed/pitch values with each VoxGen segment. Its former VoxGen `_StreamingSincPitch`, `_StreamingWsola`, and `_StreamingSpeedPitch` Python implementations are removed. Returned VoxGen streaming PCM is consumed directly after finite-value sanitization and PCM16 conversion. The completed-WAV VoxGen path likewise skips client-side speed/pitch processing.

Python speed/pitch helpers remain only under explicit Gemini-specific names because Gemini TTS is a separate backend and does not run through VoxGen.

Dynamic Dictionary requires `/health.native_playback_dsp == true` for the VoxGen backend. This prevents an older server from silently ignoring new request fields and producing divergent playback.

## Static validation performed

The package was validated with:

- `validate_native_playback_dsp.py`
- existing current VoxGen validators (historical iteration 5/6/7 snapshot validators excluded)
- Dynamic Dictionary `validate_voxgen_native_playback_dsp.py`
- Python `compileall` / AST parsing for Dynamic Dictionary

The duration identity was checked across representative speed/pitch extrema: sinc resampling by `p` followed by WSOLA tempo `s/p` yields final duration `1/s`, independent of pitch.

This build environment does not contain Cargo/Rust/Vulkan SDK, so native compilation was not run here. VoxGen keeps Rust 1.87 as its declared toolchain while the playback DSP itself no longer depends on the external `wsola` crate.
