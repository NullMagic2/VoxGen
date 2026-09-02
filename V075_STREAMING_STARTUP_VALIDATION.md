# VoxGen v0.7.35 streaming-startup validation

This release targets latency from **Speak** to the first audible PCM without changing CFM quality settings.

Validated source invariants:

- the local desktop demo sends stable `reference_audio_path`/`prompt_audio_path` values instead of per-request base64 copies;
- Runtime caches AudioVAE conditioning patches using canonical path + file size + modification timestamp + pad side;
- `/v1/audio/conditioning/warm` pre-encodes the selected reference outside the speech request; startup warms the active preset reference, and preset/reference changes refresh it in the background;
- startup prebuffer is one acoustic patch at 100%/slower playback, with additional reserve retained for faster playback;
- streaming PCM publication precedes stop-predictor synchronization for the already-generated current patch;
- XTX live text-prefix command batching is 32 positions while offline timestamp profiling remains 16;
- `/health` exposes the engine version and the demo restarts a stale older listener so the low-latency server path cannot be silently bypassed;
- v0.7.34 QKV, targeted barriers, CFG-mu persistence and shared power-efficiency paths remain present.

Runtime build/benchmark must still be performed on the target Windows/RX 7900 XTX machine because the artifact environment does not include Cargo/rustc/glslc.
