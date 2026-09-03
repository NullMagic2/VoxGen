# VoxGen v0.7.47 — WSOLA Headroom / Angry Distortion Fix

The speech WSOLA overlap now uses an amplitude-complementary raised-cosine crossfade instead of an equal-power cos/sin crossfade. WSOLA aligns highly correlated portions of the same speech signal, so equal-power gains could sum to about 1.414 at overlap midpoint and add roughly +3 dB before requested output gain.

Playback DSP no longer clamps internally. Floating-point headroom is preserved through resampling/WSOLA and requested gain is applied before the final WAV/PCM serialization clamp. This prevents a later attenuation such as `gain=0.85` from merely making an already-clipped waveform quieter.

`validate_wsola_headroom.py` checks the source invariants and mathematically verifies unity amplitude for a perfectly correlated overlap.
