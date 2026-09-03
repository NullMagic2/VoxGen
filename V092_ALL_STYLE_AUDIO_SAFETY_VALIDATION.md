# VoxGen v0.7.55 — all-style audio safety validation

This pass is deliberately **style agnostic**. Managed emotion prompts remain unchanged; instead, VoxGen hardens the common audio path used by Neutral, Warm, Cheerful, Excited, Sad, Concerned, Angry, Gentle, Serious, Whisper-like and Custom control.

## 1. Final peak protection replaces destructive full-scale clipping

Older HTTP serialization multiplied the rendered waveform by request gain and then hard-clamped each sample to `[-1, +1]`. If expressive synthesis or a user/style gain pushed peaks over full scale, the flattened waveform was already distorted before the client received it.

v0.7.55 keeps floating-point headroom through AudioVAE output, sinc pitch processing and WSOLA, then applies one engine-owned `OutputPeakGuard` immediately before output serialization. The sample ceiling is `0.98`.

- complete/non-streaming utterances use one uniform attenuation factor, preserving all internal dynamics exactly;
- streaming uses block look-ahead (the complete rendered block is scanned before it is emitted), instantaneous attenuation when required, and a 250 ms monotonic release;
- the guard never boosts audio;
- requested style/user gain remains the input to this single safety stage;
- the demo applies the same shared guard after its live speed/pitch processor, because local sinc/WSOLA can create small overshoots after server output has already been protected.

This is a peak-safety mechanism, not a loudness normalizer or compressor. Below the ceiling it is mathematically transparent.

## 2. WSOLA low-confidence fallback for breathy/unvoiced speech

WSOLA is based on waveform similarity. The existing normalized-correlation matcher works well for periodic/voiced speech, but breath/noise-dominated or weakly periodic spans can have no meaningful correlation maximum. Always selecting the largest weak score can jump between unrelated noise grains, creating phasey/warbling/metallic coloration—most audible on Whisper-like speech and fricatives.

v0.7.55 retains normalized correlation and the amplitude-complementary raised-cosine overlap, but adds a conservative confidence policy:

- if overlap RMS is extremely low, use the predicted analysis position;
- if the best normalized correlation is below `0.20`, use the predicted analysis position;
- only confident waveform matches are allowed to move the analysis segment away from its nominal trajectory.

The fallback changes synchronization only; it adds no EQ, denoising, excitation, spectral shaping or style-specific DSP.

## 3. Regression coverage

`validate_all_style_audio_safety.py` checks that:

- HTTP streaming and non-streaming output use the shared peak guard;
- CLI TTS output uses the same guard;
- the desktop demo protects the output of its live shared DSP;
- the old `(v * gain).clamp(-1, 1)` HTTP path is absent;
- the WSOLA low-confidence fallback is present;
- the unity-sum raised-cosine overlap remains present;
- all managed styles remain available;
- no managed demo style boost exceeds 1.05x;
- representative hot/boosted synthetic peaks remain at or below 0.98, while below-ceiling audio remains untouched.
