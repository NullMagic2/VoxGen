# VoxGen anger / metallic-distortion research

Date: 2026-09-03

## Findings

### 1. Streaming AudioVAE context is the strongest implementation-level suspect

VoxGen v0.7.37 reconstructs each streamed 160 ms output patch by rerunning the AudioVAE decoder over a short rolling latent window. The VoxGen demo asked for four latent patches; the current Dynamic Dictionary integration asks for three.

The current AudioVAE V2 decoder is causal, but its newest output patch depends on 24 latent frames of history. VoxCPM2 uses four latent frames per acoustic patch, so a rolling compatibility implementation needs six patches to reproduce the newest patch without replacing required past context with left zero-padding.

The six-patch dependency follows directly from the released decoder topology:

- decoder rates `[8, 6, 5, 2, 2, 2]`;
- causal stem convolution, kernel 7;
- six transposed-convolution upsampling stages;
- three causal residual convolutions per stage, kernel 7, dilations `[1, 3, 9]`;
- causal final convolution, kernel 7.

A local synthetic-network parity check using the same causal topology produced the following result for the newest 7680 output samples:

- full decode vs 6-patch rolling decode: maximum difference about `1.2e-7` (floating-point noise);
- full decode vs 4-patch rolling decode: measurable waveform divergence.

Current upstream VoxCPM2 has moved to a stateful `StreamingVAEDecoder` that caches causal Conv1d and ConvTranspose1d state between chunks instead of redundantly decoding a short rolling window. That is the ideal long-term design for VoxGen as well.

### 2. The old 140% fresh-demo gain is unsafe for expressive peaks

VoxGen's AudioVAE ends in `tanh`, so raw samples are bounded. The v0.7.37 output path then multiplies those samples by requested gain and hard-clamps them to `[-1, +1]`. At the old 140% demo default, any raw peak above about 0.714 is flattened. Angry/high-arousal speech is especially likely to contain dense high peaks, so this can turn natural vocal roughness into obvious hard-clipping distortion.

The current Dynamic Dictionary integration is less exposed to this specific issue because it uses base gain 1.0 and an 85% angry gain by default, but the engine/demo should still not amplify every fresh request by 40%.

v0.7.38 therefore changes the fresh demo default to 100%. A future output stage should use true-peak-aware limiting if amplification above unity must remain available without hard clipping.

### 3. Anger should not be modeled primarily as loudness

Speech-emotion studies consistently describe anger as a combination of acoustic cues: pitch/F0 behavior, timing/speech rate, intensity, spectral balance/roughness, attacks, and voice quality. Raising neutral speech intensity by itself does not make it categorically angry, and reducing the level of angry speech does not remove all anger cues.

This suggests that prompts such as "forceful" can become counterproductive if the model interprets them as continuously loud or shouted. VoxGen v0.7.38 changes the built-in angry recipes to emphasize controlled tension, directness, phrase-level emphasis, and short bursts rather than sustained shouting.

### 4. CFG and diffusion steps are useful secondary tuning axes

OpenBMB's VoxCPM2 model card documents `cfg_value=2.0` and notes that higher guidance can improve prompt adherence but may be worse; it also notes that more inference timesteps can improve result quality at a speed cost. If the decoder/context fix is insufficient, anger should be A/B tested around CFG 1.6, 1.8, and 2.0 and 10 versus 14 diffusion steps using the same seed and reference.

## Changes in v0.7.38

1. Engine-enforced minimum rolling decode context: 6 patches.
2. HTTP streaming context default: 6 patches.
3. Demo streaming request: 6 patches.
4. Fresh demo gain: 100% instead of 140%.
5. Angry control recipes rewritten around controlled phrase-level expression rather than continuous loudness.
6. Added a static receptive-field validator (`validate_streaming_fidelity_v078.py`).

## Recommended A/B test

Use one short and one longer angry sentence with exactly the same reference and seed.

1. v0.7.37 non-streaming / complete WAV.
2. v0.7.37 streaming.
3. v0.7.38 streaming (6-patch minimum).
4. v0.7.38 streaming with gain 100% and the revised angry control.

If #1 is clean while #2 is metallic and #3 approaches #1, the rolling decoder was the dominant artifact source. If #1 itself is metallic, the next test should isolate the model conditioning/reference: neutral reference + angry instruction, then CFG/timestep sweeps.

Record peak level, RTF, first-audio latency, and whether the metallic artifact occurs in the middle of a chunk or specifically around 160 ms boundaries.

## Sources reviewed

- OpenBMB VoxCPM2 current synthesis source (`voxcpm2.py`), including stateful streaming decode.
- OpenBMB AudioVAE V2 source (`audio_vae_v2.py`), including `StreamingVAEDecoder` state handling and the decoder topology.
- OpenBMB VoxCPM2 model card / configuration guidance for CFG and inference timesteps.
- Chen et al., *The Contribution of Sound Intensity in Vocal Emotion Perception* (PLoS ONE, 2012).
- Nussbaum et al., *Electrophysiological Correlates of Vocal Emotional Processing* (Brain Sciences, 2023), for hot-anger acoustic characteristics.
- EBU Tech 3343 / R 128 guidance on controlling true peaks to avoid overload and distortion.
