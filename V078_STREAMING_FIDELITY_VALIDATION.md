# VoxGen v0.7.38 — streaming fidelity / anger artifact mitigation

## Problem

The compatibility streaming AudioVAE decoder reconstructed each emitted 160 ms patch from a short rolling latent window. The demo requested four patches and the current Dynamic Dictionary integration requested three. The decoder topology itself has a 24-latent-frame causal receptive field for the newest output patch, equal to six 4-frame latent patches. With only three or four patches, part of the required past is silently replaced by left zero-padding on every decode. High-arousal speech makes those errors especially audible as metallic, brittle, or cracking timbre.

## v0.7.38 changes

- Enforce a minimum rolling AudioVAE decode context of **6 patches**; larger caller requests remain allowed.
- Change the HTTP default and VoxGen demo request to 6 patches.
- Change the fresh demo gain default from **140% to 100%** so expressive peaks are not amplified into the existing hard clamp by default.
- Rewrite the built-in angry style recipes to express anger through tension, timing, and phrase-level emphasis rather than continuous loudness/shouting.

## Why six patches

The current AudioVAE decoder uses a causal stem convolution, six ConvTranspose stages with rates `[8, 6, 5, 2, 2, 2]`, three causal residual k7 convolutions at dilations `[1, 3, 9]` in every stage, and a causal final k7 convolution. Back-propagating the dependency interval of the newest 7680-sample output patch reaches 24 latent frames into the decoder input. With four latent frames per acoustic patch, this is exactly six patches.

Run:

```text
python validate_streaming_fidelity_v078.py
python validate_gain_setting.py
```

## Follow-up

This is an interim compatibility fix. The ideal long-term implementation is a stateful AudioVAE streaming decoder that caches causal convolution state across chunks instead of repeatedly decoding a rolling window.
