# VoxGen v0.7.36 neutral voice-anchor validation

v0.7.36 makes an explicitly configured **Neutral** reference the canonical fallback identity for the bundled demo.

Resolution order is now:

1. usable reference for the selected style preset;
2. usable `emotion_reference.neutral` voice anchor;
3. legacy `voice_sample=` reference for older settings;
4. runtime default reference;
5. native zero-shot only when no reference/anchor has ever been configured.

If a configured Neutral/default WAV is missing or moved, synthesis is rejected with a visible diagnostic rather than silently switching to zero-shot and producing a different speaker identity. A missing style-specific clip falls back to the neutral anchor when one exists. The same resolver is used for Speak, A/B benchmark, XTX profiling, style-change prewarm, and model-load prewarm.

The generated acoustic model, CFG, temperature, CFM step count, streaming startup and v0.7.34 power-efficiency kernels are unchanged by this policy.

Static regression check:

```text
python validate_neutral_voice_anchor.py
```
