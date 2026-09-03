# V083 — Warmth and subtle-positive prosody research

VoxGen v0.7.48 distinguishes **warmth/tenderness** from **cheerfulness/joy** instead of treating warmth as neutral speech plus a smile.

Research basis used for the managed VoxCPM2 control recipes:

- Positive-emotion reviews find high-arousal joy/happiness tends toward higher and more variable pitch, higher loudness and faster rate, while lower-arousal positive states such as tenderness, pleasure and contentment use lower/moderate pitch, quieter delivery and slower rate.
- Recent acted-speech measurements likewise report tenderness with slower tempo, longer pauses, lower pitch, quieter volume and reduced high-frequency presence compared with high-activity happiness.
- Vocal-smile research shows a smile can raise formant frequencies / spectral energy, so a strong smile cue can drift toward cheerfulness. Warmth therefore uses only a faint/subtle smile cue and relies more on affiliative closeness, low vocal effort, smooth phrasing and mellow resonance.
- Synthetic-speech work on friendliness/likeability shows warmth is multidimensional; loudness, F0, spectral flux and formants all contribute, so one cue alone is insufficient.

Implementation rules:

1. Subtle warmth must remain perceptible rather than collapsing to neutral.
2. Warmth preserves the cloned speaker's habitual pitch centre and stays low-arousal.
3. Warmth uses slightly slower/softer delivery, smooth legato timing, softened attacks, gentle vowel lengthening and mellow/full resonance.
4. Cheerfulness remains distinct: audible smile, slight spectral brightness, and a small but definite increase in pitch variation/rhythmic buoyancy, without a global pitch or loudness increase.
5. No EQ, formant shifting, pitch shifting, compression or other post-generation effect is added; these are model-control instructions only.
