# v0.7.52 Excited / Gentle prosody validation

## Research basis

- Positive high-arousal speech is commonly characterized by higher and more variable F0, wider pitch range, and greater loudness variability. Excited speech should therefore use dynamic local pitch/intensity peaks rather than sustained loudness.
- Positive emotions are acoustically heterogeneous; positive surprise is not interchangeable with excitement/interest. The subtle Excited recipe no longer seeds the model with surprise.
- Vocal-effort studies show that minimal vocal effort reduces subglottal pressure, maximum flow declination rate and laryngeal resistance, while perceived effort tracks SPL and spectral/voice-quality measures. Gentle is therefore modeled primarily as low effort and low projection, not as Warm/Tender emotion.

## VoxGen policy

No EQ, pitch shifting, compression, formant processing, or hidden engine gain is added. Managed prompts steer VoxCPM2; demo-only gain multipliers remain visible request values. Explicit API gain and CFG remain authoritative.
