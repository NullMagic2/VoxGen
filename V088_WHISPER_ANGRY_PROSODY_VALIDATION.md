# V088 — Whisper-like / Angry prosody validation

VoxGen v0.7.51 extends the managed prosody compiler to Whisper-like and controlled Angry.

## Scientific basis

### Whisper-like

A true whisper differs phonationally from ordinary quiet voiced speech: normal F0/periodic voice information is absent or greatly reduced, turbulent/noise-like airflow becomes a primary excitation source, whispered speech is typically softer, and vowel duration / spectral tilt / formant structure can differ from modal speech. Because fully removing periodic voicing can weaken intelligibility and cloned-speaker identity, VoxGen targets a **controlled near-whisper** rather than mechanically filtering the generated waveform.

Sources consulted:

- Smith et al./speech variability review discussion of whispered speech: absence of voice-frequency information, softer level, vowel-duration and spectral-tilt differences. https://pmc.ncbi.nlm.nih.gov/articles/PMC4959306/
- Kallail & Emanuel, whispered vs phonated vowel formants: whispered formants trend higher, especially F1. https://pubmed.ncbi.nlm.nih.gov/6738036/
- Perceptual/acoustical/aerodynamic study of whispering: airflow/pressure and articulation cues are important for whisper intelligibility. https://pubmed.ncbi.nlm.nih.gov/8064511/
- Traunmüller & Eriksson, vocal effort: whispering changes multiple acoustic dimensions beyond simple SPL reduction. https://pubmed.ncbi.nlm.nih.gov/10875388/

Implementation target:

- low vocal effort and minimal projection;
- audible airflow/noise;
- softened attacks;
- reduced periodic voicing rather than simple volume attenuation;
- slightly lengthened/careful articulation;
- preserve intelligibility and speaker identity;
- short exclamations must not project or revert to ordinary voiced speech.

Managed steering: +0.10 CFG when CFG is not explicitly supplied. Demo-only gain multiplier: 0.85x.

### Angry

General/hot anger is often associated with elevated F0, increased loudness/energy, faster rate and more high-frequency energy. However, **cold/suppressed anger** has been reported with moderate or low F0 mean and F0 range. VoxGen deliberately targets controlled/cold anger because sustained high loudness was both perceptually undesirable and previously exposed clipping/headroom problems.

Sources consulted:

- Schewski et al. 2025 systematic review: anger commonly associates with increased F0, volume/energy and speech rate, though some pitch findings differ across studies. https://pmc.ncbi.nlm.nih.gov/articles/PMC12289014/
- Paulmann et al. discussion of hot vs cold anger: hot anger tends toward high F0/intensity; cold anger toward moderate/low F0 mean and range. https://pmc.ncbi.nlm.nih.gov/articles/PMC7383972/
- Review of affective prosody: anger is associated with high-frequency spectral energy and changes in pitch/intensity/rate. https://pmc.ncbi.nlm.nih.gov/articles/PMC2831710/
- Affective prosody review/Banse & Scherer synthesis: hot anger is high/bright and acoustically distinct; cold anger is a separate lower-salience category. https://pmc.ncbi.nlm.nih.gov/articles/PMC12231869/

Implementation target for VoxGen's controlled Angry:

- pitch centre near neutral or slightly lower;
- compact purposeful pitch range rather than globally high pitch;
- firm vocal tension and hard clean attacks;
- quicker compact timing and shorter pauses;
- brief local emphasis/energy peaks followed by release;
- tight/bright voice quality cue;
- moderate sustained loudness; punctuation sharpens timing/attack rather than loudness;
- no scream/growl/rasp requirement and no automatic CFG escalation.

Managed steering: +0.00 CFG. Demo-only gain multiplier: 0.90x.

## Validation

Run:

```text
python validate_whisper_angry_prosody.py
```

The validator checks that Whisper-like is not implemented as merely quiet modal speech, that controlled Angry uses cold-anger timing/tension cues rather than sustained loudness, that the short-line anti-projection/anti-shout guards are present, and that health metadata reports the intended managed-prosody v5 policy.
