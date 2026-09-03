# v0.7.54 Neutral prosody cleanup validation

## Goal

Neutral is a **natural conversational reference baseline**, not flat speech and not a mild emotional style. The managed compiler therefore preserves habitual speaker pitch centre/range, rate, loudness, lexical stress, syntax-driven intonation and small timing variation while suppressing only deliberately imposed affective coloration.

## Research basis

- Neutral material is characterized perceptually by the absence of one uniquely ascribed emotion; that does not imply absence of prosody.
- Emotional prosody is communicated through stress and pitch variation, so a neutral baseline must retain ordinary linguistic stress/intonation rather than remove it.
- Human-vs-synthetic naturalness work shows that reduced lexical-stress differences in pitch/duration are associated with reduced perceived naturalness.

## Engine policy

- Managed Neutral Subtle / Normal / Strong all compile to dedicated natural-neutral instructions.
- Normal Neutral explicitly preserves lexical stress, syntax-driven contours and subtle timing variation.
- Strong Neutral removes emotion-specific exaggeration but explicitly prohibits monotone, cold/robotic delivery and forced low pitch.
- Short `!` / `?` lines interpret punctuation linguistically, not as automatic emotional escalation.
- Neutral managed tuning remains exactly CFG +0.00 and demo gain 1.00x.
- Legacy v0.7.53 neutral recipe families remain recognized.
- Arbitrary custom controls remain untouched.

## Regression command

```text
python validate_neutral_prosody_cleanup.py
```
