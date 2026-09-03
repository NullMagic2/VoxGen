# v0.7.53 Serious / Sad prosody validation

## Research target

Sadness is one of the more consistently described emotional-prosody profiles: compared with neutral speech, studies and reviews commonly report lower mean F0, narrower F0 range, lower intensity and slower speaking rate. VoxGen therefore expresses Sad through relative changes to the cloned speaker rather than fixed Hz/dB targets, and guards against confusing low arousal with boredom, sleepiness, depression-like flattening, whispering, or high-arousal grief/wailing.

Seriousness is not a canonical basic emotion with one validated acoustic signature. VoxGen therefore treats Serious as a communicative stance. Research on confidence/certainty supports using deliberate timing, stable/firm delivery and resolved/falling intonation as useful cues, but does not justify forcing every speaker into a globally low pitch. Strong Serious therefore no longer begins from `grave, authoritative`; it targets committed, consequential delivery around the speaker's natural range.

## Policy

- Sad CFG delta: +0.10; demo gain 0.97x / 0.94x / 0.90x (Subtle / Normal / Strong).
- Serious CFG delta: +0.05 / +0.10 / +0.10; demo gain unchanged at 1.00x.
- Explicit API gain and CFG remain authoritative.
- No emotion-specific EQ, pitch shifter, compressor, formant transform, or post-generation DSP is added.

Run `python validate_serious_sad_prosody.py` plus the existing managed-prosody/DSP regression suite.
