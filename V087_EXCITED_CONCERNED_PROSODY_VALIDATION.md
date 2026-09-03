# Excited / Concerned prosody validation — VoxGen v0.7.50

VoxGen v0.7.50 extends the managed prosody compiler with two research-guided profiles.

## Excited

The profile targets high-arousal positive speech with a moderately higher pitch centre, wider/dynamic pitch excursions, faster temporal transitions, crisp articulation, shorter pauses, and local emphasis peaks followed by release. Sustained loudness, shouting, squealing, and frantic delivery are explicitly rejected.

## Concerned

The profile deliberately avoids equating concern with panic or generalized anxiety. Concern-bearing phrases use mild vocal tension, responsive/local pitch elevation, focused articulation, and modest urgency; reassurance releases that tension with a pitch centre returning toward neutral, a slightly slower rate, softer attacks, lower energy, and smoother falling contours. Managed Concerned receives only a +0.10 CFG delta; explicit API CFG remains authoritative.

## Scientific basis

The implementation was informed by affective-prosody research showing that high-arousal emotions including excitement tend to have higher mean F0, and that emotion perception relies strongly on F0 together with intensity and temporal cues. Reviews of positive vocal emotion report that happy/high-arousal voices typically have higher pitch, greater pitch variability/range, and differences in loudness and rate. Anxiety/stress findings are less consistent, which argues against a rigid global "anxious" pitch/rate transform for Concerned. Empathy/caring studies instead show that slower rate, lower pitch, quieter/stabler energy, and smoother delivery can increase perceived caring and sympathy. The Concerned profile therefore uses a controlled alert-to-reassuring trajectory rather than a single global acoustic state.

Run `python validate_excited_concerned_prosody.py` plus the existing managed-prosody/DSP validators.
