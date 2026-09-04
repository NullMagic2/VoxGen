# VoxGen v0.7.58 — Extended Mood + Pace Transition Validation

## Scope

v0.7.58 extends the managed single-pass transition compiler introduced in v0.7.56. The transition remains part of VoxCPM2's natural-language prosody conditioning; VoxGen does not crossfade rendered WAVs and does not reconfigure WSOLA repeatedly inside a phrase.

## Dedicated narrative trajectories

The compiler now contains dedicated motion for these high-value transitions in addition to the earlier transition set:

- Neutral ↔ Serious
- Cheerful ↔ Excited
- Concerned ↔ Serious
- Warm ↔ Gentle
- Warm ↔ Sad
- Sad ↔ Serious
- Concerned ↔ Sad
- Excited → Serious
- Angry → Neutral
- Angry → Concerned
- Neutral ↔ Whisper-like
- Serious ↔ Whisper-like

Uncommon pairs continue to use the generic continuous transition compiler.

## Managed pace transitions

A structured mood transition may optionally include both `from_speed_percent` and `to_speed_percent`. These values describe the desired speaking-rate trajectory produced by VoxCPM2 itself.

Rules:

- valid endpoint range: 50–200%;
- a zero delta is normalized to no managed pace transition;
- a non-zero change smaller than 5 percentage points is rejected as perceptually trivial;
- a change larger than 45 percentage points is rejected as too large for one naturally continuous phrase;
- `quick` transitions are allowed, but the rate change must still occupy at least two stressed words or one short prosodic unit;
- instantaneous one-word or one-syllable rate jumps are forbidden;
- both speed endpoints must be provided together.

## Separation from playback DSP

Managed pace is not implemented through mid-utterance WSOLA changes. When a managed pace transition is active, uniform playback `speed_percent` must remain at 100%. The demo also locks its local live speed DSP to 100% for that utterance. Pitch remains independently adjustable.

This keeps one continuous WSOLA history and preserves the clipping/metallic-distortion protections introduced in v0.7.55.

## Validation

`validate_extended_mood_speed_transitions.py` verifies:

- all dedicated transition trajectories;
- HTTP and CLI structured speed endpoints;
- 5-point minimum meaningful delta;
- 45-point maximum single-phrase delta;
- anti-abrupt quick-transition wording;
- rejection of double tempo control;
- demo Start/End pace controls and persistence;
- demo local-speed lock during managed pace transitions.

The complete current validation set consists of 49 validators and passes in v0.7.58. Historical Iteration 5/6 validators are intentionally excluded because they assert superseded v0.5/v0.6 architecture expectations; current Iteration 7 validation passes.
