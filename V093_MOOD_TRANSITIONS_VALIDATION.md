# VoxGen v0.7.56 — managed mood transitions

## Goal

Allow one VoxCPM2 synthesis request to evolve continuously from one managed delivery style/intensity to another without splicing or crossfading separately generated audio.

## Architecture

The transition is compiled in `src/prosody_control.rs` into one natural-language control instruction. Runtime synthesis still receives one `(control)text` prefix and generates one continuous acoustic sequence. There is no mood-transition PCM DSP, waveform interpolation, double synthesis, or reference-WAV crossfade.

Supported endpoint styles are Neutral, Warm, Cheerful, Excited, Sad, Concerned, Angry, Gentle, Serious, and Whisper-like. Endpoint intensity can independently be Subtle, Normal, or Strong. Modes are:

- `gradual`: the midpoint must sound like a believable intermediate state;
- `quick`: the change happens over a short natural span, then the destination style remains established.

Same-style transitions are allowed when intensity changes, e.g. Angry Normal → Angry Subtle. Identical style+intensity endpoints are rejected.

## Pair-specific motion

The compiler contains explicit cue-release/cue-introduction rules for common pairs including:

- Angry → Serious
- Concerned → Warm
- Sad → Warm
- Excited → Neutral
- Warm ↔ Serious
- Neutral ↔ Warm
- Neutral ↔ Sad
- Neutral ↔ Concerned

All other managed pairs use a generic continuity rule that preserves one speaker identity, one acoustic space, and continuous phrase timing.

## API

`POST /v1/audio/speech` accepts:

```json
{
  "input": "This was unacceptable, but now we need to decide what happens next.",
  "transition": {
    "from_style": "angry",
    "from_intensity": "normal",
    "to_style": "serious",
    "to_intensity": "normal",
    "mode": "gradual"
  }
}
```

`control` and `transition` are mutually exclusive. Transition control is also incompatible with Ultimate/prompt-continuation cloning. Explicit `cfg_value` remains authoritative. If CFG is omitted, VoxGen averages the two endpoint managed CFG deltas and caps the automatic transition delta at +0.20.

## Reference policy

A transition should normally use the Neutral reference WAV as its speaker anchor when a client owns a per-style reference bank. The engine exposes `recommended_transition_reference_style() == "neutral"`, but the HTTP server never invents or replaces the client's supplied reference path.

The wxDragon demo follows this recommendation automatically when its transition controls are enabled.

## Audio safety

v0.7.56 retains the complete v0.7.55 shared audio-safety path: floating-point headroom, normalized-correlation WSOLA with low-confidence fallback, amplitude-complementary overlap, and the shared output peak guard. Mood transitions do not add any new audio DSP stage.

## Validation

`validate_mood_transitions.py` checks the shared compiler, pair-specific rules, HTTP schema/one-pass handoff, CLI flags, demo controls/settings, Neutral-reference policy, explicit-CFG precedence, and documentation. The full current validator suite passes with the intentionally obsolete Iteration 5/6 validators excluded.
