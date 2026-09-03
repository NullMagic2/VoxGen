# VoxGen v0.7.45 — Warm/Cheerful Managed Prosody

## Problem

The metallic streaming artifact had been removed, but low-intensity positive
styles still failed perceptually: **subtle cheerful** could remain neutral or
turn into an announcer-like burst, while **warm** often sounded merely serious
or generic.

The old recipes used broad semantic labels followed by the same generic suffix:

> with natural phrase-level variation in emphasis and emotion rather than a fixed tone

That phrase does not identify the acoustic cues that carry positive valence, and
on a very short line it can invite an unnecessarily large expressive excursion.

## Research basis

- Happy/joyful speech is commonly associated with higher and more variable F0,
  brighter/high-frequency energy, faster articulation and stronger energy than
  neutral speech. These cues overlap substantially with anger, so using all of
  them strongly is unsuitable for *subtle* cheerfulness.
- Smiling changes the vocal tract and is audible in speech. Moderate smiling is
  associated with positive interpretations and warmth; excessive smiling has
  diminishing perceptual benefit.
- Positive-emotion intensity is graded rather than categorical. Listeners can
  distinguish multiple levels of happiness in short utterances, with pitch,
  loudness and rate among the important cues.
- Empathic/warm speech is associated with steadier tone and volume; excessive
  pitch and energy can reduce perceived empathy.

Key sources:

- Yildirim et al., *An acoustic study of emotions expressed in speech*,
  Interspeech 2004.
- Dimosa, Dick & Dellwo, *Perception of levels of emotion in speech prosody*,
  ICPhS 2015.
- Aubergé et al., *The Prosody of Smile*, SpeechEmotion 2000.
- Pearsell & Pape, *The effects of different voice qualities on the perceived
  personality of a speaker*, Frontiers in Communication, 2023.
- OpenBMB VoxCPM2 documentation on natural-language control instructions.

## Implementation

VoxGen now owns a `prosody_control` module.

1. The desktop demo consumes the engine's shared recipe builder rather than
   maintaining a second style table.
2. The runtime recognizes only the exact VoxGen/Dynamic Dictionary managed
   warm and cheerful recipe families. Arbitrary CLI/API controls are preserved.
3. Managed recipes are compiled before tokenization into concise acoustic goals.
4. Subtle cheerful requests:
   - a light audible smile;
   - slightly brighter resonance than neutral;
   - small buoyant pitch lifts;
   - gently lively rhythm;
   - steady conversational loudness.
5. Subtle warm requests:
   - a soft audible smile;
   - smooth connected phrasing;
   - relaxed articulation and soft consonant attacks;
   - steady moderate-low loudness;
   - small welcoming pitch lifts.
6. Very short lines receive a guard against building to a climax. An exclamation
   mark is explicitly treated as friendly brightness rather than extra loudness.

No playback DSP, reference normalization, punctuation rewriting, or generated
waveform filtering was added. The change acts only on managed style control text
before VoxCPM2 tokenization.
