# Managed style strength validation — VoxGen v0.7.49

## Goal
Make low-arousal managed styles audible without turning them theatrical or introducing another DSP/effect layer.

## Changes
- Managed **Warm**: base CFG +0.20 (capped at 3.0).
- Managed **Gentle**: base CFG +0.15 (capped at 3.0).
- Managed **Subtle Cheerful**: base CFG +0.10 (capped at 3.0).
- HTTP applies the managed delta only when `cfg_value` is omitted. Explicit client CFG is never overridden.
- The demo resolves the engine-owned managed instruction before sending it, so its `Style control (effective)` log is exactly the text Runtime tokenizes.
- Managed Warm in the demo gets a +5% request-gain multiplier (~+0.42 dB). The user's base gain remains unchanged in `settings.cfg`.
- External HTTP clients do **not** receive an implicit Warm gain lift; they continue to own their explicit gain policy.

## Regression checks
`validate_managed_style_strength.py` checks tuning values, CFG capping/override semantics, effective-control logging, effective gain request wiring, and preservation of base settings.
