# VoxGen v0.7.57 — Demo transition scope build fix

v0.7.56 introduced native managed mood transitions and added transition controls to the wxDragon demo. The transition controls themselves were correctly created and captured by benchmark-related callbacks, but the main **Speak** callback omitted the local snapshots that convert the three ComboBox selections into request values.

That caused Rust E0425 errors for `transition_enabled`, `transition_to_style`, `transition_to_intensity`, and `transition_mode` when compiling `voxgen-demo`.

v0.7.57 fixes only that demo callback boundary. At the start of the Speak handler it now reads:

```rust
let transition_to_style = table_key(&TRANSITION_TARGETS, transition_to_control_copy.get_selection()).to_string();
let transition_to_intensity = table_key(&INTENSITIES, transition_intensity_control_copy.get_selection()).to_string();
let transition_mode = table_key(&TRANSITION_MODES, transition_mode_control_copy.get_selection()).to_string();
let transition_enabled = transition_to_style != "none";
```

These values are therefore in scope before transition validation, settings persistence, neutral-reference resolution, logging, and `build_demo_expressive_request` construction.

The release also removes a final `playback_started = true` assignment that occurred after the last read of that flag. Removing it silences the compiler warning without changing playback state or timing results.

No HTTP API, transition compiler, model inference, gain, WSOLA, peak guard, streaming, or prosody behavior changed from v0.7.56.

`validate_demo_transition_scope_fix.py` prevents recurrence by checking that all four transition locals are declared before their first use in the Speak callback and that the dead final playback assignment is absent.
