# VoxGen v0.7.45 speech DSP echo / affect-coloration fix

## Regression identified

VoxGen v0.7.40-v0.7.42 moved playback time scaling from the client into the engine by adopting the `wsola` 0.1.0 crate. The architectural ownership change was correct, but the algorithm was not behaviorally equivalent to the pre-migration speech path.

The previous speech path selected WSOLA candidates using normalized waveform correlation, a 7.5 ms search half-range, and a candidate stride of roughly 1/6000 second. The generic crate instead used an unnormalized dot product, a 12 ms search half-range, and every-sample search. For expressive or amplitude-varying speech, raw dot-product matching can favor a louder candidate rather than the best phase/waveform continuation. Repeated poor overlap choices can be heard as short echo/doubling, phasiness, or an unnaturally tense timbre.

## v0.7.45 correction

`src/playback_dsp.rs` now owns a `SpeechWsola` implementation directly. It restores the speech-oriented behavior:

- 30 ms analysis window;
- 15 ms overlap;
- 7.5 ms search half-range;
- candidate stride ~= sample_rate / 6000;
- normalized correlation `dot / (||reference|| * ||candidate||)`;
- nearest-to-predicted tie breaking to avoid unnecessary pitch-period hopping;
- exact pitch-neutral sinc bypass, so speed-only requests do not traverse the resampler.

The external `wsola` dependency has been removed from the engine. The demo and HTTP server still consume the same `voxgen::playback_dsp` implementation, so the single-DSP-ownership architecture remains intact.

## Affect interpretation

VoxGen does not classify `neutral`, `concerned`, or other Dynamic Dictionary style labels. It synthesizes from the supplied text/control/reference and then applies the requested numeric playback DSP. A request at 89% is intentionally slower than the generated performance and can perceptually sound more deliberate/serious even when the acoustic reference is neutral. v0.7.45 fixes unintended DSP coloration; it does not reinterpret or ignore the requested speed.
