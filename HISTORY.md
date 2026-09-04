# VoxGen history

This file consolidates the historical validation and research notes that previously lived as many separate Markdown files in the repository root. It is intentionally a concise engineering history rather than a second README.

## v0.7.67 — latency simplification

A source review found several avoidable operations on the time-to-first-audio path. This release removes them instead of preserving compatibility layers.

- Managed styles now compile directly to compact final VoxCPM2 conditioning text. The former managed-suffix recipe, runtime detection/refinement pass, and text-parsed tuning path were removed.
- Style/intensity tuning is a direct structured lookup; custom controls no longer pass through managed-control detection.
- Short-utterance safeguards remain, but are expressed as compact style-specific clauses rather than long secondary prompt expansions.
- Automatic style transitions use one compact continuous transition compiler. The unused quick/alias transition modes are gone.
- A sequence now owns one PCM writer for its full lifetime instead of creating and joining a writer thread for every acoustic run.
- Runtime conditioning builds one plan per acoustic run and reuses that same plan for prefill instead of rebuilding it.
- Synthesis no longer constructs full `RuntimeStatus` snapshots just to read immutable CFG/context/readiness values; those are cached once when the runtime loads.
- The synthesis hot path skips diagnostic prefix checksums/position snapshots after prefill; diagnostic APIs still compute them when explicitly requested.
- Streaming callbacks no longer duplicate every emitted PCM block into `TtsResult.samples`; the callback owns streamed PCM, while the CLI explicitly collects it only when it needs a complete output buffer.
- Rolling AudioVAE input reuses one buffer instead of allocating a temporary context vector and sample buffer on every generated patch.
- Redundant per-chunk socket flushes were removed.
- Canonical managed style/intensity names are required; old managed aliases are no longer accepted.

The deeper inference costs are intentionally unchanged in this cleanup. Streaming AudioVAE still uses its six-patch rolling compatibility decoder, and the autoregressive loop still has several serialized GPU submit/wait boundaries. Replacing those requires stateful decoder work and/or command-scheduling changes, not a safe simplification pass.

### Repository validation cleanup

- Removed the historical per-version and source-shape Python validators. They duplicated implementation details, made refactoring brittle, and did not participate in the runtime or build.
- Kept only three repository smoke validators: `validate_build_layout.py`, `validate_clean_source.py`, and `validate_demo.py`.
- Behavioral invariants belong in Rust unit/integration tests next to the implementation; existing native tests cover managed prosody controls, BaseLM invariants, acoustic helpers, and tokenizer behavior.
- No inference, HTTP, DSP, prosody, model-loading, or streaming behavior changed in this repository cleanup.

## Early implementation milestones

### Iteration 5 — UnifiedCFM latent generation

- Implemented the VoxCPM2 UnifiedCFM inference contract: temperature-scaled initial noise, sway-transformed timestep schedule, CFG-Zero* startup, conditional/unconditional LocDiT passes, optimized negative-scale reduction, guided velocity, and Euler updates.
- Kept positive/unconditional estimator work on Vulkan; `mu=0` for the unconditional branch used a GPU buffer fill rather than host round-trips.
- Added deterministic parity mode through explicit `--cfm-initial-x-f32` so solver/LocDiT differences could be isolated from RNG differences.
- This milestone stopped at acoustic latent patches; AudioVAE/waveform output was deliberately deferred.

### Iteration 6 — AudioVAE V2

- Added VoxCPM2 AudioVAE V2 encoding and decoding, including causal convolution/transposed-convolution geometry, Snake activations, sample-rate conditioning, and the 4-frame acoustic patch contract.
- Established the core timing geometry: 2,560 input samples at 16 kHz and 7,680 output samples at 48 kHz per 160 ms acoustic patch.
- Connected WAV conditioning to the earlier conditioning pipeline and added deterministic VAE fixtures.
- Validation covered shader/source parity, causal indexing, padding/alignment contracts, tensor dimensions, and standalone-project isolation.
- The autonomous autoregressive TTS loop, stop predictor, streaming decoder state, and end-to-end speech API remained the next milestone.

### Iteration 7 — complete text-to-waveform inference

- Connected tokenizer, reference/prompt AudioVAE conditioning, BaseLM/ResidualLM prefix prefill, CFM patch generation, acoustic stop prediction, LocEnc feedback, rolling AudioVAE decode, and HTTP speech output into a complete VoxCPM2 inference loop.
- Added an internal GGUF-native byte-fallback BPE tokenizer, removing the runtime Python/Hugging Face tokenizer dependency.
- Added `/v1/audio/speech` and `/v1/audio/speech/stream`, including reference and prompt audio support.
- Kept the generation fast path on GPU except for intentionally small control/output transfers such as generated latent patches, stop logits, and PCM.

## v0.7.34 — power-efficiency pass

- Fused Q/K/V projection dispatches in BaseLM/ResidualLM and sequence Q/K/V where appropriate.
- Replaced broad shader-memory barriers with targeted buffer-scoped compute dependencies in transformer hot loops.
- Kept positive CFG `mu` resident across a CFM solve and removed unnecessary save/restore copies and buffer clears.
- Preserved model weights, sampling settings, CFM/CFG math, playback DSP, clocks, voltage, and power limits.

## v0.7.35 — streaming startup latency

- Switched the desktop demo to stable reference/prompt paths instead of per-request base64 copies.
- Added path-based AudioVAE conditioning cache and `/v1/audio/conditioning/warm` prewarm support.
- Reduced startup prebuffer to one acoustic patch at normal/slower playback while retaining extra reserve for faster playback.
- Published the current PCM patch before stop-predictor synchronization.
- Increased XTX live text-prefix batching and exposed engine version through `/health`.

## v0.7.36 — neutral voice anchor

- Made an explicitly configured Neutral reference the canonical fallback speaker identity.
- Reference resolution became: requested style reference → Neutral anchor → legacy/default reference → runtime default → zero-shot only when no anchor has ever been configured.
- Missing configured Neutral/default references now fail visibly rather than silently switching speaker identity.
- The same reference resolver is used for synthesis, benchmarking, profiling, and prewarming.

## v0.7.37 — mid-generation cancellation

- Added a dedicated Stop path to the wxDragon demo.
- Cancellation sets a shared atomic flag and, on Windows, resets the active WinMM device to flush queued PCM immediately.
- Added `/v1/audio/speech/cancel` without taking the inference lifecycle lock, allowing cancellation while synthesis owns the inference gate.
- Runtime cancellation remains safe-boundary based: it checks between completed GPU operations/acoustic patches and never interrupts an in-flight Vulkan submission.

## v0.7.38 — streaming fidelity and anger-artifact mitigation

Research identified the rolling AudioVAE decoder as a major source of metallic/brittle streaming artifacts. The newest 7,680-sample output patch depends on 24 latent frames, equal to six 4-frame patches; three- or four-patch rolling decode silently replaced required history with zeros.

Changes:

- Enforced a six-patch minimum rolling AudioVAE context.
- Changed HTTP/demo defaults to six patches.
- Reduced fresh-demo gain from 140% to 100% to avoid hard-clipping expressive peaks.
- Reworked Angry guidance around controlled tension, directness, timing, and phrase-level emphasis instead of sustained loudness/shouting.
- Recommended same-seed A/B testing against non-streaming output and CFG/timestep sweeps when model conditioning remained suspect.

The longer-term direction identified here was a truly stateful AudioVAE streaming decoder with cached causal state.

## v0.7.39 — clean-source scripts

- Added root and demo cleanup entry points for Windows and Linux.
- Cleaners remove Cargo intermediates, project-local model/download/cache trees, generated lockfiles, and smoke-test outputs while preserving final executables and checked-in fixtures.
- External/global Cargo caches and model directories are intentionally untouched.
- Windows reparse points are removed as links rather than recursively traversing external targets.

## v0.7.40 — native playback DSP

- Centralized playback speed/pitch in `src/playback_dsp.rs`.
- Pitch: 24-tap Lanczos-windowed sinc resampling with anti-alias handling.
- Tempo: speech-oriented stateful WSOLA.
- Independent controls use pitch transposition plus tempo compensation; neutral `100% / 0 st` has a dry bypass.
- HTTP and demo share the same engine-owned implementation, eliminating duplicated client-side VoxGen DSP.
- Added `speed_percent` and `pitch_semitones` to speech requests.

## v0.7.45 — playback CLI, WSOLA correction, and managed positive prosody

### CLI playback controls

- Added concise `--speed` and `--pitch` flags while keeping unit-bearing internal/HTTP field names.

### Speech WSOLA correction

The generic WSOLA implementation used during the initial DSP migration was not behaviorally equivalent to the earlier speech path and could cause echo/doubling, phasiness, or unintended affect coloration.

- Restored 30 ms windows, 15 ms overlap, 7.5 ms search half-range, sparse candidate stride, normalized waveform correlation, and nearest-to-predicted tie breaking.
- Added exact pitch-neutral sinc bypass for speed-only requests.
- Removed the external `wsola` dependency; VoxGen again owns `SpeechWsola` directly.

### Warm/Cheerful managed prosody

- Introduced engine-owned managed prosody compilation rather than duplicating recipes in the demo.
- Subtle Cheerful emphasizes a light audible smile, slight brightness, small buoyant pitch lifts, lively rhythm, and conversational loudness.
- Subtle Warm emphasizes a soft smile, connected phrasing, relaxed articulation, soft attacks, moderate-low loudness, and small welcoming pitch lifts.
- Very short lines are guarded against building to a climax; punctuation does not automatically imply extra loudness.
- No post-generation EQ, formant shifting, pitch shifting, compression, or waveform filtering was introduced for these styles.

## v0.7.47 — WSOLA headroom and shared-library import fixes

- Replaced equal-power WSOLA overlap with amplitude-complementary raised-cosine overlap. For aligned copies of the same speech signal this avoids the roughly +3 dB midpoint overshoot possible with cos/sin gains.
- Removed internal playback-DSP clamping so floating-point headroom survives until final output handling.
- Fixed the binary/library crate import boundary for `prosody_control` and removed an unused playback-DSP constant.

## v0.7.48 — warmth/subtle-positive research refinement

- Distinguished low-arousal Warm/Tender delivery from high-arousal Cheerful/Joyful delivery.
- Warmth preserves the speaker's habitual pitch centre and relies on low effort, smooth phrasing, softened attacks, gentle vowel lengthening, mellow resonance, and only a faint smile cue.
- Cheerfulness remains brighter and rhythmically more buoyant with a definite but restrained increase in pitch variation.
- The refinement remained entirely model-control based; no emotion-specific DSP was added.

## v0.7.49 — managed style strength

- Added small automatic CFG deltas for low-arousal managed styles when the client does not explicitly supply CFG: Warm +0.20, Gentle +0.15, Subtle Cheerful +0.10, capped at 3.0.
- Explicit client CFG remains authoritative.
- Demo effective-control logging now shows the exact instruction tokenized by Runtime.
- Warm receives a visible demo-only +5% request-gain multiplier; external HTTP clients retain explicit gain ownership.

## v0.7.50 — Excited and Concerned profiles

- Excited targets high-arousal positive delivery using moderately elevated pitch centre, wider/dynamic pitch excursions, faster transitions, crisp articulation, shorter pauses, and local emphasis peaks without sustained shouting.
- Concerned intentionally avoids treating concern as panic; it uses mild tension, responsive/local pitch elevation, focused articulation, modest urgency, and a release toward calmer reassurance.
- Concerned receives only a small automatic CFG delta when CFG is omitted.

## v0.7.51 — Whisper-like and controlled Angry

### Whisper-like

- Targets a controlled near-whisper rather than simply lowering volume.
- Guidance emphasizes low vocal effort, airflow/noise, softened attacks, reduced periodic voicing, careful articulation, intelligibility, and speaker identity.
- Managed steering uses a small CFG lift; the demo uses a visible 0.85x gain multiplier.

### Angry

- Targets controlled/cold anger rather than sustained hot/shouted anger.
- Guidance keeps pitch centre near neutral or slightly lower, uses a compact purposeful pitch range, firm tension, clean attacks, quicker timing, short pauses, brief local emphasis peaks, and moderate sustained loudness.
- No automatic CFG escalation is applied; demo gain is reduced to 0.90x.

## v0.7.52 — Excited/Gentle refinement

- Refined Excited toward dynamic local pitch/intensity peaks instead of sustained loudness and removed surprise-like cues from Subtle Excited.
- Defined Gentle primarily through low vocal effort and low projection rather than treating it as a synonym for Warm/Tender.
- Explicit API gain/CFG remain authoritative; no hidden emotion-specific DSP was added.

## v0.7.53 — Serious and Sad profiles

- Sad uses relative lower mean F0, narrower F0 range, lower intensity, and slower rate while avoiding boredom, sleepiness, depression-like flattening, whispering, or grief/wailing.
- Serious is treated as a communicative stance rather than a basic emotion: deliberate timing, stable/firm delivery, resolved intonation, and the speaker's natural pitch range.
- Sad receives modest CFG/gain adjustments by intensity; Serious receives small CFG guidance but no demo gain change.

## v0.7.54 — Neutral prosody cleanup

- Reframed Neutral as a natural conversational baseline, not flat speech.
- Preserves habitual pitch centre/range, lexical stress, syntax-driven contours, rate, loudness, and subtle timing variation while suppressing imposed emotional coloration.
- Strong Neutral explicitly rejects monotone/robotic delivery and forced-low pitch.
- Neutral tuning remains CFG +0.00 and demo gain 1.00x.

## v0.7.55 — all-style audio safety

### Final peak protection

- Replaced destructive per-sample full-scale clipping with a shared `OutputPeakGuard` at a 0.98 ceiling.
- Offline output uses one uniform attenuation factor; streaming performs block look-ahead, immediate attenuation when necessary, and monotonic release.
- The guard never boosts below-ceiling audio and is not a loudness normalizer/compressor.

### WSOLA low-confidence fallback

- Added conservative fallback to the predicted analysis position when overlap energy is extremely low or normalized-correlation confidence is below 0.20.
- This reduces phasey/warbling/metallic behavior on breathy, fricative, and Whisper-like material without adding EQ/denoising/spectral processing.

## v0.7.56 — managed mood transitions

- Added one-pass transitions between managed style/intensity destinations through natural-language prosody conditioning.
- No waveform crossfade, double synthesis, PCM interpolation, or reference-WAV crossfade is used.
- Supports gradual and quick modes, same-style intensity transitions, pair-specific trajectories for common transitions, and a generic continuous fallback for uncommon pairs.
- Transition requests preserve explicit CFG precedence and normally use the Neutral reference as the stable speaker anchor when the client maintains a reference bank.

## v0.7.57 — demo transition scope build fix

- Fixed missing local snapshots for transition controls in the main Speak callback, resolving Rust E0425 build errors.
- Removed a dead final `playback_started` assignment.
- No inference, HTTP, DSP, prosody, gain, reference, or streaming behavior changed.

## v0.7.58 — extended mood and managed-pace transitions

- Added dedicated trajectories for additional high-value style pairs including Neutral/Serious, Cheerful/Excited, Concerned/Serious, Warm/Gentle, Warm/Sad, Sad/Serious, Concerned/Sad, Angry/Neutral, Angry/Concerned, and Whisper-like transitions.
- Added optional `from_speed_percent`/`to_speed_percent` managed pace trajectories realized by VoxCPM2 conditioning rather than mid-utterance WSOLA changes.
- Enforced 50–200% endpoints, a 5-point minimum meaningful delta, a 45-point maximum one-phrase delta, and anti-abruptness rules.
- Uniform playback speed remains 100% while managed pace transitions are active; pitch stays independently adjustable.

## v0.7.59 — JSON macro recursion build fix

- Increased Rust macro recursion headroom to 512 in the binary, shared library, and demo crate roots after the enlarged `/health` JSON object exceeded the default `serde_json::json!` recursion limit.
- Runtime behavior was unchanged.

## v0.7.60 — automatic continuity

- Removed the public explicit transition object in favor of destination-only requests: style, intensity, managed pace, continuity ID, and boundary.
- Added continuity-session state with 30-minute expiry, a 256-ID cap, success-only state commits, reset endpoints, and generation epochs preventing reset races.
- Continuity is scoped to speaker conditioning; model load/unload and pipeline reset clear continuity state.
- Managed pace suppresses changes under 5 points and limits one successful phrase to at most a 45-point advance toward a distant target.
- Active continuity requires playback `speed_percent=100`; prosodic speaking-rate changes use managed `pace_percent`.

## v0.7.61 — authoritative runtime contract

- Consolidated managed-profile, intensity, boundary, managed-pace, output-audio, speech-request, seed-policy, and execution-mode metadata under VoxGen's own runtime contract.
- Kept playback speed/pitch ranges tied directly to engine DSP constants rather than duplicate values.
- `/health` remained the engine's factual status/capability surface.

## v0.7.62 — engine-owned trailing pauses

- Added semantic `pause_after` handling to speech requests.
- VoxGen owns pause parsing and realization, including trailing silence in both streaming and offline output.
- Responses expose the effective semantic pause.

## v0.7.63 — native speech-sequence streaming

- Added `SpeechSequenceRequest` with an ordered `segments` array and `/v1/audio/speech/sequence/stream`.
- A sequence acquires the inference gate once and consumes the complete semantic plan within one sequence operation.
- Single-phrase streaming remains available separately.

## v0.7.64 — client buffering ownership

- Removed server-prescribed client prebuffer/rebuffer durations.
- VoxGen reports factual delivery behavior; playback buffering policy belongs to the consuming client.

## v0.7.65 — progressive sequence delivery

- Replaced whole-acoustic-run buffering with direct publication from Runtime's live PCM callback.
- Processed PCM enters the writer channel immediately and each block is exposed as soon as it is available.
- Continuity state commits after successful acoustic runs; coalesced semantic entries commit atomically to the final state of their compiled run.

## v0.7.66 — semantic sequence run compiler

- Added compilation of adjacent compatible semantic destinations into fewer physical acoustic generation runs.
- Coalescing requires compatible managed destination, steady continuity state, speaker conditioning, decoding settings, and boundary/pause semantics.
- Explicit seeds or per-segment generation windows prevent coalescing; long pauses, hard cuts, speaker changes, and active gradual transitions remain run boundaries.
- Generation ceilings scale for merged runs.
- Sequence responses identify the adjacent-compatible steady-state compiler, and diagnostics report semantic-segment count versus physical acoustic-run count.
- The result preserves semantic planning while avoiding unnecessary model-generation boundaries.

## Research notes retained from the former standalone files

The historical work repeatedly used the following principles:

- Expressive delivery should be modeled through combinations of F0/pitch behavior, timing/rate, intensity, spectral/voice-quality cues, and articulation rather than loudness alone.
- Low-arousal positive states such as warmth/tenderness are distinct from high-arousal cheerfulness/excitement.
- Concern is not equivalent to panic; controlled/caring delivery can involve slower rate, steadier energy, and smoother contours.
- Whispering is a phonation change, not merely quiet speech.
- Hot and cold anger have different acoustic signatures; VoxGen deliberately favors controlled/cold anger to protect naturalness and headroom.
- Neutral speech must retain ordinary linguistic prosody rather than flattening pitch, stress, and timing.
- Emotion/style realization remains model-conditioning driven; shared playback DSP and output-safety stages are style agnostic.

Notable research sources consulted during these passes included OpenBMB VoxCPM2/AudioVAE documentation and source, affective-prosody reviews and experimental studies on vocal emotion, smiling, empathy/caring, whispering, vocal effort, anger subtypes, and EBU peak/headroom guidance.

## Historical validation policy

The old per-release Markdown files documented source-level validators because some packaging environments lacked Rust/Cargo and Vulkan shader compilers. Where native compilation was unavailable, those documents explicitly avoided claiming Cargo/SPIR-V/runtime validation and deferred native numerical/performance checks to the target Vulkan machine.

The old per-release Python source-inspection suite was removed in v0.7.67. Only the build-layout, clean-source, and demo smoke validators remain; behavior-specific regression coverage belongs in native Rust tests.

## v0.7.67 repository fixture cleanup

- Removed the committed deterministic `test_*.f32` smoke-test fixtures from the source package.
- The existing fixture generators remain authoritative and recreate those inputs on demand.
- Root clean-source scripts now remove generated `.f32` smoke inputs as disposable artifacts.
- No inference, model, DSP, HTTP, streaming, or GPU behavior changed.
