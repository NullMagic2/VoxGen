# VoxGen v0.7.55 — All-style clipping / metallic safety

## v0.7.55: Shared output peak guard and low-confidence WSOLA fallback

This release hardens the common audio path for **every** managed style and custom control without changing their emotion recipes. VoxGen now replaces the old final `(sample * gain).clamp(-1, +1)` behavior with an engine-owned `OutputPeakGuard`: complete utterances receive a single uniform attenuation only when their post-gain peak would exceed 0.98, while streaming blocks use block look-ahead, immediate safe attenuation and a 250 ms slow release. Below the ceiling the guard is transparent. The CLI and HTTP paths use the same guard, and the demo protects the output again after its live native speed/pitch DSP because sinc/WSOLA can create small post-server overshoots.

WSOLA also gains a low-confidence synchronization fallback. Normalized-correlation matching remains authoritative for voiced speech, but very low-energy overlaps or matches below NCC 0.20 now stay at the predicted analysis position instead of chasing accidental correlations in breath/noise-dominated material. This specifically reduces phasey/warbling/metallic artifacts on Whisper-like speech, fricatives and other weakly periodic spans while retaining the amplitude-complementary raised-cosine overlap that fixed the earlier +3 dB/echo problem. No per-style EQ, compressor, pitch shifter, spectral enhancer or hidden gain stage is introduced. See `V092_ALL_STYLE_AUDIO_SAFETY_VALIDATION.md`.

# VoxGen v0.7.54 — Neutral prosody cleanup

## v0.7.54: Natural neutral baseline

This release makes managed **Neutral** a true reference baseline rather than an instruction for vague "emotional balance". Neutral now preserves the cloned speaker's habitual pitch centre, ordinary pitch variability, normal rate/loudness, lexical stress, syntax-driven phrase contours and small human timing variation while avoiding any deliberately imposed affective stance. It explicitly avoids the two common failure modes of neutral TTS: becoming emotionally coloured (warm/serious/etc.) or becoming unnaturally flat, robotic, cold or monotone. Short questions/exclamations retain ordinary linguistic intonation without being promoted into emotional acting.

Neutral receives **no automatic CFG delta and no demo gain multiplier** (`+0.00`, `1.00x`), making it a cleaner acoustic baseline for comparison with every managed style. Older v0.7.53 Neutral recipe strings remain recognized and are upgraded by the runtime compiler. No playback DSP, EQ, pitch shifting, compression, reference conditioning, or generation topology changes are introduced.

# VoxGen v0.7.53 — Serious / Sad realism

## v0.7.53: Serious stance and low-arousal Sad prosody

This release adds dedicated research-guided compilers for **Serious** and **Sad**. Serious is treated as a communicative stance rather than a basic emotion: natural pitch centre, controlled but living pitch excursions, deliberate semantic pauses, firm clean articulation, stable conversational intensity, selective prominence, and resolved falling endings where linguistically appropriate. It explicitly avoids the old Strong-Serious `grave, authoritative` seed, forced deep pitch, ominous/movie-trailer delivery, anger and monotone. Managed Serious receives only conservative CFG guidance (+0.05 Subtle; +0.10 Normal/Strong) and no automatic demo gain change.

Sad follows the robust low-arousal acoustic pattern reported across emotional-prosody studies: lower mean pitch, narrower pitch range, lower intensity and slower rate than neutral. The compiler scales those cues by intensity and explicitly prevents Sad from collapsing into sleepiness, boredom, depression-like flatness, whispering, or high-arousal grief/wailing. The demo applies visible request-time Sad level multipliers of 0.97x / 0.94x / 0.90x for Subtle / Normal / Strong and a +0.10 managed CFG delta. External clients that explicitly specify gain/CFG remain authoritative; no post-synthesis EQ, pitch shifting, compression, or hidden emotion DSP is added.

# VoxGen v0.7.52 — Excited / Gentle realism

## v0.7.52: Excited re-audit and Gentle low-effort voice

This release re-audits **Excited** and adds a full managed compiler for **Gentle**. Excited is no longer seeded from a "pleasantly surprised" recipe: subtle excitement now means positive anticipation/interest with a slightly raised pitch centre, more variable/wider pitch movement, lightly faster timing, and brief local pitch/intensity peaks that release back toward conversational level. Normal and Strong Excited keep high-arousal positive dynamics without sustained loudness, shrillness, frantic delivery, or surprise-like pitch shocks.

Gentle is now defined as a **low-vocal-effort speaking style**, not a synonym for Warm/Tender. It uses reduced projection/loudness, smooth connected phrasing, light clean attacks, relaxed articulation and restrained but living pitch movement while preserving the text's own emotional valence. The demo uses visible request-time Gentle level multipliers of 0.98x / 0.95x / 0.92x for Subtle / Normal / Strong and conservative CFG deltas of +0.10 / +0.15 / +0.15. These are not hidden engine/API gain stages; explicit external client gain/CFG remains authoritative.

## v0.7.51: Whisper-like and controlled-Angry realism

This release extends the research-guided managed-prosody compiler to **Whisper-like** and **Angry**. Whisper-like now targets low vocal effort, audible airflow/noise, softened attacks and reduced periodic voicing instead of merely asking for a quiet breathy voice. Because a true whisper removes ordinary periodic voicing/F0 and can weaken cloned-speaker identity, VoxGen deliberately targets a controlled near-whisper that preserves intelligibility and as much identity as possible.

Angry now targets **controlled/cold anger**, not explosive hot anger: firm vocal tension, hard clean attacks, compact purposeful pitch movement, quicker timing/shorter pauses and brief local emphasis while keeping sustained loudness moderate. Short exclamations/questions sharpen timing and attack rather than loudness. The demo applies a conservative 0.90x Angry level and 0.85x Whisper-like level; these are visible request-time multipliers and are not hidden engine/API gain stages.

Managed CFG remains conservative: Whisper-like gets +0.10; Angry gets no automatic CFG increase. Explicit API CFG remains authoritative.


This revision adds research-guided managed-prosody compilers for **Excited** and **Concerned**. Excited speech is steered through dynamic high-arousal cues—moderately raised/wider pitch, quicker transitions, crisp articulation, and brief phrase-level energy peaks followed by release—rather than sustained loudness. Concerned speech is treated as a mixed contour: mild alert tension and responsive pitch on concern-bearing phrases, followed by lower/slower/softer, smoother reassurance where the wording allows. Strong Excited and Strong Concerned are now correctly recognized as strong intensity by the compiler. Managed Concerned receives a conservative +0.10 CFG delta when CFG is not explicitly fixed; Excited receives no automatic CFG increase to avoid over-expression. No EQ, artificial pitch shifting, compression, or post-synthesis emotion processing is added.

# VoxGen v0.7.49 — Managed style strength and Warm level

This revision makes low-arousal managed styles more perceptible without adding post-synthesis coloration. The demo now resolves and logs the **exact effective control instruction** that reaches VoxCPM2. Managed Warm adds +0.20 CFG over the user's base CFG, Gentle adds +0.15, and Subtle Cheerful adds +0.10, all capped at 3.0. HTTP clients that omit `cfg_value` receive the same conservative managed guidance automatically; an explicitly supplied API CFG remains authoritative.

The demo also gives managed Warm a small **+5% request-gain lift** (about +0.42 dB). This is sent as the ordinary VoxGen `gain` value, not applied as a second playback/post-processing stage. The UI continues to store the user's base gain unchanged, and the log shows both base and effective values. External clients such as Dynamic Dictionary retain explicit control of their own per-style gain and do not receive a hidden Warm gain multiplier.

## v0.7.48 warmth / subtle-positive prosody refinement

This revision makes managed **warm** delivery explicitly low-arousal, affiliative and tender instead of treating warmth as neutral speech plus pitch lifts. Managed **subtle cheerful** delivery now keeps loudness/pitch centre near neutral while retaining a small, definite pitch-variability and rhythmic cue floor so it remains perceptibly positive. No new post-generation EQ, pitch, formant, compression or loudness processing is added.

## v0.7.47 shared-library import build fix

- Fixes the managed prosody compiler import in the binary runtime (`voxgen::prosody_control`, not `crate::prosody_control`).
- Removes the unused `DSP_PULL_CHUNK` constant and its build warning.
- Adds a regression validator for binary/library module ownership.

## v0.7.47 WSOLA headroom / expressive-speech distortion fix

- uses an amplitude-complementary raised-cosine WSOLA overlap so correlated speech is not boosted by up to ~3 dB;
- removes destructive clipping from inside playback DSP;
- preserves floating-point headroom until requested gain and final output serialization;
- specifically prevents high-energy/angry speech at non-100% playback speeds from becoming progressively clipped and distorted.

## v0.7.45 warm and subtle-cheerful prosody

VoxGen now compiles its managed **Warm** and **Cheerful** style recipes into concise, research-guided acoustic goals before VoxCPM2 tokenization. Subtle cheerfulness requests a light audible smile, slightly brighter resonance than neutral, small buoyant pitch lifts, gently lively rhythm, and steady conversational loudness. Warmth requests smooth connected phrasing, relaxed articulation, soft consonant attacks, stable moderate-low loudness, and small welcoming pitch lifts. Very short positive lines receive a guard against building to a climax; an exclamation mark is treated as friendly brightness rather than an instruction to shout.

This compiler is deliberately narrow: it recognizes only the exact managed recipe families emitted by VoxGen/Dynamic Dictionary. Arbitrary user-authored `--control` or HTTP `control` instructions remain unchanged. The demo now imports the engine's shared style-recipe builder, eliminating another source of client/demo divergence. No playback DSP, reference normalization, punctuation rewriting, CFG, temperature, or CFM behavior changed in this release.

## v0.7.41 concise playback CLI flags

The VoxGen command-line interface exposes native playback DSP as `--speed` (percent, 50–200, default 100) and `--pitch` (semitones, -12..+12, default 0). The former automatically generated Clap names `--speed-percent` and `--pitch-semitones` are intentionally not accepted and are not retained as aliases. HTTP/API field names remain `speed_percent` and `pitch_semitones` because they encode the units explicitly.

Example:

```text
voxgen --text "Hello" --speed 115 --pitch -2 --output-wav out.wav
```

## v0.7.40 native playback DSP

VoxGen now owns speed/pitch playback processing as part of the engine crate. `src/playback_dsp.rs` is the single authoritative implementation: a band-limited sinc pitch transposer followed by engine-owned, normalized-correlation speech WSOLA time scaling. The HTTP speech API accepts `speed_percent` (50–200, default 100) and `pitch_semitones` (-12..+12, default 0) and applies the DSP before returning streaming or completed PCM/WAV. `/health` advertises `native_playback_dsp: true` plus the supported ranges.

The wxDragon demo consumes the same engine module through a thin live-control adapter instead of maintaining its own copy of the algorithm. At 100% / 0 semitones the original finite PCM takes a neutral dry bypass. The engine now owns the speech WSOLA implementation directly. The v0.7.43 matcher restores normalized waveform correlation and the narrower search geometry used by the pre-migration playback path, avoiding the energy-biased candidate selection that could sound echoey or unnaturally tense. External clients should send speed/pitch controls to VoxGen rather than post-processing VoxGen audio independently.


## v0.7.39 clean-source scripts

VoxGen now includes `clean_source.bat` (Windows) and `clean_source.sh` (Linux) at the project root. Running either cleaner removes engine/demo Cargo build intermediates, project-local downloaded model/cache folders, generated smoke-test outputs, and Cargo lockfiles while preserving any final `voxgen` / `voxgen.exe` and `voxgen-demo` / `voxgen-demo.exe` binaries found under the debug or release target directories. The cleaner is deliberately project-local: it never deletes the global Cargo cache or model paths outside the VoxGen tree. Matching wrappers are also included under `demo/`, so the same cleanup can be launched from either the engine root or demo folder.

## v0.7.38 streaming-fidelity update

VoxGen now protects the compatibility rolling AudioVAE stream decoder from undersized context windows. The current decoder topology needs six latent patches of causal context for the newest 160 ms chunk, so values below six are raised internally to six. This specifically targets brittle, metallic, or cracking artifacts that become conspicuous in highly expressive/high-energy speech. Fresh demo settings also start at 100% gain rather than 140%, and the built-in angry recipes emphasize controlled phrase-level tension instead of continuous loudness.

The longer-term target remains a fully stateful AudioVAE streaming path with cached convolution state.
# VoxGen v0.7.36 — neutral voice anchor

VoxGen is a standalone Rust/Vulkan inference engine specialized for **VoxCPM2**. Version 0.7.36 changes reference fallback behavior so a configured Neutral WAV remains the canonical speaker identity whenever a style-specific clip is unavailable. It preserves the v0.7.35 low-latency streaming startup work and the v0.7.34/v0.7.33 power-efficiency kernels.

## Neutral voice anchoring (v0.7.36)

- **Preset miss -> neutral anchor:** if the selected Warm/Cheerful/Excited/Sad/etc. preset has no usable dedicated WAV, the demo uses `emotion_reference.neutral` rather than switching to an unanchored voice.
- **Neutral is authoritative:** a saved Neutral preset is consulted directly on Speak, A/B benchmark, XTX profiling, style-change prewarm, and model-load prewarm. It no longer depends on whether the legacy `voice_sample=` field happened to be populated.
- **No silent hallucinated fallback after a configured anchor disappears:** if a configured Neutral/default WAV has been moved or deleted, VoxGen refuses the request and tells the user to restore or reselect it instead of dropping into zero-shot generation.
- **Legacy settings remain compatible:** older `voice_sample=` references still act as the default anchor when no explicit Neutral preset exists. On load, an explicit Neutral preset is mirrored into an empty legacy `voice_sample=` field.
- **Zero-shot remains available only when truly unconfigured:** if no preset reference, Neutral anchor, legacy/default voice sample, or runtime reference has ever been configured, Controllable Reference mode may still fall back to VoxCPM2 native zero-shot.
- Selecting Neutral through either reference picker updates the canonical default anchor; clearing Neutral clears the matching legacy/runtime alias as well.
- No model weights, CFG, temperature, CFM steps, power tuning, or streaming DSP behavior are changed by this policy.

## Low-latency streaming startup (v0.7.35)

- **Stable localhost reference path:** the bundled demo sends `reference_audio_path` / `prompt_audio_path` instead of re-reading and base64-expanding the WAV for every Speak operation. External API clients can still use base64.
- **AudioVAE conditioning cache + prewarm:** reference/prompt latents are cached by canonical path, size, modification time, and pad side. The active preset reference is pre-encoded after model load; changing the style preset or selecting a new reference warms that exact WAV in the background, moving that work out of click-to-audio latency.
- **First-patch playback at 100% or slower:** WinMM can start from the first 160-ms acoustic patch. Faster playback retains a larger startup reserve.
- **PCM before stop-head readback:** streaming publishes the current decoded patch before running the stop predictor because that decision only controls whether another patch is generated.
- **XTX live prefill32:** live XTX prefill records 32 text positions per submit/wait batch; offline timestamp profiling remains at 16 positions for bounded query accounting.
- **Version-safe local reuse:** `/health` exposes the VoxGen package version and the demo restarts an older listener instead of silently reusing a server that lacks the new cache/prefill/startup behavior.
- These changes do not reduce CFM steps, alter CFG/temperature, or change the generated current patch to gain startup speed.


## QKV + targeted synchronization power pass (v0.7.34)

- BaseLM and ResidualLM now project **Q, K, and V in one Vulkan dispatch**. The combined shader supports the same F16/Q8_0 matrix formats as the previous separate matvecs; XTX 7900 uses a wave32/subgroup-reduction variant. This removes two pipeline dispatches per transformer layer without changing the projection equations.
- LocEnc/LocDiT use an equivalent **sequence QKV fusion** in Normal and default XTX mode. When the experimental XTX cooperative-matrix path is explicitly enabled, VoxGen intentionally keeps the three coopmat projections rather than replacing them with a non-coopmat fusion.
- The BaseLM, ResidualLM, LocEnc, and LocDiT transformer loops now use **buffer-scoped compute barriers** for their true producer/consumer buffers instead of global shader-memory barriers after every stage. Unrelated storage buffers are no longer included in those hot-path dependencies.
- The positive CFG `mu1/mu2` vectors now stay resident for the entire CFM solve. `pack_locdit` emits zero mu tokens directly for the unconditional pass, so VoxGen removes both the old negative-pass fills **and** the per-step save/restore copies and compute↔transfer barriers that those fills forced. The mathematical positive/negative conditioning is unchanged.
- Workgroup sizes remain at the previously validated 256-thread defaults. v0.7.34 deliberately does **not** guess a lower-workgroup policy without target-hardware measurements; this avoids turning a power optimization into an unmeasured throughput regression.
- No clock cap, power limit, undervolt, or quality/CFM change is applied.

## Shared transformer power-efficiency pass (v0.7.33)

The BaseLM, ResidualLM, and LocalDiT transformer hot paths now share two architecture-neutral fusions designed to reduce dispatch count, synchronization, and intermediate VRAM traffic without changing model mathematics:

- **Residual + RMSNorm fusion:** the residual branch is accumulated into the hidden state and normalized in one compute dispatch. The end-of-layer fusion also prepares the next layer's attention-normalized input (or the final output norm), removing a separate residual dispatch, a separate RMSNorm dispatch, and an intervening global compute barrier.
- **Gate + Up + SwiGLU fusion:** the gate and up projections are accumulated together from the same normalized input and the finished SwiGLU activation is written directly to the FFN activation buffer. This removes the second projection dispatch, the separate SiLU/multiply dispatch, the `up` intermediate buffer in BaseLM/ResidualLM, and one synchronization point.
- Both fusions have **portable Normal shaders** and **subgroup-optimized XTX 7900 shaders** selected through the existing execution-mode mechanism.
- LocalDiT keeps the existing cooperative-matrix gate/up path when cooperative matrices are explicitly enabled; the fused SwiGLU path is used for Normal mode and the default stream-safe XTX configuration where cooperative matrices are off. This avoids replacing an explicitly selected specialized path without benchmark evidence.
- The first attention RMSNorm remains separate because it follows embedding/fusion input rather than a residual branch. Subsequent attention norms are produced by the preceding fused residual-normalization dispatch.

These changes target lower energy per generated audio second and should also reduce kernel-launch overhead. The package intentionally does **not** impose a GPU power limit, clock cap, or undervolt, so it does not trade throughput for a lower board-power number.

## Live streaming speed recovery (v0.7.32)

The Windows demo now treats a live Speed/Pitch change as a DSP synchronization boundary. WSOLA overlap/search state is rebuilt for the new rate (and the pitch resampler is reset only when pitch itself changes), preventing a previously slower tempo from leaking into later streamed chunks. Live WinMM submission is also bounded to two pending blocks and the capacity wait happens **before** DSP rendering, so audio is rendered using the most recent controls instead of accumulating seconds of already-slow PCM ahead of playback. This behavior is shared by Normal and RX 7900 XTX modes because it belongs to the demo playback layer, not the inference kernels.

## Adaptive streaming reserve (v0.7.31)

Historically (v0.7.31), Windows streaming reserved **two patches** before the first WinMM submission, growing the reserve for marginal cadence or faster playback. **v0.7.36 supersedes that startup policy at 100% speed and below:** playback can now begin from the first 160-ms patch; faster playback still retains an explicit reserve. Once playback begins, newly generated PCM is queued immediately.

The normal benchmark block now distinguishes **First PCM ready** from **Time to playback start** and reports the adaptive startup reserve. This makes the latency cost of underrun protection visible instead of hiding it inside TTFA.

## Controlled Normal-vs-XTX benchmark (v0.7.31)

The demo includes **Benchmark Normal vs XTX**. It runs the current text once in each execution mode using the **same seed, same models, same reference/zero-shot state, same clone mode, CFG, temperature, and CFM steps**, with playback disabled. It then restores the engine mode selected in the UI. The result block includes a stable text fingerprint, model filenames, reference path, engine/wall time, first-PCM time, RTF, patch count, and percentage improvement for XTX. If stop prediction produces a different patch count despite the identical seed, the report calls that out explicitly.

## Offline XTX GPU profiler (v0.7.31)

`--benchmark-profile` is an offline-only XTX mode. It requires `--mode xtx7900 --stream off`, enables Vulkan timestamp profiling, and is deliberately separate from the stream-safe live server. The HTTP API exposes `GET /v1/profile/gpu` and `POST /v1/profile/gpu/reset`. The demo's **Profile XTX** button temporarily launches this offline profile server, runs one synthesis with no playback, prints the hottest kernels by total GPU time/call count/average time, then restores the selected live mode.

```text
voxgen.exe --server --mode xtx7900 --stream off --benchmark-profile
```

GPU timestamp readback remains **off during normal XTX streaming**. Cooperative matrices also remain opt-in via `--xtx-coopmat on`.

## Automatic demo benchmark block (v0.7.31)

After each completed synthesis variation, the demo appends a `--- Benchmark results ---` section to the main log. Streaming runs include first-PCM readiness, actual buffered playback start, adaptive reserve size, patch-arrival cadence, wall time and RTF. End-to-end RTF includes startup/prefill latency, while **streaming cadence headroom** compares actual inter-patch delivery against the nominal 160-ms patch deadline (adjusted for playback speed).

Example:

```text
--- Benchmark results ---
Mode: XTX 7900
Streaming: on
Input: 484 characters, 73 words
Variation: 1/1
Acoustic patches: 149
Generated audio: 23.84 s
Generation wall time: 22.151 s
First PCM ready: 3.479 s
Time to playback start: 3.606 s
Adaptive startup buffer: 2 patches (~320 ms at current speed)
RTF: 0.929
End-to-end throughput headroom: +7.1% (includes startup latency)
Patch delivery: avg 126.2 ms, max 157.1 ms, late >160.0 ms: 0/148
Streaming cadence headroom: avg +21.1%, worst +1.8% (PASS)
Seed: ...
XTX tuning: shared QKV + targeted barriers + residual-rms/swiglu + wave32 + subgroup reductions + x4 linear + prefill32-live; GPU profile off; coopmat off
-------------------------
```

## XTX 7900 stream-safe defaults (v0.7.31)

`--mode xtx7900` keeps the validated streaming subset: forced wave32, subgroup reductions, x4 F16/Q8 linear lanes, and 32-position live cross-engine text prefill batching (16 positions under offline timestamp profiling). GPU timestamp collection and cooperative matrices remain disabled by default because either can increase patch jitter. Use the dedicated offline `--benchmark-profile` path for timing work rather than profiling live audio.


- BaseLM component: `VoxCPM2-BaseLM-Q8_0.gguf` **or** `VoxCPM2-BaseLM-F16.gguf`
- Acoustic component: `VoxCPM2-Acoustic-F16.gguf`

Both components are loaded **simultaneously** for speech synthesis. Q8_0 vs F16 is the selectable BaseLM alternative; the Acoustic component is not an alternative to BaseLM.

There is no application-specific integration in this package. VoxGen owns its model loading, tokenizer, generation loop, WAV conditioning, PCM streaming, diagnostics and HTTP interface.

## GPU execution modes

`normal` is the default and uses the portable Vulkan shader set:

```text
voxgen.exe ... --mode normal
```

`xtx7900` is an explicit opt-in for the **AMD Radeon RX 7900 XTX**:

```text
voxgen.exe ... --mode xtx7900
```

The optimized mode refuses to start on a different selected GPU rather than silently falling back. It also requires compute subgroup arithmetic and a reported subgroup size of 32 or 64. At startup VoxGen prints the selected execution mode and subgroup size.

The XTX path requires `VK_EXT_subgroup_size_control` and forces a 32-lane subgroup for its tuned compute pipelines. v0.7.31 keeps the cooperative-matrix implementation and Vulkan timestamp profiler available for controlled experiments, but **does not enable either one by default**. Prefer `--benchmark-profile` for offline timing; use `--xtx-coopmat on` only for explicit cooperative-matrix experiments. `normal` does not enable XTX tuning and remains the reference implementation. Because this environment cannot execute the user's 7900 XTX, speed gains must be benchmarked on the target machine rather than assumed.

The same mode applies to server startup:

```text
voxgen.exe --server --mode xtx7900 --stream on
```

A model-lifecycle server remembers its startup mode and applies it when `POST /v1/models/load` creates the runtime. `/health` and `/v1/models/current` report the active mode.

## End-to-end pipeline

```text
text
  ↓ GGUF-native VoxCPM2 byte-fallback BPE tokenizer
reference WAV (optional) ── AudioVAE encoder ──┐
prompt WAV + transcript (optional) ────────────┤
                                               ↓
conditioning prefix → LocEnc → BaseLM → FSQ/fusion → ResidualLM
                                               ↓
                              current BaseLM/ResidualLM hidden
                                               ↓
                                     LocDiT + UnifiedCFM
                                  CFG-Zero* + Euler solver
                                               ↓
                                      one 4×64 latent patch
                                               ↓
                         stop predictor + autoregressive feedback
                                               ↓
                         full-sequence AudioVAE decoder
                         (compatibility rolling stream path)
                                               ↓
                                  native 48-kHz mono PCM
```

Each generated 4×64 latent patch represents **160 ms** of audio. Non-streaming synthesis decodes the complete generated latent sequence. On Windows, the wxDragon demo now consumes the compatibility streaming path as patches become available and queues them directly to the OS audio device; this removes sentence-duration-dependent playback startup. The compatibility streaming decoder defaults to a four-patch rolling context; a fully stateful AudioVAE streaming decoder remains a separate optimization/correctness milestone.

## Expressive speech and voice cloning modes

VoxGen exposes the VoxCPM2 conditioning choices as explicit modes rather than treating pitch/speed DSP as emotion control:

- **Auto/text prosody** — no control prefix; punctuation and sentence meaning are passed through verbatim and the model chooses prosody.
- **Controllable reference cloning** — `--clone-mode reference --reference-wav speaker.wav` preserves the reference timbre while allowing a natural-language `--control` instruction.
- **Ultimate cloning** — `--clone-mode ultimate` uses prompt audio plus its exact `--prompt-text`; when only one WAV is supplied, VoxGen reuses it as both prompt and reference conditioning for strong speaker/style similarity.
- **Continuation/auto compatibility** — `--clone-mode auto` preserves the older explicit `--prompt-wav`/`--prompt-text` behavior.

Native style control is textual. VoxGen prepends `(instruction)` to the target text before tokenization while preserving the target's punctuation and wording. Because prompt/Ultimate cloning already supplies the delivery style acoustically, VoxGen intentionally rejects `--control` together with prompt audio/text.

Post-generation **Speed**, **Pitch**, **Word spacing**, and **Gain** remain independent fine controls; they do not replace model-level expressive conditioning.

## CLI

Basic synthesis:

```bat
voxgen.exe ^
  --base-lm models\VoxCPM2-BaseLM-Q8_0.gguf ^
  --acoustic models\VoxCPM2-Acoustic-F16.gguf ^
  --text "Hello from VoxGen." ^
  --output-wav out.wav
```

Streaming is disabled unless explicitly enabled. For one-shot CLI synthesis use `--stream on` to exercise rolling AudioVAE output; in server mode it also enables `POST /v1/audio/speech/stream`:

```bat
voxgen.exe --server --host 127.0.0.1 --port 8091 --stream on
```

`--stream off` is the default. Bare `--stream` remains accepted as a compatibility alias for `--stream on`.

Speech output gain is linear: `--gain 1.0` is neutral, `--gain 1.40` is +40% amplitude, and `--gain 0` mutes emitted audio without changing inference. The server uses the CLI gain as its default, and individual `/v1/audio/speech` requests may override it with a JSON `gain` field.

Reference voice cloning:

```bat
voxgen.exe ^
  --base-lm models\VoxCPM2-BaseLM-Q8_0.gguf ^
  --acoustic models\VoxCPM2-Acoustic-F16.gguf ^
  --text "This sentence should use the reference timbre." ^
  --reference-wav speaker.wav ^
  --output-wav cloned.wav
```


Controllable expressive cloning:

```bat
voxgen.exe ^
  --base-lm models\VoxCPM2-BaseLM-Q8_0.gguf ^
  --acoustic models\VoxCPM2-Acoustic-F16.gguf ^
  --clone-mode reference ^
  --reference-wav speaker.wav ^
  --control "warm and genuinely pleased, conversational, with natural changes in emphasis" ^
  --text "Χαίρομαι πολύ που σε βλέπω." ^
  --output-wav warm.wav
```

Ultimate cloning from an expressive reference and its exact transcript:

```bat
voxgen.exe ^
  --base-lm models\VoxCPM2-BaseLM-Q8_0.gguf ^
  --acoustic models\VoxCPM2-Acoustic-F16.gguf ^
  --clone-mode ultimate ^
  --reference-wav expressive-reference.wav ^
  --prompt-text "Exact transcript of expressive-reference.wav" ^
  --text "Target sentence." ^
  --output-wav ultimate.wav
```

Generate three alternate performances with distinct deterministic seeds:

```bat
voxgen.exe ... --reference-wav speaker.wav --control "cheerful and warm" ^
  --text "Target sentence." --variations 3 --output-wav candidate.wav
```

This writes `candidate_v01.wav`, `candidate_v02.wav`, and `candidate_v03.wav`. `--cfg-value`, `--temperature`, `--inference-timesteps`, and `--seed` are visible aliases for the existing CFM generation controls; their normal defaults remain CFG 2.0, temperature 1.0, 10 timesteps, and the configured seed.

Continuation conditioning:

```bat
voxgen.exe ^
  --base-lm models\VoxCPM2-BaseLM-Q8_0.gguf ^
  --acoustic models\VoxCPM2-Acoustic-F16.gguf ^
  --prompt-wav prompt.wav ^
  --prompt-text "Exact transcript of prompt.wav. " ^
  --text "Continue speaking from here." ^
  --output-wav continuation.wav
```

Exercise the rolling streaming decoder:

```bat
voxgen.exe ^
  --base-lm models\VoxCPM2-BaseLM-Q8_0.gguf ^
  --acoustic models\VoxCPM2-Acoustic-F16.gguf ^
  --text "Streaming test." ^
  --stream on ^
  --output-wav stream-assembled.wav
```

The CLI streaming mode still writes the assembled result at exit; the HTTP streaming endpoint emits chunks live.

Tokenizer inspection:

```bat
voxgen.exe --base-lm models\VoxCPM2-BaseLM-Q8_0.gguf --tokenize "你好, VoxGen."
```

## HTTP compatibility interface

VoxGen supports both **startup model loading** and **API-managed model loading**.

Traditional startup loading remains valid:

```bat
voxgen.exe ^
  --base-lm models\VoxCPM2-BaseLM-Q8_0.gguf ^
  --acoustic models\VoxCPM2-Acoustic-F16.gguf ^
  --host 127.0.0.1 --port 8091
```

For applications that need selectable model paths, start an empty server instead:

```bat
voxgen.exe --server --host 127.0.0.1 --port 8091
```

Then load or replace the model explicitly:

```http
POST /v1/models/load
Content-Type: application/json

{
  "base_lm": "D:/models/VoxCPM2-BaseLM-Q8_0.gguf",
  "acoustic": "D:/models/VoxCPM2-Acoustic-F16.gguf",
  "base_format": "auto",
  "max_context": 8192
}
```

`base_lm` and `acoustic` are **server-side filesystem paths**. `base_format` may be `auto`, `q8_0`, or `f16`. `gpu` and `max_context` are optional per-load overrides.

Model lifecycle endpoints:

- `GET /v1/models/current` — report the exact currently loaded paths, format, GPU, context, and readiness;
- `POST /v1/models/load` — unload the old runtime and load the supplied paths;
- `POST /v1/models/unload` — release the current model/VRAM;
- `GET /v1/models` — compatibility model listing plus current selection.

Reloads are serialized against speech inference. VoxGen releases the old runtime before allocating the replacement, avoiding a temporary two-model VRAM spike. If a replacement load fails after the old runtime has been released, the server remains alive but unloaded so the client can choose another path.

Speech endpoints use the **currently loaded model**; model paths are deliberately not repeated on each TTS request:

- `POST /v1/audio/speech` — complete WAV or raw float PCM response;
- `POST /v1/audio/speech/stream` — chunked WAV/48-kHz f32 stream; one new 160-ms PCM chunk per acoustic patch;
- `GET /v1/health`;
- `GET /v1/voxgen/diagnostics`.

Minimal speech JSON:

```json
{
  "input": "Hello from VoxGen.",
  "response_format": "wav"
}
```

Expressive voice-cloning fields:

```json
{
  "input": "Target speech.",
  "reference_audio": "<base64 WAV>",
  "clone_mode": "reference",
  "control": "warm, conversational, slightly cheerful",
  "inference_timesteps": 10,
  "cfg_value": 2.0,
  "temperature": 1.0,
  "seed": 42
}
```

For Ultimate cloning, set `"clone_mode": "ultimate"`, supply `prompt_text`, and provide prompt/reference audio. VoxGen may reuse one supplied WAV for both roles. `control` and Ultimate/prompt conditioning are mutually exclusive.

For local use, `reference_audio_path` and `prompt_audio_path` are also accepted as VoxGen extensions. Base64 data-URI payloads are accepted in addition to bare base64.

## Stop prediction

The autonomous loop evaluates the VoxCPM2 acoustic GGUF tensors:

```text
stop_predictor.linear1.weight / bias
       ↓
     SiLU
       ↓
stop_predictor.linear2.weight
       ↓
continue / stop logits
```

The stop class is accepted only after `--min-steps` (default 2); `--max-steps` (default 200) is the hard safety ceiling.

## Performance behavior

The normal speech path intentionally differs from the diagnostic smoke paths:

- BaseLM and ResidualLM hidden states remain device-local;
- text/audio prefix prefill uses GPU-only LM steps;
- generated-patch feedback uses GPU-only LocEnc → BaseLM → ResidualLM steps;
- CFM's conditional/unconditional velocity work stays on Vulkan;
- only the generated 256-float patch, two stop logits, and decoded PCM need host-visible control/output transfers;
- no neural-network CPU fallback is available.

The first optimization target after correctness validation is persistent/captured command graphs around the repeated LocDiT/CFM and rolling AudioVAE decoder work.

## Building

Install Rust 1.87+ and the Vulkan SDK / `glslc`. The project root contains the paired build and source-clean launchers:

```text
build_voxgen.bat     Windows master build launcher
build_voxgen.sh      Linux master build launcher
clean_source.bat     Windows project-local source cleaner
clean_source.sh      Linux project-local source cleaner
```

Windows:

```bat
build_voxgen.bat
```

Linux:

```bash
chmod +x build_voxgen.sh build_linux/*.sh
./build_voxgen.sh
```

Both master launchers delegate to the platform directories. All Windows `.bat` smoke/build helpers live in `build_windows/`; all Linux `.sh` equivalents live in `build_linux/`. `build.rs` compiles every `.comp` shader with `glslc` for Vulkan 1.2 before compiling VoxGen.

Supported master build modes are `release` (default), `debug`, `check`, and `clean`; append `--no-probe` to release/debug to skip the post-build Vulkan device probe.

Linux model scripts default to `$VOXGEN_ROOT/models` and support `VOXGEN_MODEL_DIR`, `VOXGEN_BASE_Q8`, `VOXGEN_BASE_F16`, and `VOXGEN_ACOUSTIC`. Windows uses the same environment variable names and falls back to the historical `C:\Software\VoxCPM-Q8\models` path when no project-local model directory exists.

This package was generated in an environment without the Rust/Vulkan SDK toolchain, so the included validation is static/source-level. The first target-machine action should be a native build followed by the platform smoke scripts.

## RX 7900 XTX pass 3: cross-engine prefill batching + x4 linear lanes

When `--mode xtx7900` is active, VoxGen detects contiguous text-prefix positions and records BaseLM followed by ResidualLM for up to **16 positions in one Vulkan command buffer**. Barriers preserve the exact autoregressive KV-cache ordering, but the CPU no longer submits and waits after every BaseLM and ResidualLM token step. Normal mode intentionally keeps the previous sequential submission path for A/B correctness and performance comparison. GPU profiling reports these batches as `prefill.cross_engine_batch`.

The XTX variants of `matvec`, `linear_bias`, `fusion_linear`, and `seq_linear_bias` now consume **four F16 or Q8 values per lane iteration**. F16 paths unpack two packed-half words per iteration; Q8 paths reuse one block scale across four adjacent quantized values. This is especially relevant to the Q8 BaseLM GEMV hot path.

## wxDragon demo (v0.7.31)

A native desktop demo is included in [`demo/`](demo/). It keeps the two-textbox layout and model controls and adds **Style / emotion**, **Intensity**, **Custom instruction**, **Clone mode**, **Reference transcript**, preset-specific emotional reference WAVs, **Variations**, **CFG**, **Temperature**, and **CFM steps** alongside the existing **Word spacing**, **Speed**, **Pitch**, and **Gain** controls. The demo starts/uses VoxGen's local HTTP server and manages model paths through `/v1/models/load`. On Windows it consumes `/v1/audio/speech/stream` incrementally and queues PCM16 blocks through WinMM `waveOut`; v0.7.31 adds an adaptive two-to-four-patch startup reserve before the first WinMM submission. Speed and pitch remain live during streamed playback.

The top model area keeps the compact **Engine mode** selector plus one **Load VoxCPM2** action. A separate **Diagnostics** row exposes **Benchmark Normal vs XTX** and **Profile XTX**. The A/B action uses one identical seed and disables playback; the profile action temporarily launches an offline timestamp-enabled XTX server and restores the selected live mode afterward. The lower input box remains unchanged after synthesis so the same sentence can immediately be replayed, benchmarked, profiled, or edited.

The demo reads/writes `settings.cfg` **beside the running demo executable** and persists BaseLM/Acoustic paths, voice sample, **engine mode (`normal`/`xtx7900`)**, style/intensity/custom control, clone mode/transcript, per-preset emotional references, variations, CFG/temperature/timesteps, word spacing, speed, pitch, gain, and its stream preference. The demo defaults `stream=on`; if `voxgen.exe` (Windows) or `voxgen` (Linux) is placed beside the demo executable, that adjacent engine binary is used before project `target/release`/`target/debug` fallbacks.

See `demo/README.md` for Windows/Linux build and model-path details.