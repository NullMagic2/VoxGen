# VoxGen wxDragon demo

## Native playback DSP (v0.7.40)

The demo no longer contains its own speed/pitch DSP algorithm. Its live Speed and Pitch controls call the shared `voxgen::playback_dsp` module from the engine crate, so demo playback and VoxGen's HTTP API use the same sinc + normalized-correlation speech WSOLA math. Live slider changes remain supported: the demo is only a thin control/playback adapter over the shared stateful processor.

## Cleaning downloaded/build files (v0.7.39)

Run `clean_source.bat` on Windows or `./clean_source.sh` on Linux from this `demo/` folder. The wrapper invokes the project-level cleaner, so both the demo and engine trees are cleaned together while their final debug/release binaries are preserved. Project-local `models/`, download/cache directories, Cargo build intermediates, generated smoke-test outputs, and generated lockfiles are removed. Global Cargo caches and model directories outside the VoxGen source tree are never touched.


This is a deliberately small native desktop front end for VoxGen.

## UI

The demo intentionally keeps the two main text boxes from the original design:

- the large upper text box is a read-only activity/status log;
- the lower text box is where you enter the text to synthesize.

Controls:

- **Select BaseLM component...** — choose `VoxCPM2-BaseLM-Q8_0.gguf` or `VoxCPM2-BaseLM-F16.gguf`;
- **Select Acoustic component...** — choose `VoxCPM2-Acoustic-F16.gguf`;
- **Load VoxCPM2** — asks the running VoxGen server to unload the current model and load the selected server-side GGUF paths;
- **Select voice sample...** — choose a preset reference WAV; the Neutral preset is the canonical fallback voice anchor;
- **Style / emotion** — choose the destination managed style (or Custom);
- **Intensity** — choose the destination managed intensity;
- **Managed pace %** — choose the destination speaking pace used by automatic continuity;
- **Continuity** — Continuous or Hard cut. The demo sends only the destination plus its continuity id; VoxGen looks up the previous successful state and realizes any required style/intensity/pace transition internally;
- **Custom instruction** — free-form VoxCPM2 delivery instruction used by the Custom preset;
- **Clone mode** — **Controllable reference** uses the style instruction with reference timbre; **Ultimate cloning** uses the reference/prompt audio plus its exact transcript and disables textual control;
- **Reference transcript** — exact transcript required by Ultimate cloning;
- **Set preset reference... / Clear preset ref** — associates a WAV with the selected style. A preset-specific WAV wins when present; otherwise VoxGen anchors to the Neutral reference when configured. A configured-but-missing Neutral/default WAV causes an explicit error instead of zero-shot voice invention;
- **Variations** — generate 1–3 alternate performances with distinct seeds. Windows plays the candidates sequentially; complete-WAV mode concatenates them with a short comparison gap;
- **CFG (%)** — model conditional-guidance control, default 200% (= 2.0);
- **Temperature (%)** — acoustic sampling temperature, default 100% (= 1.0);
- **CFM steps** — inference timesteps, default 10;
- **Word spacing (ms)** — numeric 0–100 ms control, default 30 ms. It extends only short low-energy gaps likely to be between words;
- **Speed (%)** — live 50–200% tempo control, default 100%;
- **Pitch (semitones)** — live -12 to +12 semitone control, default 0;
- **Gain (%)** — 0–400% request gain, default 100%;
- **Speak** — synthesize and play the lower-box text. The typed text remains in the box after synthesis;
- **Stop** — immediately flush queued playback and cancel the active speech request. GPU generation stops cooperatively after the current safe acoustic-patch/GPU operation completes.

The demo uses VoxGen's public model-lifecycle and speech APIs. It does not contain another inference implementation.

### Streaming fidelity / anger artifact mitigation (v0.7.38)

- The compatibility rolling AudioVAE decoder now retains at least 6 latent patches of context, even if an older client requests only 3 or 4.
- This avoids repeatedly replacing required causal history with zero-padding, which can sound brittle, metallic, or cracked on highly expressive speech.
- Fresh demo gain now starts at 100% rather than 140%.
- Angry style recipes use controlled tension, timing, and phrase-level emphasis instead of continuous loudness/shouting.


### Mid-generation Stop (v0.7.37)

- The Stop button is enabled only while a Speak request is active.
- Windows streaming calls `waveOutReset` immediately, so already queued PCM is flushed instead of continuing to play after Stop.
- The demo also posts `/v1/audio/speech/cancel`; this endpoint deliberately bypasses the model-lifecycle inference gate so it remains responsive while synthesis owns that gate.
- Runtime cancellation is cooperative: in-flight Vulkan work is allowed to complete, then VoxGen refuses to start the next acoustic patch. This keeps GPU state safe while bounding generation-stop latency to approximately one patch.
- A cancelled request is treated as normal control flow rather than a synthesis failure, and the next Speak starts with a fresh cancellation state.

### Neutral voice anchoring (v0.7.36)

- A usable preset-specific reference is preferred; if it is absent, `emotion_reference.neutral` becomes the canonical fallback identity.
- A configured Neutral/default WAV that is missing causes synthesis to stop with an explicit diagnostic rather than silently switching to zero-shot.
- The same reference resolver is used by Speak, diagnostics, and background prewarming.
- Zero-shot remains available only when no reference/default anchor was configured at all.

### Low-latency streaming startup (v0.7.35)

- Local reference WAVs are passed by stable path instead of base64 on every Speak.
- The selected reference is AudioVAE-prewarmed after model load/selection so the first synthesis can hit the conditioning cache.
- At 100% or slower playback, WinMM starts from the first acoustic patch; faster speeds retain extra reserve.
- Streaming exposes PCM before the stop-head readback and XTX live text prefill uses 32-position submit batches.

### QKV + targeted synchronization power pass (v0.7.34)

Normal and XTX 7900 now share one-dispatch Q/K/V projection kernels in BaseLM and ResidualLM, and LocEnc/LocDiT use the same strategy unless the explicit XTX cooperative-matrix experiment is enabled. Transformer-stage synchronization is narrowed to the actual buffers crossing each dependency. The positive CFG mu vectors remain resident for the whole solve, while the unconditional LocDiT pass writes zero mu tokens directly; this removes the old buffer fills plus the per-step save/restore copies they required. The 256-thread workgroup defaults remain unchanged until target-hardware profiling proves a different size is at least as fast.

### Shared power-efficiency kernels (v0.7.33)

Both demo execution modes use the engine's shared residual+RMSNorm and Gate+Up+SwiGLU fusions. Normal mode uses portable Vulkan compute shaders; XTX 7900 uses subgroup-tuned variants. LocalDiT preserves the explicit cooperative-matrix path when that optional XTX feature is enabled. No clock, voltage, or power-limit changes are made by the demo.

### Live speed recovery (v0.7.32)

Changing Speed during streamed playback now resets WSOLA rate history and bounds the WinMM pending queue, so raising Speed after a slowdown takes effect on the next near-playback chunks instead of being masked by old pre-rendered slow audio. Pitch changes additionally reset resampler phase; returning to 100% / 0 semitones returns directly to the neutral dry path.

### Realtime speed/pitch quality (v0.7.31)

The demo now uses a speech-oriented WSOLA + 24-tap Lanczos-windowed sinc pipeline instead of the previous 1024-point phase vocoder. WSOLA changes tempo by aligning similar waveform segments before overlap-add, which preserves consonant attacks and avoids the metallic/underwater coloration common on speech. Pitch is changed by band-limited resampling, with WSOLA compensating the duration so pitch and speed remain independent. The exact 100% / 0-semitone setting is still a dry PCM bypass.


### Live speed and pitch

The **Speed (%)** and **Pitch (semitones)** controls now use a speech-oriented WSOLA + band-limited resampling pipeline. WSOLA performs time-scale modification in the time domain by matching similar waveform regions before each overlap, which keeps consonants and vocal attacks much more coherent than the previous small-window phase vocoder. Pitch is shifted with a 24-tap Lanczos-windowed sinc resampler; a compensating WSOLA tempo factor restores the requested duration, so pitch and speed remain independent.

- neutral: **100% / 0 semitones**; the demo plays the original PCM directly, so the default voice remains uncolored;
- speed range: **50–200%**;
- pitch range: **-12 to +12 semitones**;
- speed and pitch are independent (`WSOLA tempo = speed / pitch_factor`);
- upward pitch shifts apply an anti-alias cutoff in the sinc resampler;
- the DSP graph is kept warm while neutral, and a 10 ms transition crossfade hides alignment changes when live DSP is engaged or disengaged.

On Windows the controls stay enabled while `/v1/audio/speech/stream` is active. New values are picked up on subsequent streamed audio, so long utterances can still be adjusted while they are playing. Very large pitch shifts will naturally change vocal formant character, but modest adjustments such as ±1–3 semitones should no longer have the characteristic metallic/phasey effect of the old implementation.

## Speech pacing / Greek

Some VoxCPM2 voice clones speak Greek more rapidly than the same speaker would naturally. The demo therefore defaults to **+30 ms Word spacing**. Set **Word spacing (ms)** anywhere from **0 to 100 ms**; **30 ms** is the default.

This is deliberately not a sample-rate slowdown: changing the output rate would lower pitch. Instead, the playback processor detects short adaptive low-energy gaps (8–120 ms) after voiced material and extends only those gaps. Long punctuation pauses and trailing silence are left unchanged. The processor is stateful across Windows streaming chunks, so a word gap that crosses a 160-ms acoustic patch boundary is still handled continuously.



### Adaptive live buffer + diagnostics (v0.7.31)

Windows streaming now waits for a small adaptive startup reserve before the first WinMM submission: two acoustic patches normally, three when early cadence approaches the playback deadline, and up to four when an early patch is already late. The benchmark log distinguishes **First PCM ready** from **Time to playback start** and reports the chosen reserve.

The demo also adds a compact **Diagnostics** row:

- **Benchmark Normal vs XTX** — runs the current text once in each engine mode with one identical seed and no playback, then reports comparable wall/engine/first-PCM/RTF results and restores the selected mode.
- **Profile XTX** — temporarily relaunches VoxGen with `--mode xtx7900 --stream off --benchmark-profile`, resets the GPU counters, runs one offline synthesis, prints the hottest Vulkan kernels, and restores the selected live server afterward.

The controlled reports include the BaseLM/Acoustic filenames, reference path (or zero-shot), inference settings, and a stable text fingerprint so pasted benchmark blocks are self-describing.

### Engine execution mode (v0.7.31)

The top model row has an explicit **Engine mode** selector with the compact labels **Normal** and **XTX 7900**, corresponding to `--mode normal` and `--mode xtx7900`. The selection is persisted as `mode=normal` or `mode=xtx7900` in the `settings.cfg` beside the demo executable.

The **Load VoxCPM2** button is now the single model/mode action. If the selected **Engine mode** differs from the running VoxGen server, the button safely restarts VoxGen in the selected mode and then reloads the selected BaseLM/Acoustic files. If the mode is unchanged, it keeps the current server and only reloads the models. If port 8091 contains VoxGen left behind by an older demo/server session, the load action first verifies `/health` identifies the listener as VoxGen, then takes it over when a restart is required. v0.7.31+ uses a loopback-only graceful shutdown endpoint; on Windows, older servers are recovered by resolving the listening PID and verifying its process image contains `VoxGen` before `taskkill`. A legacy VoxGen `/health` response with no `mode` field is treated as Normal.

`xtx7900` is intentionally rejected by VoxGen when the selected device is not an AMD Radeon RX 7900 XTX.

### XTX 7900 stream-safe defaults (v0.7.31)

The demo launches `XTX 7900` with `--gpu-profile off --xtx-coopmat off`. This keeps the pass-3 subgroup/x4/prefill optimizations while avoiding synchronous timestamp-query readback and the experimental cooperative-matrix path during real-time playback.


### No reference sample / zero-shot generation (v0.7.31)

If **Controllable reference** is selected but neither the current preset nor the global fallback has a valid WAV, the demo now sends `clone_mode=auto` and performs native zero-shot VoxCPM2 generation. The log explicitly says `No reference sample selected: using zero-shot generation.` Ultimate cloning remains strict and still requires reference audio and its exact transcript.

### Automatic benchmark results (v0.7.31)

Every completed synthesis variation appends a `--- Benchmark results ---` block to the demo log. It records the selected **Normal** or **XTX 7900** mode, input size, acoustic patch count/audio duration, request wall time, RTF, seed, and the active engine path. Windows streaming additionally reports **First PCM ready**, buffered **Time to playback start**, adaptive startup reserve, average/max inter-patch delivery time, late-patch count, and cadence headroom. End-to-end RTF includes startup latency; cadence headroom is the more direct stutter diagnostic once playback has begun.
