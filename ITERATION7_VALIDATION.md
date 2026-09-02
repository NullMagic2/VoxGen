# VoxGen iteration 7 validation notes

## Scope

Version 0.7.2 connects the already implemented VoxCPM2 stages into complete text-to-waveform inference:

1. GGUF-native byte-fallback BPE tokenization;
2. reference/prompt WAV -> AudioVAE latent conditioning;
3. text/audio prefix prefill into BaseLM and ResidualLM caches;
4. LocDiT/UnifiedCFM generation of one 4x64 patch per autoregressive step;
5. acoustic stop predictor (`2048 -> 2048 -> 2`, SiLU);
6. LocEnc feedback of each generated patch into BaseLM/ResidualLM;
7. three-patch rolling AudioVAE decode and 48-kHz PCM emission;
8. non-streaming and chunked HTTP speech APIs.

## Upstream contracts mirrored

The step-7 loop follows the OpenBMB VoxCPM inference ordering: CFM patch generation from the current LM states; append/stream the patch; stop-head decision on the current BaseLM state; if continuing, LocEnc the patch and advance both autoregressive language-model caches.

The default stop minimum is 2 and the default streaming latent context is 3 patches. One VoxCPM2 patch is 4 latent frames and decodes to 7,680 samples at 48 kHz (160 ms).

For prompt continuation, the rolling AudioVAE decode begins with up to the last `streaming_prefix_len - 1` prompt patches. Those context samples are trimmed from a complete non-streaming response and only the newest 7,680 samples are emitted for each streaming generation step.

## Tokenizer

The BaseLM GGUF parser now retains `tokenizer.ggml.tokens` and `tokenizer.ggml.merges`. VoxGen implements the VoxCPM2 tokenizer internally:

- prepend `▁`;
- replace spaces with `▁`;
- BPE merges using GGUF merge rank;
- `<0xXX>` UTF-8 byte fallback;
- GGUF BOS/EOS/UNK metadata;
- VoxCPM2 multi-character CJK token expansion.

No Python/Hugging Face tokenizer is required at runtime.

## Stop predictor

The acoustic validator requires:

- `stop_predictor.linear1.weight`: `[2048,2048]`, F16;
- `stop_predictor.linear1.bias`: `[2048]`, F16/F32;
- `stop_predictor.linear2.weight`: `[2048,2]`, F16.

Execution remains on Vulkan through the second linear projection; only the two logits are copied to the host.

## Generation fast path

Iteration 7 adds non-diagnostic ResidualLM methods that avoid the checksums/readbacks used by the iteration-3 smoke paths. Prefix text, prefix audio and generated-patch feedback therefore keep the 2048-D hidden tensors on the GPU. The remaining intentional control/output transfers are the generated 256-float latent patch, two stop logits and PCM output.

## HTTP

`POST /v1/audio/speech` accepts OpenAI-style `input`, `response_format`, `reference_audio`, `seed`, `cfg_value`, `inference_timesteps`, `max_steps`, and `temperature`, plus VoxGen prompt/continuation fields. `reference_audio` and `prompt_audio` accept raw base64 or a data URI. Local path extensions are available for standalone use.

`POST /v1/audio/speech/stream` uses HTTP chunked transfer and a streaming float-WAV header. It emits each new 160-ms f32 PCM chunk immediately after rolling AudioVAE decode.

## Static validation performed here

`validate_iteration7.py` verifies:

- Cargo version/dependencies;
- iteration-7 runtime status;
- stop tensor contract and execution path;
- tokenizer GGUF arrays and specialized tokenizer code;
- autoregressive/streaming generation path;
- HTTP speech/stream routes and cloning inputs;
- all 34 Rust-embedded SPIR-V names exactly match all 34 GLSL compute sources;
- Rust delimiter consistency across all source modules;
- presence of four step-7 smoke launchers;
- standalone project isolation.

It reports 12 Rust modules and 34 Vulkan compute shaders.

## Native validation boundary

This build environment does not contain `cargo`/`rustc` or `glslc`, so a native Rust + SPIR-V compilation and numerical GPU comparison cannot be claimed here. On Windows, run:

```bat
build_voxgen.bat
build_windows\smoke_tts.bat
build_windows\smoke_tts_stream.bat
build_windows\smoke_voice_clone_reference.bat speaker.wav
build_windows\smoke_voice_clone_continuation.bat prompt.wav "Exact prompt transcript. "
```

On Linux, run:

```bash
./build_voxgen.sh
./build_linux/smoke_tts.sh
./build_linux/smoke_tts_stream.sh
./build_linux/smoke_voice_clone_reference.sh speaker.wav
./build_linux/smoke_voice_clone_continuation.sh prompt.wav "Exact prompt transcript. "
```

The first native validation should compare F16 BaseLM output against the upstream PyTorch/C++ implementation before evaluating Q8_0 performance.
