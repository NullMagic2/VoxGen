# VoxGen iteration 6 validation notes

## Scope

Iteration 6 implements VoxCPM2 AudioVAE V2 in both directions and connects WAV conditioning to the iteration-4 conditioning pipeline. It deliberately stops short of the autonomous autoregressive generation/stop/streaming loop planned for iteration 7.

## Upstream-equivalence points checked

### AudioVAE geometry

Expected VoxCPM2 V2 configuration:

- encoder dim 128;
- latent dim 64;
- decoder dim 2048;
- encoder rates `[2, 5, 8, 8]`;
- decoder rates `[8, 6, 5, 2, 2, 2]`;
- encoder sample rate 16,000 Hz;
- decoder output 48,000 Hz;
- depthwise mode enabled;
- decoder sample-rate boundaries `[20000, 30000, 40000]`;
- `scale_bias` sample-rate conditioning;
- no decoder noise block;
- patch size 4.

Derived contracts:

- encoder hop = 640 samples/latent frame;
- decoder hop = 1,920 samples/latent frame;
- one four-frame patch = 2,560 input samples = 160 ms;
- one four-frame patch = 7,680 output samples = 160 ms.

### Encoder

VoxGen implements:

1. causal stem Conv1d 1 -> 128, k7;
2. four encoder blocks, each with three residual units at dilation 1/3/9;
3. depthwise k7 convolution in the residual units;
4. Snake activations;
5. causal strided convolution with kernel `2*stride`;
6. channel doubling after each block;
7. final causal `fc_mu` k3 from 2048 -> 64.

`fc_logvar` is validated in the GGUF but intentionally not executed because VoxCPM2 inference consumes `mu`.

### Decoder

VoxGen implements:

1. depthwise latent stem 64 -> 64 k7;
2. 1x1 projection 64 -> 2048;
3. sample-rate scale/bias before every decoder block;
4. six causal transposed-convolution upsample blocks;
5. output-channel halving per block;
6. three depthwise residual units at dilation 1/3/9 after every upsample;
7. final Snake -> causal k7 -> tanh;
8. 48-kHz sample-rate bucket index 3.

### Causal padding

`CausalConv1d` uses only left padding: `2*padding-output_padding`.

`CausalTransposeConv1d` crops the right tail by the same amount. VoxGen's transposed-convolution shader directly computes the retained output range.

### GGUF weights

The llama.cpp-omni VoxCPM2 converter:

- prefixes VAE tensors with `audio_vae.`;
- strips ordinary `.weight` suffixes from the GGUF tensor name;
- writes biases as `.bias`;
- writes Snake parameters as `.alpha`;
- merges weight-normalized `weight_g` and `weight_v` into one tensor before export;
- exports the F16 acoustic checkpoint weights as F16.

VoxGen validates these exact names and shapes at startup.

### Conditioning audio alignment

Official VoxCPM2 `_encode_wav` pads to `patch_size * encoder_hop = 2560` samples:

- reference-only speaker audio: right padding;
- continuation/prompt audio: left padding.

VoxGen reproduces this role-dependent alignment before VAE encoding.

## Native preprocessing difference

The official implementation loads/resamples audio through librosa (currently high-quality soxr by default). VoxGen intentionally has no Python inference/preprocessing dependency and uses a native windowed-sinc host resampler.

It does reproduce librosa's fixed output-length convention:

`ceil(input_samples * target_sr / source_sr)`.

Use mono 16-kHz WAV for the cleanest VAE numerical parity test because resampling is then bypassed.

## GPU residency

Learned AudioVAE operations run on Vulkan. WAV parsing/downmix/resampling is host preprocessing only.

AudioVAE reuses the acoustic model's already-resident GGUF data buffer. It allocates three lazily-grown F32 device-local activation buffers. The dynamic allocation is surfaced in runtime diagnostics.

## Deterministic smoke inputs

`make_test_vae_inputs.py` creates:

- `test_vae_input.wav`: mono 16-kHz PCM16, 3,000 samples (deliberately not divisible by 2,560);
- `test_vae_pcm16k.f32`: the synthetic 3,000-sample signal as raw f32 PCM for a preprocessing-free encoder path;
- `test_vae_latents.f32`: exactly 256 floats (`4 x 64`).

Expected alignment for the WAV is 5,120 samples, producing 8 latent frames / 2 patches and, on full decode, 15,360 samples at 48 kHz.

## Static validation performed on packaging host

`validate_iteration6.py` checks:

- Cargo version/dependency markers;
- all Rust modules are present;
- every unique embedded SPIR-V name has a matching GLSL source;
- every GLSL source is represented by an embedded SPIR-V target;
- AudioVAE shader bindings/push-contract markers;
- convolution/transposed-convolution indexing against CPU reference implementations;
- derived encoder/decoder length contracts;
- right/left patch-padding contracts;
- test vector dimensions;
- no Reading Companion integration files/references;
- step-7 features remain explicitly unavailable.

## Environment limitation

This packaging host has no Rust compiler or Vulkan shader compiler installed. Therefore:

- `cargo build` is not claimed;
- GLSL -> SPIR-V compilation is not claimed;
- native Vulkan execution is not claimed;
- numerical parity against PyTorch is not claimed yet.

Those checks must be performed on the target Vulkan machine with `build_voxgen.bat` on Windows or `./build_voxgen.sh` on Linux and the platform smoke tests.

## Iteration-7 boundary

Still needed for complete VoxGen TTS/voice cloning:

- tokenizer/text-facing synthesis entry point if not already supplied by caller;
- autoregressive repeated BaseLM -> FSQ/ResidualLM -> CFM patch loop;
- stop predictor and min/max termination behavior;
- retry/bad-case policy as desired;
- stateful AudioVAE streaming decoder so successive patches retain causal convolution state without full-prefix re-decode;
- chunked 48-kHz PCM stream and compatible HTTP/API surface;
- end-to-end latency/RTF profiling and RDNA3 kernel fusion.
