# VoxGen

**VoxGen is a lightweight, Python-free Rust inference engine for VoxCPM2, built with a strong focus on AMD GPUs, Vulkan, and RX 7900 XTX optimization.**

VoxGen reimplements the VoxCPM2 inference path as a native Rust/Vulkan runtime instead of relying on the standard Python/PyTorch stack. The goal is simple: make high-quality local speech synthesis faster, leaner, easier to deploy, and especially well optimized for AMD hardware.

## Why VoxGen?

Most modern TTS runtimes are designed around Python and CUDA-first machine-learning frameworks. VoxGen takes a different approach.

It is designed as a **native inference engine** with explicit control over:

- GPU memory allocation
- Vulkan compute pipelines
- shader specialization
- synchronization and barriers
- streaming behavior
- low-latency PCM delivery
- voice-reference caching
- cancellation
- hardware-specific tuning

The result is a runtime that can be optimized around the actual characteristics of the target GPU instead of depending on general-purpose framework behavior.

## Key goals

### Lightweight native runtime

VoxGen is written in **Rust** and does not require Python, PyTorch, Triton, or a virtual environment at runtime.

This provides a much simpler deployment model:

```text
voxgen.exe + model files + Vulkan driver
```

instead of a large Python ML software stack.

### AMD-first optimization

VoxGen is developed with particular attention to **AMD Radeon GPUs**.

The engine uses Vulkan compute directly and includes a dedicated **RX 7900 XTX / RDNA3 execution mode** with optimizations such as:

- Wave32-oriented kernels
- subgroup reductions
- fused Q/K/V projections
- fused SwiGLU paths
- fused residual + RMSNorm operations
- targeted Vulkan synchronization
- optimized prefill batching
- reduced intermediate-buffer traffic
- persistent conditioning buffers
- XTX-specific shader paths

The XTX mode is intentionally separate from the portable Normal mode, allowing aggressive RDNA3 tuning without sacrificing compatibility with other Vulkan-capable GPUs.

## Performance focus

VoxGen is built around measurement rather than guesswork.

The demo and runtime include profiling for:

- Real-Time Factor (RTF)
- time to first PCM
- playback-start latency
- acoustic patch cadence
- late streaming patches
- streaming headroom
- per-kernel GPU timing
- Normal vs XTX A/B benchmarks
- deterministic seed reporting
- text hashing
- hot-kernel identification

In internal RX 7900 XTX testing, the dedicated XTX path has shown roughly **13% lower inference time than the portable Normal path** on representative workloads.

Optimization work has also reduced observed peak board power from roughly **360 W to around 250 W** on the development RX 7900 XTX while retaining the high-performance execution path.

> Performance and power figures are workload-, driver-, clock-, model-, and system-dependent. Treat these as development-machine measurements rather than guaranteed results.

## Low-latency streaming

Streaming is a first-class feature rather than an afterthought.

VoxGen supports:

- incremental acoustic-patch delivery
- immediate PCM playback
- adaptive startup buffering
- reference-conditioning prewarming
- cached voice-reference features
- live playback speed control
- playback pitch control
- immediate Stop / cancellation
- server-side generation cancellation at safe patch boundaries

Recent startup work has reduced first-play latency substantially, with playback beginning as soon as the first PCM block is available.

## Voice identity anchoring

A major design goal is **voice consistency**.

If a neutral reference sample is configured, VoxGen treats it as the canonical speaker anchor.

Reference selection follows this policy:

```text
requested style/emotion clip
        ↓
neutral reference
        ↓
default/legacy reference
        ↓
zero-shot only when no voice anchor is configured
```

If a configured neutral/default reference is missing, VoxGen can fail explicitly instead of silently falling back to an unrelated hallucinated speaker.

This makes the engine better suited to persistent characters, narration, reading assistants, and applications where maintaining the same voice matters more than unconstrained zero-shot generation.

## Memory-conscious design

VoxGen avoids much of the runtime overhead associated with a general-purpose Python ML framework.

The current configuration supports a **Q8_0 BaseLM** together with the **F16 acoustic model**, significantly reducing model memory requirements compared with a full BF16 runtime.

The native engine also keeps direct control over:

- KV-cache allocation
- scratch buffers
- staging buffers
- model mappings
- temporary tensors
- shader workspaces

This makes VoxGen particularly attractive for systems where VRAM efficiency matters.

## Portable Normal mode + optimized XTX mode

VoxGen currently provides two main execution profiles:

### Normal

A portable Vulkan path intended to work across a wider range of compatible GPUs.

### XTX 7900

A specialized path for the **Radeon RX 7900 XTX**, tuned around RDNA3 characteristics and the exact shapes used by VoxCPM2 inference.

This lets VoxGen remain portable while still taking advantage of device-specific optimizations where they matter.

## Profiling-driven development

VoxGen includes an offline GPU profiler that reports hot kernels and total GPU time.

Example profiling output can identify bottlenecks such as:

```text
local.seq_swiglu
local.seq_linear
prefill.cross_engine_batch
local.seq_qkv
```

This allows optimization effort to focus on the kernels that actually dominate runtime instead of broadly rewriting code with little measurable benefit.

## Why Rust + Vulkan?

Rust gives VoxGen:

- native performance
- predictable memory ownership
- strong safety guarantees
- straightforward multithreading
- no interpreter dependency
- simple native deployment

Vulkan gives VoxGen:

- direct access to AMD GPUs
- cross-vendor portability
- explicit synchronization
- explicit memory control
- custom compute shaders
- hardware-specific specialization without requiring CUDA

Together, they make it possible to build a TTS engine around the GPU rather than around the assumptions of a general ML framework.

## Current status

VoxGen is an actively developed experimental inference engine.

Current areas of development include:

- sustained real-time streaming cadence
- further RDNA3 shader optimization
- assembly-guided ISA analysis
- lower-latency text/reference prefill
- adaptive GPU profiles
- improved performance-per-watt
- more aggressive memory optimization
- stronger voice/prosody consistency

## Long-term direction

VoxGen began as a native VoxCPM2 inference engine, but the architecture is intentionally modular.

Longer term, the project can evolve toward independently replaceable components for:

- text/phoneme encoding
- prosody prediction
- speaker modeling
- acoustic generation
- vocoding

That opens the possibility of moving beyond a single upstream TTS architecture while keeping the same native Rust/Vulkan runtime philosophy.

## Philosophy

VoxGen is built around a few principles:

> **Native over interpreted.**  
> **Explicit over opaque.**  
> **Measured over assumed.**  
> **Portable by default, hardware-specific when it matters.**  
> **Low latency, low overhead, consistent voice identity.**

## Credits and model compatibility

VoxGen is an independent inference-engine project designed to run VoxCPM2-compatible model components.

VoxCPM2 is developed by **OpenBMB**. VoxGen is not an official OpenBMB project and is not affiliated with or endorsed by OpenBMB.

Users are responsible for complying with the licenses applicable to the model weights and other third-party components they use with VoxGen.

---

**VoxGen: native Rust TTS inference, Vulkan compute, AMD-first optimization.**
