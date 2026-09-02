# VoxGen

**VoxGen is a lightweight, Python-free Rust inference engine for VoxCPM2, built with a strong focus on AMD GPUs, Vulkan, and RX 7900 XTX optimization.**

VoxGen reimplements the VoxCPM2 inference path as a native Rust/Vulkan runtime instead of relying on the standard Python/PyTorch stack. The goal is simple: make high-quality local speech synthesis faster, leaner, easier to deploy, and especially well optimized for AMD hardware.

## Why VoxGen?

Most modern TTS runtimes are built around Python and CUDA-first machine-learning frameworks. 

VoxGen takes a different approach: it is a native Rust inference engine designed to give direct control over how the model runs on the GPU.

By using Vulkan, VoxGen can manage GPU memory, synchronization, compute shaders, streaming, and hardware-specific optimizations directly. It also gives us precise control over practical features such as low-latency audio delivery, reference caching, voice consistency, and mid-generation cancellation.

This makes it possible to optimize VoxGen around the actual capabilities of the target hardware—especially AMD GPUs—rather than relying entirely on the behavior and overhead of a general-purpose machine-learning framework.

## Why Rust + Vulkan?

Rust gives VoxGen native performance, predictable memory usage, strong safety, and simple deployment without requiring Python or a large runtime environment. It also makes it easier to build a responsive streaming engine with precise control over threads, buffers, and model state.

Vulkan gives VoxGen direct access to the GPU without depending on CUDA. It lets us manage memory and synchronization explicitly, write custom compute shaders, and optimize important parts of the inference pipeline for AMD hardware, especially the RX 7900 XTX.

Together, Rust and Vulkan give VoxGen much more control over how the model actually runs, allowing us to focus on low latency, low overhead, and efficient GPU usage.

## Installation

See the [installation guide](Install.md) for model download and setup instructions.



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
- Subgroup reductions
- Fused Q/K/V projections
- Fused SwiGLU paths
- Fused residual + RMSNorm operations
- Targeted Vulkan synchronization
- Optimized prefill batching
- Reduced intermediate-buffer traffic
- Persistent conditioning buffers
- XTX-specific shader paths

The XTX mode is intentionally separate from the portable Normal mode, allowing aggressive RDNA3 tuning without sacrificing compatibility with other Vulkan-capable GPUs.




## Memory-conscious design

VoxGen avoids much of the runtime overhead associated with a general-purpose Python ML framework.

The current configuration supports a **Q8_0 BaseLM** together with the **F16 acoustic model**, significantly reducing model memory requirements compared with a full BF16 runtime.

The native engine also keeps direct control over:

- KV-cache allocation
- Scratch buffers
- Staging buffers
- Model mappings
- Temporary tensors
- Shader workspaces

This makes VoxGen particularly attractive for systems where VRAM efficiency matters.

## Portable Normal mode + optimized XTX mode

VoxGen currently provides two main execution profiles:

### Normal

A portable Vulkan path intended to work across a wider range of compatible GPUs.

### XTX 7900

A specialized path for the **Radeon RX 7900 XTX**, tuned around RDNA3 characteristics and the exact shapes used by VoxCPM2 inference.

This lets VoxGen remain portable while still taking advantage of device-specific optimizations where they matter.

## Credits and model compatibility

VoxGen is an independent inference-engine project designed to run VoxCPM2-compatible model components.

VoxCPM2 is developed by **OpenBMB**. VoxGen is not an official OpenBMB project and is not affiliated with or endorsed by OpenBMB.

Users are responsible for complying with the licenses applicable to the model weights and other third-party components they use with VoxGen.

---

**VoxGen: native Rust TTS inference, Vulkan compute, AMD-first optimization.**
