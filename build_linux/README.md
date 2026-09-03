# VoxGen Linux scripts

All Linux build/smoke launchers live here. Run the root `./build_voxgen.sh` for the normal release build.

Requirements:
- Rust/Cargo 1.87+
- Vulkan loader/driver for your GPU
- `glslc` from the Vulkan SDK (or set `VOXGEN_GLSLC=/path/to/glslc`)
- Python 3 for deterministic smoke-vector regeneration

Model defaults are read from `models/` in the VoxGen project root. Override them with:

```bash
export VOXGEN_MODEL_DIR=/path/to/models
# or independently:
export VOXGEN_BASE_Q8=/path/to/VoxCPM2-BaseLM-Q8_0.gguf
export VOXGEN_BASE_F16=/path/to/VoxCPM2-BaseLM-F16.gguf
export VOXGEN_ACOUSTIC=/path/to/VoxCPM2-Acoustic-F16.gguf
```

Build modes:

```bash
./build_voxgen.sh                 # release + Vulkan device probe
./build_voxgen.sh debug
./build_voxgen.sh check
./build_voxgen.sh clean
./build_voxgen.sh release --no-probe
```

Smoke scripts are run from this directory or by full path, e.g. `./build_linux/smoke_tts.sh`.
