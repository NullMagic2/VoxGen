## Requirements

VoxGen currently uses the **VoxCPM 2 model weights** for speech generation. You do **not** need to install VoxCPM 2, Python, PyTorch, or the original VoxCPM runtime. VoxGen only needs the converted GGUF model files.

### 1. Download the model files

Go to the **DennisHuang648/VoxCPM2-GGUF** repository on Hugging Face:

https://huggingface.co/DennisHuang648/VoxCPM2-GGUF

Download these two files:

```text
VoxCPM2-BaseLM-Q8_0.gguf
VoxCPM2-Acoustic-F16.gguf
```

`VoxCPM2-BaseLM-Q8_0.gguf` contains the quantized BaseLM and is the **recommended version for VoxGen**.

`VoxCPM2-Acoustic-F16.gguf` contains the remaining acoustic components used by VoxGen.

If you prefer the full-precision BaseLM, you can use:

```text
VoxCPM2-BaseLM-F16.gguf
```

instead of the Q8_0 file, although it uses considerably more memory.

### 2. Store the models

The files can be placed anywhere on your computer. For example:

```text
VoxGen/
└── models/
    ├── VoxCPM2-BaseLM-Q8_0.gguf
    └── VoxCPM2-Acoustic-F16.gguf
```

VoxGen does not require a particular model directory; you select the files from the demo.

### 3. Load them in VoxGen

Start the VoxGen demo and select:

```text
BaseLM:
VoxCPM2-BaseLM-Q8_0.gguf

Acoustic:
VoxCPM2-Acoustic-F16.gguf
```

Then select the desired engine mode:

```text
Normal
```

or, for an AMD Radeon RX 7900 XTX:

```text
XTX 7900
```

Click **Load VoxCPM2**. Once both components have loaded, VoxGen is ready to synthesize speech.

### 4. That's it

No Python environment or PyTorch installation is required to run the models. VoxGen loads the GGUF weights directly and performs inference through its native **Rust + Vulkan** runtime.

The GGUF files are conversions of the original OpenBMB VoxCPM 2 weights, so the original model's applicable license and terms still apply.
