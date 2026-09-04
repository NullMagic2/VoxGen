from pathlib import Path
import math

root = Path(__file__).resolve().parent
runtime = (root / "src/runtime.rs").read_text()
http = (root / "src/http.rs").read_text()
demo = (root / "demo/src/main.rs").read_text()
prosody = (root / "src/prosody_control.rs").read_text()
root_cargo = (root / "Cargo.toml").read_text()
demo_cargo = (root / "demo/Cargo.toml").read_text()

def need(cond, msg):
    if not cond:
        raise SystemExit("FAIL: " + msg)

need('version = "0.7.60"' in root_cargo, "root version")
need('version = "0.7.60"' in demo_cargo, "demo version")
need('MIN_STREAMING_DECODE_CONTEXT_PATCHES: usize = 6' in runtime, "six-patch decoder floor")
need('options.streaming_prefix_len.max(MIN_STREAMING_DECODE_CONTEXT_PATCHES)' in runtime, "caller prefix is clamped to decoder-safe floor")
need('context.len()>decode_context_patches' in runtime, "rolling decode uses safe context")
need('streaming_prefix_len: r.streaming_prefix_len.unwrap_or(6)' in http, "HTTP safe default")
need('request["streaming_prefix_len"] = json!(6)' in demo, "demo safe streaming request")
need('const DEFAULT_GAIN_PERCENT: u32 = 100' in demo, "fresh demo no longer amplifies every sample by 40%")
need('moderate loudness rather than a constant shouted delivery' in prosody, "anger recipe separates anger from loudness")

# Recompute the current decoder's causal receptive field for the newest 160 ms patch.
# Forward topology: stem causal k7; each stage = ConvTranspose(k=2s,stride=s)
# followed by causal residual k7 at dilations 1,3,9; final causal k7.
rates = [8, 6, 5, 2, 2, 2]
latent_frames = 100
hop = math.prod(rates)  # 1920 output samples per latent frame
patch_samples = 7680   # 4 latent frames / 160 ms at 48 kHz
lo, hi = latent_frames * hop - patch_samples, latent_frames * hop - 1

# Back through final causal k7.
lo -= 6
# Reverse each decoder stage: residual dilations 9,3,1, then ConvTranspose.
for stride in reversed(rates):
    lo -= 6 * (9 + 3 + 1)
    lo = lo // stride - 1
    hi = hi // stride
# Back through stem causal k7; the 1x1 projection adds no context.
lo -= 6
frames_needed = hi - lo + 1
patches_needed = math.ceil(frames_needed / 4)
need(frames_needed == 24, f"expected 24 latent frames of causal history, got {frames_needed}")
need(patches_needed == 6, f"expected 6 latent patches, got {patches_needed}")

print("V0.7.38 streaming-fidelity validation passed: 24 latent frames = 6 patches.")
