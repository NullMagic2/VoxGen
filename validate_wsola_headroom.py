#!/usr/bin/env python3
from pathlib import Path
import math

root = Path(__file__).resolve().parent
src = (root / 'src' / 'playback_dsp.rs').read_text(encoding='utf-8')
main = (root / 'src' / 'main.rs').read_text(encoding='utf-8')
http = (root / 'src' / 'http.rs').read_text(encoding='utf-8')
cargo = (root / 'Cargo.toml').read_text(encoding='utf-8')
demo_cargo = (root / 'demo' / 'Cargo.toml').read_text(encoding='utf-8')

def need(cond, msg):
    if not cond:
        raise SystemExit('FAIL: ' + msg)

need('version = "0.7.55"' in cargo and 'version = "0.7.55"' in demo_cargo,
     'v0.7.55 package versions')
need('0.5 - 0.5 * (std::f32::consts::PI * t).cos()' in src,
     'raised-cosine WSOLA fade-in')
need('fade_out.push(1.0 - fade_in_value)' in src,
     'amplitude-complementary WSOLA fade-out')
need('angle.cos()' not in src and 'angle.sin()' not in src,
     'legacy equal-power WSOLA crossfade removed')
need('sample.clamp(-1.0, 1.0)' not in src,
     'no destructive clamp inside playback DSP')
need('(sample * gain).clamp(-1.0, 1.0)' not in main,
     'CLI gain does not clamp before WAV serialization')
need('(v * gain).clamp(-1.0, 1.0)' not in http,
     'HTTP no longer hard-clips gain at full scale')
need('OutputPeakGuard::process_all(48_000, &rendered_samples, gain)' in http,
     'HTTP offline output uses peak guard')
need('peak_guard.process(&processed, gain)' in http,
     'HTTP streaming output uses peak guard')
need('pub const OUTPUT_PEAK_CEILING: f32 = 0.98;' in src,
     'protected output ceiling leaves headroom')
need('\"internal_clipping\": false' in http,
     'health advertises no internal DSP clipping')

# Mathematical regression: complementary raised-cosine fades keep an identical
# (perfectly correlated) waveform at unity amplitude throughout the overlap.
peak = 0.0
sum_error = 0.0
for i in range(1024):
    t = i / 1023
    fi = 0.5 - 0.5 * math.cos(math.pi * t)
    fo = 1.0 - fi
    sum_error = max(sum_error, abs((fi + fo) - 1.0))
    # identical +0.95 samples on both sides of the overlap
    peak = max(peak, abs(0.95 * fo + 0.95 * fi))
need(sum_error < 1e-12, 'crossfade gains sum to unity')
need(peak <= 0.9500000001, 'correlated overlap does not amplify')

# Demonstrate the exact regression we are preventing.
equal_power_mid = math.cos(math.pi / 4) + math.sin(math.pi / 4)
need(equal_power_mid > 1.4141, 'old equal-power midpoint would amplify correlated speech')
print('v0.7.55 WSOLA headroom regression validation passed')
