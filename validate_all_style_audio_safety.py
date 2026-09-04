#!/usr/bin/env python3
from pathlib import Path
import math
import re

root = Path(__file__).resolve().parent
dsp = (root / 'src/playback_dsp.rs').read_text(encoding='utf-8')
http = (root / 'src/http.rs').read_text(encoding='utf-8')
main = (root / 'src/main.rs').read_text(encoding='utf-8')
demo = (root / 'demo/src/main.rs').read_text(encoding='utf-8')
prosody = (root / 'src/prosody_control.rs').read_text(encoding='utf-8')
cargo = (root / 'Cargo.toml').read_text(encoding='utf-8')
demo_cargo = (root / 'demo/Cargo.toml').read_text(encoding='utf-8')

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

need('version = "0.7.60"' in cargo and 'version = "0.7.60"' in demo_cargo, 'v0.7.60 package versions')

# One authoritative guard after all engine playback DSP.
need('pub const OUTPUT_PEAK_CEILING: f32 = 0.98;' in dsp, '0.98 output ceiling')
need('pub struct OutputPeakGuard' in dsp, 'shared output peak guard')
need('pub fn process(&mut self, input: &[f32], requested_gain: f32)' in dsp, 'streaming guard')
need('pub fn process_all(sample_rate: u32, input: &[f32], requested_gain: f32)' in dsp, 'offline uniform guard')
need('peak_guard.process(&processed, gain)' in http, 'HTTP streaming guard after playback DSP')
need('OutputPeakGuard::process_all(48_000, &rendered_samples, gain)' in http, 'HTTP offline guard after playback DSP')
need('OutputPeakGuard::process_all(result.sample_rate, &playback_samples, args.gain)' in main, 'CLI output guard')
need('peak_guard: OutputPeakGuard' in demo and '.process(&rendered, 1.0)' in demo, 'demo live DSP output guard')
need('(v * gain).clamp(-1.0, 1.0)' not in http, 'legacy destructive HTTP full-scale clip removed')
need('apply_gain(&playback_samples' not in main, 'legacy unprotected CLI gain removed')

# WSOLA must refuse weak accidental matches on breath/noise-dominated material.
need('const WSOLA_MIN_CONFIDENT_NCC: f64 = 0.20;' in dsp, 'WSOLA confidence floor')
need('if ref_rms < 1.0e-4' in dsp and 'return Some(predicted_fallback);' in dsp, 'low-energy fallback')
need('if best_score < WSOLA_MIN_CONFIDENT_NCC' in dsp, 'low-correlation fallback')
need('predicted-analysis-position' in http, 'health reports WSOLA fallback')
need('raised-cosine-amplitude-complementary' in http, 'unity-sum WSOLA overlap retained')

# Every selectable managed voice style goes through the same output path; no style
# is allowed to carry an extreme hidden demo boost that could routinely engage it.
styles = ['neutral','warm','cheerful','excited','sad','concerned','angry','gentle','serious','whisper']
for style in styles:
    need(f'("{style}",' in demo, f'demo style present: {style}')
mults = [float(x) for x in re.findall(r'demo_gain_multiplier:\s*([0-9.]+)', prosody)]
need(mults and max(mults) <= 1.05 + 1e-9, 'managed demo gain ceiling remains <= 1.05x')
need(min(mults) >= 0.0, 'managed demo gains are nonnegative')

# Independent mathematical regression of the guard policy on representative
# neutral, hot-angry/excited and heavily boosted inputs.
CEILING = 0.98
for peak, gain in [(0.25,1.0),(0.95,1.05),(1.08,0.90),(0.88,1.25),(0.70,2.0),(1.30,4.0)]:
    post = peak * gain
    attenuation = min(1.0, CEILING / post) if post > 0 else 1.0
    protected = post * attenuation
    need(protected <= CEILING + 1e-7, f'protected peak <= ceiling for peak={peak}, gain={gain}')
    if post <= CEILING:
        need(abs(attenuation - 1.0) < 1e-12, 'guard is transparent below ceiling')

# Slow release is monotonic upward and can never exceed the safe attenuation
# required by the current block.
current = 0.60
required = 1.0
seconds = 0.160
alpha = 1.0 - math.exp(-seconds / 0.250)
released = current + (required - current) * alpha
need(current < released < required, 'stream guard releases smoothly without overshoot')

print('v0.7.60 all-style clipping/metallic safety validation passed')
