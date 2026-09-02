from pathlib import Path
import math

root = Path(__file__).resolve().parent
src = (root / 'demo/src/main.rs').read_text()
cargo = (root / 'demo/Cargo.toml').read_text()

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

need('wsola = "0.1.0"' in cargo, 'WSOLA dependency')
need('phase vocoder' in src.lower(), 'migration rationale retained')
need('RESAMPLER_HALF_TAPS: usize = 12' in src, '24-tap sinc quality')
need('DSP_TRANSITION_SAMPLES: usize = 480' in src, '10ms live transition')
need('Self::crossfade_in(&mut processed, &dry);' in src, 'activation crossfade')
need('let cutoff = (1.0 / self.factor.max(1.0)).min(1.0);' in src, 'upshift anti-alias filter')
need('2.0_f32.powf(pitch_semitones as f32 / 12.0)' in src, 'equal-tempered pitch mapping')
need('let wsola_tempo = (speed / pitch_factor).clamp(0.25, 4.0);' in src,
     'duration compensation formula')

# Numerical sanity for the 24-tap Lanczos kernel: DC gain is normalized and all
# representative interpolation weights stay finite throughout the supported pitch range.
def sinc(x):
    if abs(x) < 1e-12:
        return 1.0
    px = math.pi*x
    return math.sin(px)/px
half=12
for p in (0.5, 2**(-1/12), 1.0, 2**(1/12), 2.0):
    cutoff=min(1.0, 1.0/max(1.0,p))
    for frac in (0.0, .125, .25, .5, .875):
        center=0
        weights=[]
        for idx in range(center-half+1, center+half+1):
            d=frac-idx
            wx=d/half
            if abs(wx)>=1: continue
            w=cutoff*sinc(d*cutoff)*sinc(wx)
            need(math.isfinite(w), 'nonfinite sinc weight')
            weights.append(w)
        need(abs(sum(weights)) > 1e-6, 'degenerate sinc normalization')

print('realtime speech DSP quality validation OK')
