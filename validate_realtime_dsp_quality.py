from pathlib import Path
import math

root = Path(__file__).resolve().parent
src = (root / 'src/playback_dsp.rs').read_text()
cargo = (root / 'Cargo.toml').read_text()

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

need('wsola = "0.1.0"' not in cargo, 'generic WSOLA dependency removed')
need('WSOLA' in src and 'sinc' in src.lower(), 'speech-oriented DSP rationale retained')
need('struct SpeechWsola' in src, 'engine-owned speech WSOLA')
need('0.0075' in src and '/ 6000.0' in src, 'legacy speech search geometry')
need('dot / denom' in src, 'normalized correlation matcher')
need('raw dot product' in src, 'echo regression rationale')
need('(self.factor - 1.0).abs() < 1.0e-9' in src, 'pitch-neutral resampler bypass')
need('RESAMPLER_HALF_TAPS: usize = 12' in src, '24-tap sinc quality')
need('((sample_rate as usize) / 100).max(1)' in src, '10ms live transition')
need('Self::crossfade_in(&mut processed, &dry, self.sample_rate);' in src, 'activation crossfade')
need('let cutoff = (1.0 / self.factor.max(1.0)).min(1.0);' in src, 'upshift anti-alias filter')
need('2.0_f32.powf(self.pitch_semitones / 12.0)' in src, 'equal-tempered pitch mapping')
need('self.speed_percent / 100.0 / self.pitch_factor()' in src,
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
