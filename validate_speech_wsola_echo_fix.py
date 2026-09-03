from pathlib import Path
import math

root = Path(__file__).resolve().parent
src = (root / 'src/playback_dsp.rs').read_text()
cargo = (root / 'Cargo.toml').read_text()
http = (root / 'src/http.rs').read_text()

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

need('version = "0.7.55"' in cargo, 'v0.7.55 root version')
need('wsola = ' not in cargo, 'external generic WSOLA dependency removed')
need('struct SpeechWsola' in src, 'engine-owned SpeechWsola')
need('sample_rate as f64 * 0.030' in src, '30 ms speech window')
need('sample_rate as f64 * 0.0075' in src, '7.5 ms search half-range')
need('sample_rate as f64 / 6000.0' in src, 'speech candidate stride')
need('dot / denom' in src, 'normalized correlation')
need('ref_norm * (probe_energy.sqrt() + 1.0e-9)' in src, 'correlation normalization denominator')
need('cand.abs_diff(predicted_rounded)' in src, 'nearest predicted-position tie break')
need('(self.factor - 1.0).abs() < 1.0e-9' in src, 'pitch-neutral resampler dry bypass')
need('"algorithm": "sinc+speech-wsola-ncc-confidence+peak-guard"' in http, 'health reports corrected native DSP')

# Why normalized correlation matters: a louder but phase-worse candidate can beat
# a quieter perfect match under raw dot product. NCC must rank the perfect match.
n = 720
reference = [math.sin(2.0 * math.pi * i / 80.0) + 0.2 * math.sin(2.0 * math.pi * i / 29.0) for i in range(n)]
good = [0.45 * x for x in reference]
shift = 5
bad_base = reference[shift:] + reference[:shift]
bad = [2.5 * x for x in bad_base]

def dot(a,b): return sum(x*y for x,y in zip(a,b))
def norm(a): return math.sqrt(sum(x*x for x in a)) + 1e-9
raw_good, raw_bad = dot(reference, good), dot(reference, bad)
ncc_good = raw_good/(norm(reference)*norm(good))
ncc_bad = raw_bad/(norm(reference)*norm(bad))
need(raw_bad > raw_good, 'fixture must expose raw-energy bias')
need(ncc_good > ncc_bad and ncc_good > 0.999, 'NCC must prefer waveform match over loudness')

print('v0.7.55 speech WSOLA echo regression validation passed')
