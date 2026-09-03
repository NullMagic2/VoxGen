from pathlib import Path

root = Path(__file__).resolve().parent
src = (root / 'src/playback_dsp.rs').read_text()
demo = (root / 'demo/src/main.rs').read_text()
cargo = (root / 'Cargo.toml').read_text()
root_cargo = (root / 'Cargo.toml').read_text()

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

need('version = "0.7.55"' in root_cargo, 'root version')
need('version = "0.7.55"' in cargo, 'demo version')
need('struct StreamingSincResampler' in src, 'streaming sinc resampler')
need('if sample.is_finite() { sample } else { 0.0 }' in src, 'DSP finite sanitizer')
need('let cutoff = (1.0 / self.factor.max(1.0)).min(1.0);' in src, 'anti-alias cutoff')
need('Self::sinc(distance * cutoff)' in src and 'Self::sinc(window_x)' in src,
     'Lanczos-windowed sinc interpolation')
need('if sample.is_finite() { sample } else { 0.0 }' in src[src.index('struct SpeechWsola'):],
     'WSOLA output finite guard without destructive clamp')
need('sample.clamp(-1.0, 1.0)' not in src[src.index('struct SpeechWsola'):],
     'WSOLA preserves floating-point headroom')
need('return Err("VoxGen returned a non-finite PCM sample".to_string());' in demo,
     'final demo PCM invariant remains')
need('pitch_shift' not in cargo, 'old phase-vocoder dependency removed')
need('DSP_WORKING_SCALE' not in src, 'obsolete i16-scale workaround removed')
print('realtime DSP finite-sample validation OK')
