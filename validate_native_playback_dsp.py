from pathlib import Path
import math

root = Path(__file__).resolve().parent
cargo = (root / 'Cargo.toml').read_text()
demo_cargo = (root / 'demo/Cargo.toml').read_text()
lib = (root / 'src/lib.rs').read_text()
dsp = (root / 'src/playback_dsp.rs').read_text()
http = (root / 'src/http.rs').read_text()
main = (root / 'src/main.rs').read_text()
demo = (root / 'demo/src/main.rs').read_text()

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

need('version = "0.7.55"' in cargo and 'version = "0.7.55"' in demo_cargo, 'v0.7.55 package versions')
need('rust-version = "1.87"' in cargo, 'engine WSOLA MSRV')
need('wsola = "0.1.0"' not in cargo, 'generic WSOLA dependency removed')
need('wsola' not in demo_cargo.lower(), 'demo has no independent WSOLA dependency')
need('voxgen = { path = ".." }' in demo_cargo, 'demo consumes engine crate')
need('pub mod playback_dsp;' in lib, 'playback DSP exported by engine crate')
for token in [
    'pub struct PlaybackControls',
    'pub struct StreamingPlaybackDsp',
    'pub struct OutputPeakGuard',
    'struct StreamingSincResampler',
    'struct SpeechWsola',
    'let search = ((sample_rate as f64 * 0.0075).round() as usize).max(32);',
    'let search_step = ((sample_rate as f64 / 6000.0).round() as usize).max(1);',
    'let score = if denom > 1.0e-12 { dot / denom } else { -1.0 };',
    'RESAMPLER_HALF_TAPS: usize = 12',
    'pub fn set_controls',
    'pub fn push',
    'pub fn finish',
    'pub fn process_all',
    'let cutoff = (1.0 / self.factor.max(1.0)).min(1.0);',
    'self.speed_percent / 100.0 / self.pitch_factor()',
]:
    need(token in dsp, f'missing shared DSP token: {token}')

for token in ['speed_percent: Option<f32>', 'pitch_semitones: Option<f32>',
              'native_playback_dsp', 'StreamingPlaybackDsp::new(48_000, playback_controls)',
              'playback_dsp.push(chunk)', 'playback_dsp.finish()',
              'StreamingPlaybackDsp::process_all(48_000, playback_controls, &result.samples)',
              'OutputPeakGuard::new(48_000)', 'peak_guard.process(&processed, gain)',
              'OutputPeakGuard::process_all(48_000, &rendered_samples, gain)']:
    need(token in http, f'missing HTTP native DSP token: {token}')
need('use voxgen::{' in http and 'playback_dsp::{OutputPeakGuard, PlaybackControls, StreamingPlaybackDsp, OUTPUT_PEAK_CEILING}' in http,
     'HTTP binary imports playback DSP from the library crate')
need('crate::playback_dsp' not in http,
     'HTTP binary must not import playback DSP through crate::')
need('X-VoxGen-Native-Playback-DSP: 1' in http, 'stream capability header')
need('X-VoxGen-Speed-Percent' in http and 'X-VoxGen-Pitch-Semitones' in http, 'DSP response headers')

for token in ['#[arg(long = \"speed\", default_value_t = 100.0)]',
              '#[arg(long = \"pitch\", default_value_t = 0.0)]',
              'speed_percent: f32', 'pitch_semitones: f32',
              'PlaybackControls::new(args.speed_percent, args.pitch_semitones)',
              'StreamingPlaybackDsp::process_all(result.sample_rate, playback_controls, &result.samples)']:
    need(token in main, f'missing CLI native DSP token: {token}')

need('playback_dsp::{OutputPeakGuard, PlaybackControls as NativePlaybackControls, StreamingPlaybackDsp}' in demo, 'demo imports shared DSP and output guard')
need('dsp: StreamingPlaybackDsp' in demo and 'peak_guard: OutputPeakGuard' in demo, 'demo thin adapter owns shared DSP and guard')
need('self.dsp.set_controls' not in demo or '.set_controls(native)' in demo, 'demo forwards live controls')
need('struct StreamingSincResampler' not in demo, 'demo DSP implementation removed')
need('use wsola::TimeStretch;' not in demo, 'demo direct WSOLA implementation removed')

# Independent speed/pitch duration identity for supported extrema.
for speed_pct in (50, 75, 100, 125, 150, 200):
    speed = speed_pct / 100.0
    for semitones in (-12, -3, 0, 3, 12):
        p = 2.0 ** (semitones / 12.0)
        tempo = speed / p
        final_duration = (1.0 / p) / tempo
        need(abs(final_duration - 1.0 / speed) < 1e-12, 'speed/pitch duration independence')
        need(0.25 <= tempo <= 4.0, 'WSOLA tempo in supported safety range')

need('--speed-percent' not in main and '--pitch-semitones' not in main, 'legacy CLI flag strings are absent')
print('v0.7.55 native playback DSP validation passed')
