from pathlib import Path

root = Path(__file__).resolve().parent
demo = root / 'demo'
src = (demo / 'src/main.rs').read_text()
cargo = (demo / 'Cargo.toml').read_text()
engine_cargo = (root / 'Cargo.toml').read_text()
dsp = (root / 'src/playback_dsp.rs').read_text()
root_cargo = (root / 'Cargo.toml').read_text()

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

need('version = "0.7.55"' in root_cargo, 'root version')
need('version = "0.7.55"' in cargo, 'demo version')
need('rust-version = "1.87"' in cargo, 'WSOLA dependency MSRV')
need('wsola = "0.1.0"' not in engine_cargo, 'generic WSOLA dependency removed')
need('wsola = "0.1.0"' not in cargo, 'demo no longer owns WSOLA dependency')
need('pitch_shift' not in cargo and 'pitch_shift' not in engine_cargo, 'old phase-vocoder dependency removed')
for token in [
    'struct LivePlaybackControls',
    'AtomicU32', 'AtomicI32',
    'DEFAULT_SPEED_PERCENT: u32 = 100',
    'MIN_SPEED_PERCENT: u32 = 50',
    'MAX_SPEED_PERCENT: u32 = 200',
    'DEFAULT_PITCH_SEMITONES: i32 = 0',
    'MIN_PITCH_SEMITONES: i32 = -12',
    'MAX_PITCH_SEMITONES: i32 = 12',
    'struct RealtimeVoiceProcessor',
    'with_label("Speed (%):")',
    'with_label("Pitch (semitones):")',
    '"Live while speaking"',
    'speed_control.on_value_changed',
    'pitch_control.on_value_changed',
    'controls.set_speed_percent',
    'controls.set_pitch_semitones',
    'realtime.push(&paced, live_controls)',
    'realtime.finish(live_controls)',
    'adjustable while streaming',
]:
    need(token in src, f'missing demo control token {token}')

for token in ['struct SpeechWsola', 'struct StreamingSincResampler', 'RESAMPLER_HALF_TAPS: usize = 12', 'pub struct StreamingPlaybackDsp']:
    need(token in dsp, f'missing shared DSP token {token}')
need('playback_dsp::{OutputPeakGuard, PlaybackControls as NativePlaybackControls, StreamingPlaybackDsp}' in src, 'demo imports shared native DSP')
need('struct StreamingSincResampler' not in src, 'demo duplicate sinc DSP removed')

# Neutral remains a literal dry bypass; the new processor is only heard when active.
need('if self.effect_was_active && !processed.is_empty()' in dsp and 'clean' in dsp,
     'neutral dry bypass / transition missing')
need('speed_control.enable(false)' not in src, 'speed control is not live')
need('pitch_control.enable(false)' not in src, 'pitch control is not live')
need('word_spacing_control.enable(false)' in src, 'word spacing freeze unexpectedly removed')

# Independent speed/pitch math: resampling by p followed by WSOLA tempo s/p
# yields final duration 1/s, regardless of p.
for speed_pct in (50, 75, 100, 105, 125, 150, 200):
    speed = speed_pct / 100.0
    for semitones in (-12, -3, -1, 0, 1, 3, 12):
        p = 2.0 ** (semitones / 12.0)
        resampled_duration = 1.0 / p
        wsola_tempo = speed / p
        final_duration = resampled_duration / wsola_tempo
        need(abs(final_duration - 1.0 / speed) < 1e-12,
             f'independent duration math failed {speed_pct}% {semitones:+d} st')
        need(0.25 <= wsola_tempo <= 4.0,
             f'WSOLA tempo outside supported range {speed_pct}% {semitones:+d} st')

idx_spacing = src.index('let paced = spacing.push(&floats);', src.index('fn speech_stream_windows'))
idx_dsp = src.index('realtime.push(&paced, live_controls)', idx_spacing)
idx_queue = src.index('player.queue_f32(&rendered)?', idx_dsp)
need(idx_spacing < idx_dsp < idx_queue, 'streaming DSP order')

print('real-time WSOLA speed/pitch playback validation OK')
