from pathlib import Path
import re

root = Path(__file__).resolve().parent
main = (root / 'src/main.rs').read_text()
http = (root / 'src/http.rs').read_text()
demo = (root / 'demo/src/main.rs').read_text()
root_cargo = (root / 'Cargo.toml').read_text()
demo_cargo = (root / 'demo/Cargo.toml').read_text()


def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

need('version = "0.7.39"' in root_cargo, 'root version')
need('version = "0.7.39"' in demo_cargo, 'demo version')

# CLI stream mode: explicit on/off, default off, bare compatibility switch -> on.
for token in [
    'enum StreamArg',
    'On,',
    'Off,',
    'default_value = "off"',
    'default_missing_value = "on"',
    'num_args = 0..=1',
    'let stream_enabled = args.stream.enabled();',
]:
    need(token in main, f'missing CLI streaming contract: {token}')
need('stream: bool' not in main, 'legacy bool --stream field still present')
need(main.count('stream_enabled,') >= 2, 'stream mode not passed to both server startup paths')
need('--output-wav/--prompt-text/--control require --text' in main, 'text-only argument guard missing')
need('--stream on require --text' not in main, 'server streaming is still incorrectly tied to --text')

# HTTP server must advertise and enforce the setting.
for token in [
    'streaming_enabled: bool',
    '"streaming_enabled": state.streaming_enabled',
    'if !state.streaming_enabled',
    '"speech streaming is disabled"',
    '"restart VoxGen with --stream on"',
    '"409 Conflict"',
]:
    need(token in http, f'missing HTTP streaming contract: {token}')

# Portable settings live next to current_exe and cover all user-required values.
for token in [
    'struct DemoSettings',
    'base_model: Option<PathBuf>',
    'acoustic_model: Option<PathBuf>',
    'voice_sample: Option<PathBuf>',
    'word_spacing_ms: u32',
    'speed_percent: u32',
    'pitch_semitones: i32',
    'gain_percent: u32',
    'stream: bool',
    'fn demo_settings_path() -> PathBuf',
    'env::current_exe()',
    'dir.join("settings.cfg")',
    'DemoSettings::load(&settings_path)',
    'initial_settings.save(&settings_path)',
    'save_shared_settings(&settings)',
    'cfg.base_model = Some(path.clone())',
    'cfg.acoustic_model = Some(path.clone())',
    'cfg.voice_sample = Some(path.clone())',
    'cfg.word_spacing_ms =',
    'cfg.speed_percent = controls.speed_percent()',
    'cfg.pitch_semitones = controls.pitch_semitones()',
    'cfg.gain_percent =',
]:
    need(token in demo, f'missing portable settings contract: {token}')

# Demo defaults stream on, explicitly passes it to an owned server, and can fall back
# to complete-WAV playback when settings.cfg says off.
need('stream: true' in demo, 'demo stream default must be on')
need('.arg("--stream")' in demo, 'demo does not pass stream switch to owned engine')
need('.arg(if stream_enabled { "on" } else { "off" })' in demo, 'demo stream value not propagated')
need('if stream_enabled {' in demo[demo.index('fn synthesize'):], 'demo does not branch streaming/file playback')
need('speech_wav(&text, sample.as_deref(), gain_percent as f32 / 100.0, seed, &expressive, Some(request_id))' in demo, 'non-stream fallback missing')
need('server_streaming_enabled() == Some(false)' in demo, 'pre-existing server stream mismatch not detected')

# Saved paths must be installed into initial state and preferred before auto-discovery.
need('existing_file(initial_settings.base_model.clone())' in demo, 'saved BaseLM not restored')
need('existing_file(initial_settings.acoustic_model.clone())' in demo, 'saved Acoustic not restored')
idx_saved = demo.index('Prefer paths persisted beside the demo executable')
idx_discovery = demo.index('let root = project_root();', idx_saved)
need(idx_saved < idx_discovery, 'saved model paths are not preferred over discovery')

print('stream mode + portable settings validation OK')
