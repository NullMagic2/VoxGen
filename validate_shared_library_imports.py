from pathlib import Path

root = Path(__file__).resolve().parent
main = (root / "src/main.rs").read_text()
http = (root / "src/http.rs").read_text()
runtime = (root / "src/runtime.rs").read_text()
lib = (root / "src/lib.rs").read_text()
dsp = (root / "src/playback_dsp.rs").read_text()
cargo = (root / "Cargo.toml").read_text()
demo_cargo = (root / "demo/Cargo.toml").read_text()

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

need('version = "0.7.55"' in cargo and 'version = "0.7.55"' in demo_cargo, 'v0.7.55 package versions')
need('pub mod playback_dsp;' in lib and 'pub mod prosody_control;' in lib, 'shared library modules exported')
need('use voxgen::playback_dsp::{OutputPeakGuard, PlaybackControls, StreamingPlaybackDsp};' in main, 'CLI imports playback DSP from library crate')
need('use voxgen::{' in http and 'playback_dsp::{OutputPeakGuard, PlaybackControls, StreamingPlaybackDsp, OUTPUT_PEAK_CEILING}' in http, 'HTTP imports playback DSP from library crate')
need('use voxgen::prosody_control::refine_control_instruction;' in runtime, 'runtime imports prosody compiler from library crate')
need('use crate::prosody_control::refine_control_instruction;' not in runtime, 'runtime must not resolve library-only prosody module through binary crate')
need('crate::playback_dsp' not in main and 'crate::playback_dsp' not in http, 'binary must not resolve shared DSP through crate::')
need('DSP_PULL_CHUNK' not in dsp, 'unused DSP_PULL_CHUNK removed')
print('v0.7.55 shared library import validation passed')
