from pathlib import Path

root = Path(__file__).resolve().parent
prosody = (root / 'src/prosody_control.rs').read_text()
http = (root / 'src/http.rs').read_text()
main = (root / 'src/main.rs').read_text()
demo = (root / 'demo/src/main.rs').read_text()
cargo = (root / 'Cargo.toml').read_text()
demo_cargo = (root / 'demo/Cargo.toml').read_text()


def need(cond, msg):
    if not cond:
        raise SystemExit(f'FAIL: {msg}')

need('version = "0.7.60"' in cargo and 'version = "0.7.60"' in demo_cargo, 'v0.7.60 package versions')

# Pair compiler remains implementation-private: clients cannot name endpoints.
for token in [
    'pub(crate) enum MoodTransitionMode',
    'pub(crate) fn build_transition_control(',
    'pub(crate) fn managed_transition_cfg_delta(',
    'pub(crate) fn recommended_transition_reference_style()',
    'Do not crossfade, double consonants, change speaker identity',
    'believable intermediate state rather than a preset switch',
    'preserve one speaker identity',
]:
    need(token in prosody, f'private transition primitive: {token}')
for forbidden in ['pub enum MoodTransitionMode', 'pub fn build_transition_control(', 'pub fn managed_transition_cfg_delta(']:
    need(forbidden not in prosody, f'explicit transition primitive leaked publicly: {forbidden}')

# Destination-only strict HTTP request + automatic continuity state.
for token in [
    '#[serde(deny_unknown_fields)]',
    'style: Option<String>', 'intensity: Option<String>', 'pace_percent: Option<f32>',
    'continuity_id: Option<String>', 'boundary: Option<String>',
    'struct ContinuityState', 'struct ContinuityStore', 'struct ContinuityPlan',
    'fn continuity_plan(', 'fn commit_continuity(',
    'Self::HardCut => "hard_cut"',
    '"single_pass_synthesis": true', '"waveform_crossfade": false',
    '"request_model": "destination-only"', '"explicit_transition_api": false',
]:
    need(token in http, f'automatic continuity HTTP contract: {token}')
need('transition: Option<SpeechTransitionRequest>' not in http, 'legacy transition request object remains')
need('struct SpeechTransitionRequest' not in http, 'legacy transition request struct remains')

# CLI is destination-only too.
for token in ['style: Option<String>', 'intensity: String', 'pace_percent: f32', '--control/--style cannot be combined']:
    need(token in main, f'destination-only CLI: {token}')
for forbidden in ['transition_from:', 'transition_to:', 'transition_from_intensity:', 'transition_to_intensity:', 'transition_mode:']:
    need(forbidden not in main, f'legacy CLI transition endpoint remains: {forbidden}')

# Demo exposes destination state, not explicit transition endpoints.
for token in [
    'Managed pace %:', 'Continuity:', 'CONTINUITY_BOUNDARIES', 'DEMO_CONTINUITY_ID',
    'request["style"]', 'request["intensity"]', 'request["pace_percent"]',
    'request["continuity_id"]', 'request["boundary"]',
]:
    need(token in demo, f'destination-only demo: {token}')
for forbidden in ['TRANSITION_TARGETS', 'Transition to:', 'End intensity:', 'request["transition"]']:
    need(forbidden not in demo, f'legacy demo transition control remains: {forbidden}')

print('v0.7.60 automatic mood continuity validation passed')
