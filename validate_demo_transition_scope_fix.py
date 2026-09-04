from pathlib import Path

root = Path(__file__).resolve().parent
src = (root / 'demo' / 'src' / 'main.rs').read_text(encoding='utf-8')
root_cargo = (root / 'Cargo.toml').read_text(encoding='utf-8')
demo_cargo = (root / 'demo' / 'Cargo.toml').read_text(encoding='utf-8')

def need(cond, msg):
    if not cond:
        raise SystemExit(f'FAIL: {msg}')

need('version = "0.7.60"' in root_cargo and 'version = "0.7.60"' in demo_cargo, 'root/demo package versions')
need('const DEMO_CONTINUITY_ID: &str = "voxgen-demo";' in src, 'stable demo continuity id')
for token in [
    'Managed pace %:', 'Continuity:', 'managed_pace_percent', 'continuity_boundary',
    'request["style"] = json!(style);', 'request["intensity"] = json!(expressive.intensity.as_str());',
    'request["pace_percent"] = json!(expressive.pace_percent);',
    'request["continuity_id"] = json!(expressive.continuity_id.as_deref().unwrap_or(DEMO_CONTINUITY_ID));', 'request["boundary"] = json!(expressive.boundary.as_str());',
]:
    need(token in src, f'demo automatic continuity token: {token}')
for forbidden in ['TRANSITION_TARGETS', 'TRANSITION_MODES', 'Transition to:', 'End intensity:', 'request["transition"]']:
    need(forbidden not in src, f'legacy explicit transition demo surface remains: {forbidden}')
# Preserve the prior dead-assignment regression fix.
tail = src[src.find('let pacing_tail = spacing.finish();'):src.find('let audio_seconds = generated_patches as f64 * 0.160;')]
need('playback_started = true;' not in tail, 'no dead final playback_started assignment')
print('v0.7.60 demo automatic continuity scope regression validation passed')
