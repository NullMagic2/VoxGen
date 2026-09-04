from pathlib import Path

root = Path(__file__).resolve().parent
prosody = (root/'src/prosody_control.rs').read_text()
http = (root/'src/http.rs').read_text()
main = (root/'src/main.rs').read_text()
demo = (root/'demo/src/main.rs').read_text()
cargo = (root/'Cargo.toml').read_text()
demo_cargo = (root/'demo/Cargo.toml').read_text()


def need(cond, msg):
    if not cond:
        raise SystemExit(f'FAIL: {msg}')

need('version = "0.7.60"' in cargo and 'version = "0.7.60"' in demo_cargo, 'v0.7.60 package versions')

# Internal pair trajectory coverage is retained, but hidden from clients.
for pair in [
    '("neutral", "serious")', '("serious", "neutral")',
    '("cheerful", "excited")', '("excited", "cheerful")',
    '("concerned", "serious")', '("serious", "concerned")',
    '("warm", "gentle")', '("gentle", "warm")',
    '("warm", "sad")', '("sad", "warm")',
    '("sad", "serious")', '("serious", "sad")',
    '("concerned", "sad")', '("sad", "concerned")',
    '("angry", "neutral")', '("neutral", "whisper")', '("whisper", "neutral")',
]:
    need(pair in prosody, f'pair-specific trajectory {pair}')
for token in [
    'pub(crate) struct MoodSpeedTransition',
    'MIN_MEANINGFUL_MOOD_SPEED_DELTA_PERCENT: f32 = 5.0',
    'MAX_NATURAL_MOOD_SPEED_DELTA_PERCENT: f32 = 45.0',
    'pub(crate) fn build_transition_control_with_speed(',
    'speaking-rate targets, not as a pitch shift',
]:
    need(token in prosody, f'private pace compiler policy: {token}')

# HTTP owns managed pace state: <5 pp is held, >45 pp advances only 45 pp.
for token in [
    'magnitude < MIN_MEANINGFUL_MOOD_SPEED_DELTA_PERCENT',
    'delta.signum() * MAX_NATURAL_MOOD_SPEED_DELTA_PERCENT',
    'speed_percent must remain 100 while continuity_id is active; use pace_percent',
    '"suppress_delta_below_percent_points": MIN_MEANINGFUL_MOOD_SPEED_DELTA_PERCENT',
    '"max_advance_per_phrase_percent_points": MAX_NATURAL_MOOD_SPEED_DELTA_PERCENT',
    '"realization": "single-pass-prosody-conditioning-not-midstream-wsola"',
]:
    need(token in http, f'automatic pace continuity: {token}')
for forbidden in ['from_speed_percent: Option<f32>', 'to_speed_percent: Option<f32>', 'fn speed_transition(&self)']:
    need(forbidden not in http, f'legacy public pace endpoint remains: {forbidden}')

# CLI/demo expose only a destination pace.
need('pace_percent: f32' in main, 'CLI destination pace')
need('managed_pace_percent: u32' in demo and 'Managed pace %:' in demo, 'demo destination pace')
need('speed_percent = if managed_continuity { 100.0 }' in demo, 'demo prevents local WSOLA bypass')
for forbidden in ['transition_from_speed_percent', 'transition_to_speed_percent', 'Start pace % (0=off):', 'End pace % (0=off):']:
    need(forbidden not in demo and forbidden not in main, f'legacy pace endpoint remains: {forbidden}')

print('v0.7.60 automatic mood + pace continuity validation passed')
