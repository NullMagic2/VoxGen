from pathlib import Path
import re

root = Path(__file__).resolve().parent
http = (root/'src/http.rs').read_text(encoding='utf-8')
main = (root/'src/main.rs').read_text(encoding='utf-8')
demo = (root/'demo/src/main.rs').read_text(encoding='utf-8')
prosody = (root/'src/prosody_control.rs').read_text(encoding='utf-8')

def need(c, m):
    if not c:
        raise SystemExit('FAIL: ' + m)

need('#[serde(deny_unknown_fields)]\nstruct SpeechRequest {' in http, 'strict SpeechRequest unknown-field checking')
request_block = http[http.index('struct SpeechRequest {'):http.index('struct SpeechCancelRequest')]
for field in ['style:', 'intensity:', 'pace_percent:', 'continuity_id:', 'boundary:']:
    need(field in request_block, 'destination request field ' + field)
for field in ['transition:', 'from_style:', 'to_style:', 'from_intensity:', 'to_intensity:', 'from_speed_percent:', 'to_speed_percent:']:
    need(field not in request_block, 'legacy request field remains: ' + field)

for flag in ['--transition-from', '--transition-to', '--transition-from-speed-percent', '--transition-to-speed-percent']:
    need(flag not in main and flag not in demo, 'legacy client flag remains: ' + flag)
need('request["transition"]' not in demo, 'legacy demo transition payload remains')

# Internal compiler may still use pair endpoints, but it must not be public Rust API.
for symbol in ['MoodSpeedTransition', 'MoodTransitionMode']:
    need(f'pub(crate) struct {symbol}' in prosody or f'pub(crate) enum {symbol}' in prosody, f'{symbol} is not crate-private')
for fn in ['build_transition_control', 'build_transition_control_with_speed', 'managed_transition_cfg_delta', 'recommended_transition_reference_style']:
    need(re.search(rf'pub\(crate\) fn {fn}\b', prosody) is not None, f'{fn} is not crate-private')

print('v0.7.60 breaking transition API validation passed')
