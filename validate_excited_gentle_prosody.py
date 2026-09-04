from pathlib import Path
root=Path(__file__).resolve().parent
pros=(root/'src/prosody_control.rs').read_text()
http=(root/'src/http.rs').read_text()
cargo=(root/'Cargo.toml').read_text()
demo=(root/'demo/Cargo.toml').read_text()

def need(cond,msg):
    if not cond: raise SystemExit('FAIL: '+msg)

need('version = "0.7.60"' in cargo and 'version = "0.7.60"' in demo, 'v0.7.60 package versions')
need('mildly excited and interested' in pros, 'subtle Excited no longer surprise-seeded')
need('positive anticipation rather than surprise' in pros, 'Excited semantic separation from surprise')
need('more variable pitch movement' in pros or 'pitch movement perceptibly more variable' in pros, 'Excited pitch variability cue')
need('brief local peaks' in pros or 'local pitch-and-energy peaks' in pros, 'Excited local peak/release cue')
need('fn gentle_profile(' in pros, 'Gentle managed compiler')
need('low-vocal-effort speaking style' in pros, 'Gentle low-effort semantics')
need('modestly reduced projection and loudness' in pros, 'Gentle reduced projection/SPL cue')
need('Preserve whatever emotional valence' in pros, 'Gentle preserves text emotion')
need('do not automatically sound warm' in pros, 'Gentle separated from Warm')
need('demo_gain_multiplier: 0.95' in pros and 'demo_gain_multiplier: 0.92' in pros, 'Gentle visible demo level scaling')
need('"version": 10' in http, 'managed prosody v10 health metadata')
need('low-vocal-effort-low-projection-emotion-preserving' in http, 'Gentle health semantics')
print('v0.7.60 Excited/Gentle prosody validation passed')
