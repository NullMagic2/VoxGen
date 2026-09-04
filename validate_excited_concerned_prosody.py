from pathlib import Path
root=Path(__file__).resolve().parent
pros=(root/'src/prosody_control.rs').read_text()
http=(root/'src/http.rs').read_text()
cargo=(root/'Cargo.toml').read_text()
demo_cargo=(root/'demo/Cargo.toml').read_text()

def need(cond,msg):
    if not cond: raise SystemExit('FAIL: '+msg)

need('version = "0.7.60"' in cargo and 'version = "0.7.60"' in demo_cargo, 'v0.7.60 package versions')
need('fn excited_profile(' in pros, 'Excited managed compiler')
need('moderately raised pitch centre' in pros, 'Excited raised pitch cue')
need('positive anticipation rather than surprise' in pros, 'Excited is not conflated with surprise')
need('clearly wider and more dynamic pitch movement' in pros, 'Excited pitch-range cue')
need('brief phrase-level peaks' in pros and 'not continuously loud' in pros, 'Excited local-energy/release cue')
need('fn concerned_profile(' in pros, 'Concerned managed compiler')
need('Begin concern-bearing phrases with mild vocal tension' in pros, 'Concerned alert phase')
need('As reassurance becomes appropriate, audibly relax' in pros, 'Concerned release phase')
need('lower the pitch centre toward neutral' in pros and 'slow the rate a little' in pros, 'Concerned reassurance acoustics')
need('lower.contains("genuinely excited")' in pros, 'Strong Excited intensity recognition')
need('lower.starts_with("mildly excited and interested")' in pros, 'Subtle Excited intensity recognition')
need('lower.contains("clearly worried")' in pros, 'Strong Concerned intensity recognition')
concerned=pros[pros.index('fn concerned_profile'):pros.index('/// Compile a VoxGen-managed')]
need('short_concerned_guard(text)' in concerned and 'short_friendly_guard(text)' not in concerned,
     'Concerned uses its own anti-panic short-line guard')
need('cfg_delta: 0.10, demo_gain_multiplier: 1.0' in pros, 'Conservative Concerned CFG tuning')
need('"excited_delta": 0.0' in http and '"concerned_delta": 0.10' in http, 'health tuning diagnostics')
need('"version": 10' in http, 'managed prosody v10 health metadata')
print('v0.7.60 Excited/Concerned prosody validation passed')
