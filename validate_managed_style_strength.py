from pathlib import Path
import math
root=Path(__file__).resolve().parent
pros=(root/'src/prosody_control.rs').read_text()
http=(root/'src/http.rs').read_text()
demo=(root/'demo/src/main.rs').read_text()
cargo=(root/'Cargo.toml').read_text()
demo_cargo=(root/'demo/Cargo.toml').read_text()

def need(cond,msg):
    if not cond: raise SystemExit('FAIL: '+msg)

need('version = "0.7.55"' in cargo and 'version = "0.7.55"' in demo_cargo, 'v0.7.55 package versions')
need('cfg_delta: 0.20, demo_gain_multiplier: 1.05' in pros, 'Warm +0.20 CFG / +5% demo gain tuning')
need('cfg_delta: 0.10, demo_gain_multiplier: 1.0' in pros, 'subtle Cheerful +0.10 CFG tuning')
need('cfg_delta: 0.15, demo_gain_multiplier: 0.95' in pros, 'Gentle normal +0.15 CFG / 0.95x demo tuning')
need('cfg_delta: 0.10, demo_gain_multiplier: 0.98' in pros, 'Gentle subtle +0.10 CFG / 0.98x demo tuning')
need('cfg_delta: 0.15, demo_gain_multiplier: 0.92' in pros, 'Gentle strong +0.15 CFG / 0.92x demo tuning')
need('(base_cfg + tuning.cfg_delta).clamp(1.0, 3.0)' in pros, 'managed CFG cap')
need('Some(explicit) => explicit' in http and 'None => apply_managed_cfg(cfg, r.control.as_deref())' in http,
     'HTTP preserves explicit CFG and tunes only omitted CFG')
need('X-VoxGen-CFG' in http, 'effective HTTP CFG diagnostics')
need('Style control (effective):' in demo, 'demo logs effective control')
need('refine_control_instruction(raw, text)' in demo, 'demo resolves actual control before request')
need('Managed style guidance: base CFG' in demo, 'demo logs base/effective CFG')
need('Managed style level: base gain' in demo, 'demo logs base/effective gain')
need('"gain": effective_gain' in demo, 'Warm lift is sent in VoxGen request')
need('cfg.gain_percent = gain_percent' in demo, 'settings store base gain, not managed lift')
need(math.isclose(1.05, 10**(0.423785981398762/20), rel_tol=1e-9), '+5% is about +0.42 dB')
print('v0.7.55 managed style strength validation passed')
