from pathlib import Path

root = Path(__file__).resolve().parent
prosody = (root / 'src/prosody_control.rs').read_text(encoding='utf-8')
http = (root / 'src/http.rs').read_text(encoding='utf-8')

def need(cond, msg):
    if not cond:
        raise SystemExit(f'FAIL: {msg}')

# Whisper-like must target phonation/airflow rather than mere volume.
need('fn whisper_profile' in prosody, 'whisper profile exists')
need('audible airflow/noise' in prosody, 'whisper airflow cue')
need('reduced periodic voicing' in prosody, 'whisper reduced-periodicity cue')
need('not simply like normal speech played quietly' in prosody, 'whisper is not quiet modal speech')
need('As close to a true whisper as the cloned voice can naturally sustain' in prosody,
     'strong whisper remains a controlled near-whisper target')
need('do not project, raise loudness' in prosody, 'short whisper exclamation anti-projection guard')

# Controlled anger must target cold/suppressed anger rather than hot/shouted anger.
need('fn angry_profile' in prosody, 'angry profile exists')
need('closer to cold anger than explosive hot anger' in prosody, 'cold anger semantics')
need('compact purposeful pitch movement' in prosody, 'controlled anger pitch cue')
need('hard clean attacks' in prosody, 'controlled anger articulation cue')
need('moderate sustained loudness' in prosody, 'controlled anger loudness cue')
need('rather than shouting or continuous volume' in prosody, 'anger anti-shouting cue')
need('let punctuation sharpen timing and attack, not loudness' in prosody,
     'short angry punctuation anti-shout guard')

# Steering/level policy: small whisper CFG only, no angry CFG escalation.
need('cfg_delta: 0.10, demo_gain_multiplier: 0.85' in prosody,
     'whisper managed tuning is modest and lower-level')
need('cfg_delta: 0.0, demo_gain_multiplier: 0.90' in prosody,
     'angry managed tuning avoids CFG escalation and trims demo level')

# Health metadata documents the new behavior.
need('"version": 10' in http, 'managed prosody v10 health metadata')
need('controlled-cold-anger-tension-timing-not-loudness' in http, 'angry health semantics')
need('low-effort-near-whisper-airflow-reduced-periodic-voicing' in http, 'whisper health semantics')
need('"whisper_delta": 0.10' in http and '"angry_delta": 0.0' in http,
     'health guidance deltas')

print('whisper/angry prosody validation OK')
