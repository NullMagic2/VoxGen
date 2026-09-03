from pathlib import Path
root=Path(__file__).resolve().parent
prosody=(root/'src/prosody_control.rs').read_text(encoding='utf-8')
http=(root/'src/http.rs').read_text(encoding='utf-8')
cargo=(root/'Cargo.toml').read_text(encoding='utf-8')
demo=(root/'demo/Cargo.toml').read_text(encoding='utf-8')

def need(cond,msg):
    if not cond: raise SystemExit('FAIL: '+msg)

need('version = "0.7.55"' in cargo and 'version = "0.7.55"' in demo, 'v0.7.55 package versions')
need('fn sad_profile(' in prosody and 'fn serious_profile(' in prosody, 'dedicated Sad/Serious compilers')
need('moderately lower pitch centre' in prosody and 'noticeably narrower pitch range' in prosody, 'Sad pitch cues')
need('slower phrasing' in prosody and 'softer intensity' in prosody, 'Sad temporal/intensity cues')
need('not tired, bored, depressed' in prosody or 'do not sound sleepy, bored' in prosody, 'Sad anti-sleepy/flat guard')
need('never wail, sob, break the voice' in prosody, 'Sad anti-grief guard')
need('conveying commitment rather than a separate negative emotion' in prosody, 'Serious stance semantics')
need('natural pitch centre or only very slightly lower' in prosody, 'Serious avoids forced low pitch')
need('decisive falling phrase endings' in prosody, 'Serious resolved intonation')
need('Do not imitate a movie-trailer voice' in prosody, 'Strong Serious anti-theatrical guard')
need('strongly serious, deliberate and focused' in prosody, 'Strong Serious seed replaces grave-authoritative seed')
need('demo_gain_multiplier: 0.97' in prosody and 'demo_gain_multiplier: 0.94' in prosody and 'demo_gain_multiplier: 0.90' in prosody, 'Sad intensity-scaled demo levels')
need('"version": 8' in http, 'managed prosody v8 health metadata')
need('committed-attentional-stance-not-forced-low-pitch' in http, 'Serious health semantics')
need('low-arousal-lower-narrower-softer-slower-not-grief-or-sleepiness' in http, 'Sad health semantics')
print('v0.7.55 Serious/Sad prosody validation passed')
