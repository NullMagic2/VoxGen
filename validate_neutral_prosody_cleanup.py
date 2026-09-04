from pathlib import Path

ROOT = Path(__file__).resolve().parent
cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
demo = (ROOT / "demo" / "Cargo.toml").read_text(encoding="utf-8")
prosody = (ROOT / "src" / "prosody_control.rs").read_text(encoding="utf-8")
http = (ROOT / "src" / "http.rs").read_text(encoding="utf-8")
readme = (ROOT / "README.md").read_text(encoding="utf-8")

def need(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL: {msg}")

need('version = "0.7.60"' in cargo and 'version = "0.7.60"' in demo, "v0.7.60 package versions")
need('fn neutral_profile(' in prosody, "dedicated Neutral compiler")
need('fn short_neutral_guard(' in prosody, "Neutral short-line guard")
need('Neutral, natural conversational speech with no deliberately imposed emotional colour.' in prosody, "normal Neutral target")
need("Preserve the speaker's habitual pitch centre" in prosody, "habitual pitch baseline")
need('clear lexical stress' in prosody and 'syntax-driven phrase contours' in prosody, "linguistic prosody retained")
need('do not flatten the melody' in prosody, "anti-flatness guard")
need('Deliberately affect-neutral while still fully natural and human.' in prosody, "strong Neutral target")
need('do not force a deep register' in prosody, "anti-forced-low-pitch guard")
need('managed_neutral' in prosody, "managed Neutral recognition")
need('natural, conversational and emotionally balanced' in prosody, "legacy Normal Neutral compatibility")
need('clear, composed and deliberately neutral' in prosody, "legacy Strong Neutral compatibility")
need('ManagedStyleTuning::default()' in prosody, "neutral/default tuning remains available")
need('"version": 10' in http, "managed prosody v10 health metadata")
need('"neutral"' in http and 'natural-linguistic-prosody-without-imposed-affect-not-flatness' in http, "Neutral health semantics")
need('Neutral receives **no automatic CFG delta and no demo gain multiplier**' in readme, "README Neutral tuning policy")
print("v0.7.60 Neutral prosody cleanup validation passed")
