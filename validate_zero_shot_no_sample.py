from pathlib import Path
root = Path(__file__).resolve().parent
src = (root / "demo/src/main.rs").read_text()
root_cargo = (root / "Cargo.toml").read_text()
demo_cargo = (root / "demo/Cargo.toml").read_text()

def need(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL: {msg}")

need('version = "0.7.60"' in root_cargo, 'root version')
need('version = "0.7.60"' in demo_cargo, 'demo version')
need('if sample.is_none() && expressive.clone_mode == "reference"' in src, 'reference-without-sample fallback guard')
need('expressive.clone_mode = "auto".to_string();' in src, 'zero-shot clone mode fallback')
need('No reference sample selected: using zero-shot generation.' in src, 'zero-shot log message')
need('Ultimate cloning requires a voice/reference WAV.' in src, 'ultimate mode remains strict')
need('VoxGen closed the streaming connection without an HTTP response.' in src, 'empty HTTP status diagnostic')
print('PASS: no-sample reference mode falls back to zero-shot while Ultimate remains strict')
