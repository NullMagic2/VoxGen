from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent

def need(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL: {msg}")

root_main = (ROOT / 'src/main.rs').read_text(encoding='utf-8')
lib = (ROOT / 'src/lib.rs').read_text(encoding='utf-8')
demo_main = (ROOT / 'demo/src/main.rs').read_text(encoding='utf-8')
http = (ROOT / 'src/http.rs').read_text(encoding='utf-8')
cargo = (ROOT / 'Cargo.toml').read_text(encoding='utf-8')
demo_cargo = (ROOT / 'demo/Cargo.toml').read_text(encoding='utf-8')

need('version = "0.7.60"' in cargo and 'version = "0.7.60"' in demo_cargo,
     'root/demo package version is not v0.7.60')
for name, text in [('binary', root_main), ('library', lib), ('demo', demo_main)]:
    m = re.match(r'#!\[recursion_limit\s*=\s*"(\d+)"\]', text)
    need(m is not None, f'{name} crate root has no recursion_limit attribute')
    need(int(m.group(1)) >= 256, f'{name} recursion_limit is below 256')

need('Ok(json!({' in http, '/health no longer uses the large json! response shape expected by this regression test')
need('"automatic_continuity"' in http, 'health automatic continuity metadata missing')
need('speed_transition' in http or 'pace' in http, 'health pace-transition metadata missing')
print('v0.7.60 JSON macro recursion-limit build regression validation passed')
