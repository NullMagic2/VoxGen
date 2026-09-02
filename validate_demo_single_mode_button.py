from pathlib import Path
root=Path(__file__).resolve().parent
src=(root/'demo/src/main.rs').read_text()
root_cargo=(root/'Cargo.toml').read_text()
demo_cargo=(root/'demo/Cargo.toml').read_text()

def need(cond,msg):
    if not cond: raise SystemExit(f"FAIL: {msg}")

need('version = "0.7.37"' in root_cargo, 'root version')
need('version = "0.7.37"' in demo_cargo, 'demo version')
need('with_label("Load VoxCPM2")' in src, 'single model/mode button label')
need('with_label("Apply mode + reload")' not in src, 'second mode button removed')
need('apply_mode_button' not in src, 'second mode button handler removed')
need('load_models_button.on_click' in src, 'unified button handler')
need('let mode_mismatch = engine_check()' in src, 'active-vs-selected mode comparison')
need('if mode_mismatch' in src and 'stop_existing_voxgen_server(&state)' in src, 'conditional engine restart')
need('.and_then(|_| ensure_server(&state, stream_enabled, default_gain, &engine_mode))' in src, 'selected mode applied after conditional restart')
need('.and_then(|_| load_models(&base, &acoustic))' in src, 'models loaded by same action')
need('Click Load VoxCPM2 to switch mode and reload the models.' in src, 'mode selector guidance')
print('PASS: single conditional model/mode button validation')
