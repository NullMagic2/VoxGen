from pathlib import Path

root = Path(__file__).resolve().parent
main = (root / 'demo' / 'src' / 'main.rs').read_text(encoding='utf-8')
http = (root / 'src' / 'http.rs').read_text(encoding='utf-8')
root_cargo = (root / 'Cargo.toml').read_text(encoding='utf-8')
demo_cargo = (root / 'demo' / 'Cargo.toml').read_text(encoding='utf-8')

def need(cond, msg):
    if not cond:
        raise SystemExit(f'FAIL: {msg}')

need('version = "0.7.39"' in root_cargo, 'root version')
need('version = "0.7.39"' in demo_cargo, 'demo version')
need('("normal", "Normal")' in main, 'compact Normal UI label')
need('("xtx7900", "XTX 7900")' in main, 'compact XTX 7900 UI label')
need('Normal (--mode normal)' not in main, 'old verbose Normal UI label removed')
need('XTX 7900 optimized (--mode xtx7900)' not in main, 'old verbose XTX UI label removed')
need('with_label("Apply mode + reload")' not in main, 'obsolete apply/reload button removed')
need('apply_mode_button' not in main, 'obsolete apply/reload handler removed')
need('load_models_button.on_click' in main, 'single Load VoxCPM2 handler')
need('if mode_mismatch' in main and 'stop_existing_voxgen_server(&state)' in main, 'single load action restarts only on mode mismatch')
need('stop_existing_voxgen_server(&state)' in main, 'mode switch takes over an existing VoxGen listener')
need('server_execution_mode().as_deref() != Some(engine_mode.as_str())' in main, 'manual model load detects mode mismatch')
need('unwrap_or("normal")' in main, 'legacy no-mode server treated as Normal')
need('windows_listener_pid(PORT)' in main, 'Windows legacy port-owner recovery')
need('listing.contains("voxgen")' in main, 'legacy process image verified before kill')
need('Command::new("taskkill")' in main, 'Windows legacy VoxGen termination fallback')
need('("POST", "/v1/server/shutdown")' in http, 'graceful server shutdown endpoint')
need('addr.ip().is_loopback()' in http, 'shutdown endpoint restricted to loopback')
need('"pid": std::process::id()' in http, 'health exposes server PID')
need('.arg("--mode")' in main and '.arg(engine_mode)' in main, 'server launch passes --mode')
need('.and_then(|_| load_models(&base, &acoustic))' in main, 'single load action reloads selected models')
need('cfg.engine_mode = mode.clone()' in main, 'mode persistence update')
need('mode_control.enable(false)' in main and 'mode_control.enable(true)' in main, 'mode selector protected during unified load')
need('speak_button.enable(false)' in main and 'speak_button.enable(true)' in main, 'speak disabled until reload completes')
print('PASS: demo execution-mode restart/takeover validation')
