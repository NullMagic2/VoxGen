from pathlib import Path

root = Path(__file__).resolve().parent
main = (root / "demo" / "src" / "main.rs").read_text(encoding="utf-8")
root_cargo = (root / "Cargo.toml").read_text(encoding="utf-8")
demo_cargo = (root / "demo" / "Cargo.toml").read_text(encoding="utf-8")

def need(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL: {msg}")

need('version = "0.7.39"' in root_cargo, 'root version')
need('version = "0.7.39"' in demo_cargo, 'demo version')
need('fn update_emotion_sample_button' in main, 'dynamic sample button helper')
need('format!("Select {emotion} sample...")' in main, 'emotion-specific button label')
need('button.set_min_size(Size::new(-1, -1));' in main, 'old minimum reset before resize')
need('let best = button.get_best_size();' in main, 'native best-size calculation')
need('button.set_min_size(best);' in main and 'button.set_size(best);' in main, 'button best size applied')
need('panel.layout();' in main, 'layout refreshed after label resize')
need('style_control.on_selection_changed' in main, 'label updates when emotion changes')
need('cfg.emotion_references.insert(preset.to_string(), path.clone());' in main, 'bottom button stores selected preset reference')
need('if preset == "neutral"' in main and 'cfg.voice_sample = Some(path.clone());' in main, 'neutral sample becomes fallback voice')
need('dialog.set_message(&format!("Reference for {preset_label} emotion:"));' in main, 'emotion-specific picker title')
need('with_label("Select voice sample...")' not in main, 'obsolete generic sample label removed')
print('emotion sample button validation passed')
