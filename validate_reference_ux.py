from pathlib import Path

def need(cond, msg):
    if not cond:
        raise SystemExit(f'FAIL: {msg}')

root = Path(__file__).resolve().parent
main = (root/'demo/src/main.rs').read_text(encoding='utf-8')
root_cargo = (root/'Cargo.toml').read_text(encoding='utf-8')
demo_cargo = (root/'demo/Cargo.toml').read_text(encoding='utf-8')
need('version = "0.7.60"' in root_cargo, 'root version')
need('version = "0.7.60"' in demo_cargo, 'demo version')
need('with_label("Transcript of reference audio:")' in main, 'clear reference transcript label')
need('Enter exactly what is spoken in the selected reference WAV. This is not the text to synthesize.' in main, 'reference transcript tooltip')
need('prompt_text_label.set_tooltip(REFERENCE_TRANSCRIPT_TOOLTIP);' in main, 'label tooltip')
need('prompt_text_control.set_tooltip(REFERENCE_TRANSCRIPT_TOOLTIP);' in main, 'text control tooltip')
need('dialog.set_message(&format!("Reference for {preset_label} emotion:"));' in main, 'emotion-specific reference picker prompt')
need('STYLE_PRESETS.get(i as usize).map(|(_, label)| *label)' in main, 'human-readable emotion label')
print('PASS: Ultimate-cloning/reference UX wording is explicit and emotion-specific.')
