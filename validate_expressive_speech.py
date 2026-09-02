from pathlib import Path

root=Path(__file__).resolve().parent
main=(root/'src/main.rs').read_text()
rt=(root/'src/runtime.rs').read_text()
http=(root/'src/http.rs').read_text()
demo=(root/'demo/src/main.rs').read_text()
root_cargo=(root/'Cargo.toml').read_text()
demo_cargo=(root/'demo/Cargo.toml').read_text()
readme=(root/'README.md').read_text()

def need(c,m):
    if not c: raise AssertionError(m)

need('version = "0.7.37"' in root_cargo,'root version')
need('version = "0.7.37"' in demo_cargo,'demo version')

# Native VoxCPM2 textual control, without rewriting the user's target text.
for token in [
    'pub text:String, pub control:Option<String>',
    'let controlled_text=if let Some(c)=control{format!("({c}){text}")}else{text.to_owned()};',
    'VoxCPM2 style control cannot be combined',
    'pub text:String, pub control:Option<String>',
]: need(token in rt, f'runtime expressive contract: {token}')

# CLI modes, aliases and multi-seed candidates.
for token in [
    'enum CloneModeArg',
    'control: Option<String>',
    'clone_mode: CloneModeArg',
    'variations: u32',
    'CloneModeArg::Reference',
    'CloneModeArg::Ultimate',
    'variation_output_path',
    '0x9E37_79B9_7F4A_7C15',
    'visible_alias = "inference-timesteps"',
    'visible_alias = "cfg-value"',
    'visible_alias = "temperature"',
    'visible_alias = "seed"',
]: need(token in main, f'CLI expressive contract: {token}')
need('--control cannot be combined with prompt/ultimate cloning' in main,'CLI mode exclusion')

# HTTP mirrors the engine controls.
for token in [
    'control: Option<String>',
    'clone_mode: Option<String>',
    '"reference" | "controllable"',
    '"ultimate"',
    'clone_mode=ultimate requires prompt_text',
    'req.control.as_deref()',
]: need(token in http, f'HTTP expressive contract: {token}')

# Demo presets/intensity use natural-language model guidance, not pitch recipes.
for token in [
    'const STYLE_PRESETS:', '"warm", "Warm"', '"excited", "Excited"',
    '"sad", "Sad"', '"concerned", "Concerned"', '"angry", "Angry"',
    '"whisper", "Whisper-like"', '"custom", "Custom"',
    'const INTENSITIES:', '"subtle", "Subtle"', '"strong", "Strong"',
    'fn build_style_control',
    'natural phrase-level variation in emphasis and emotion rather than a fixed tone',
    'enthusiasm rising on important phrases without shouting',
    'becoming gently reassuring',
    'avoiding a constant shouted delivery',
    'Custom instruction:',
]: need(token in demo, f'demo style control: {token}')

# Clone-mode UI, exact transcript, emotional reference profiles.
for token in [
    'Controllable reference', 'Ultimate cloning', 'Transcript of reference audio:',
    'Set preset reference...', 'Clear preset ref',
    'emotion_references: BTreeMap<String, PathBuf>',
    'emotion_reference.',
    'resolve_reference_sample',
    'clone_mode == "ultimate"',
    'clone_mode_control.on_selection_changed',
    'prompt_text_control_copy.enable(ultimate)',
]: need(token in demo, f'demo cloning/reference contract: {token}')

# Candidate performances + expert generation controls.
for token in [
    'DEFAULT_VARIATIONS: u32 = 1', 'MAX_VARIATIONS: u32 = 3',
    'Variations:', 'CFG (%):', 'Temperature (%):', 'CFM steps:',
    'cfg_value: f32', 'temperature: f32', 'inference_timesteps: u32',
    'combine_pcm16_wavs',
]: need(token in demo, f'demo variation/generation contract: {token}')

# Portable settings include all expressive state.
for key in [
    'style_preset={}', 'style_intensity={}', 'custom_control={}',
    'clone_mode={}', 'prompt_text={}', 'variations={}', 'cfg={:.2}',
    'temperature={:.2}', 'inference_timesteps={}', 'emotion_reference.{key}={}',
]: need(key in demo, f'settings key: {key}')

# Preserve the prior ownership regression fix.
need('let speak_live_playback_controls = live_playback_controls.clone();' in demo,'Speak controls pre-clone')
need('speak_live_playback_controls.clone()' in demo,'Speak uses dedicated controls clone')

# Release should not regress into carrying patch-note artifacts.
need(not list(root.rglob('PATCH_*.md')), 'PATCH_*.md files present')
need(not list(root.rglob('*.patch')), '*.patch files present')

for rel in [
    'build_windows/smoke_expressive_control.bat',
    'build_linux/smoke_expressive_control.sh',
    'build_windows/smoke_ultimate_clone.bat',
    'build_linux/smoke_ultimate_clone.sh',
]: need((root/rel).is_file(),f'missing {rel}')

need('--control' in readme and '--clone-mode ultimate' in readme and '--variations 3' in readme,'README expressive examples')
print('Expressive speech validation passed.')
