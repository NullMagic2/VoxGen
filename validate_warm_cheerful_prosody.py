from pathlib import Path

root = Path(__file__).resolve().parent
cargo = (root / 'Cargo.toml').read_text()
demo_cargo = (root / 'demo/Cargo.toml').read_text()
lib = (root / 'src/lib.rs').read_text()
prosody = (root / 'src/prosody_control.rs').read_text()
runtime = (root / 'src/runtime.rs').read_text()
http = (root / 'src/http.rs').read_text()
demo = (root / 'demo/src/main.rs').read_text()


def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

need('version = "0.7.55"' in cargo and 'version = "0.7.55"' in demo_cargo,
     'v0.7.55 package versions')
need('pub mod prosody_control;' in lib, 'prosody compiler exported by engine crate')
need('prosody_control::refine_control_instruction' in runtime,
     'runtime imports engine prosody compiler')
need('let effective_control=control.map(|c|refine_control_instruction(c,text));' in runtime,
     'runtime compiles control before tokenization')
need('effective_control.as_deref()' in runtime,
     'effective control is used for model text')
need('build_style_control' in demo and 'prosody_control::{' in demo,
     'demo consumes shared engine recipe builder')
need('fn build_style_control(' not in demo,
     'demo-local style recipe implementation removed')
need('native_managed_prosody' in http and '"version": 8' in http and 'short_utterance_guard' in http,
     'health advertises current managed prosody support')
need('low-arousal-affiliative-tender' in http and 'subtle_positive_cue_floor' in http,
     'health advertises warmth semantics and subtle cue floor')

for token in [
    'Mildly cheerful and optimistic, clearly positive but not excited.',
    'light audible smile',
    'slightly brighter resonance',
    'small but definite increase in pitch variation',
    'gently quicker lighter rhythm',
    'Gently warm, kind and close',
    "speaker's natural pitch centre",
    'slightly slower and softer than neutral',
    'smooth connected phrasing',
    'gently lengthened stressed vowels',
    'mellow full resonance',
    'faint smile',
    'friendly brightness, not extra loudness',
]:
    need(token in prosody, f'missing research-guided cue: {token}')

need('lower.starts_with("lightly cheerful and optimistic")' in prosody,
     'only managed cheerful recipe is rewritten')
need('lower.starts_with("slightly warm and friendly")' in prosody,
     'only managed warm recipe is rewritten')
need('return trimmed.to_owned();' in prosody,
     'custom controls have an exact-preservation path')
need('custom_controls_are_untouched' in prosody,
     'custom control preservation regression test')
need('subtle_cheerful_gets_low_arousal_positive_cues' in prosody,
     'subtle cheerful regression test')
need('subtle_warm_gets_stable_social_cues' in prosody,
     'subtle warm regression test')

# Keep managed prompts concise enough that control does not dominate short target text.
subtle_cheer = next(line.strip().strip('"') for line in prosody.splitlines()
                    if 'Mildly cheerful and optimistic, clearly positive but not excited.' in line)
subtle_warm = next(line.strip().strip('"') for line in prosody.splitlines()
                   if 'Gently warm, kind and close' in line)
need(len(subtle_cheer.split()) <= 60, 'subtle cheerful profile remains concise')
need(len(subtle_warm.split()) <= 65, 'subtle warm profile remains concise')

print('v0.7.55 warm/cheerful managed prosody validation passed')
