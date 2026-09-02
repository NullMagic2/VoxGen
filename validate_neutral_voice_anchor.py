from pathlib import Path
root = Path(__file__).resolve().parent
demo = (root / "demo/src/main.rs").read_text(encoding="utf-8")
root_cargo = (root / "Cargo.toml").read_text(encoding="utf-8")
demo_cargo = (root / "demo/Cargo.toml").read_text(encoding="utf-8")

def need(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL: {msg}")

need('version = "0.7.37"' in root_cargo, 'root version 0.7.37')
need('version = "0.7.37"' in demo_cargo, 'demo version 0.7.37')
need('enum ReferenceSource' in demo and 'NeutralAnchor' in demo, 'neutral-anchor resolution type')
need('fn resolve_reference_sample(' in demo, 'central reference resolver')
need('cfg.emotion_references.get("neutral")' in demo, 'explicit neutral preset is consulted')
need('source: ReferenceSource::NeutralAnchor' in demo, 'neutral path is classified as the anchor')
need('anchoring identity to neutral sample' in demo, 'visible neutral fallback diagnostic')
need('Refusing zero-shot voice generation' in demo, 'missing configured anchor refuses hallucinated fallback')
need('No neutral voice anchor is available, so VoxGen will not invent a replacement voice.' in demo, 'missing preset without fallback is explicit')
need('if out.voice_sample.is_none()' in demo and 'out.emotion_references.get("neutral")' in demo, 'old settings migrate neutral to legacy default')
need(demo.count('resolve_reference_sample(') >= 6, 'resolver is shared by Speak/diagnostics/prewarm/style paths')
need('let sample = sample_override;' in demo, 'synthesis does not bypass central resolver with hidden state fallback')
need('sample_override.or_else(|| state.lock()' not in demo, 'old implicit synthesis fallback removed')
need('if preset == "neutral"' in demo and 'cfg.voice_sample = Some(path.clone());' in demo, 'selecting neutral updates default anchor')
need('Cleared the neutral voice anchor.' in demo, 'neutral clearing semantics are explicit')
# Zero-shot is intentionally retained only for the truly unconfigured case.
need('if sample.is_none() && expressive.clone_mode == "reference"' in demo, 'unconfigured zero-shot compatibility remains')
need('expressive.clone_mode = "auto".to_string();' in demo, 'unconfigured zero-shot remains explicit')
print('PASS: v0.7.37 preset-miss -> neutral anchor policy prevents configured voices from silently falling into zero-shot')
