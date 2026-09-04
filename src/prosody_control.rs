//! Compact managed-prosody compilation for VoxCPM2.
//!
//! Managed style requests are compiled directly to the final textual control
//! consumed by VoxCPM2. There is deliberately no legacy recipe/suffix layer and
//! no second refinement pass: fewer control tokens mean less TTFA-critical text
//! prefill while preserving the same style/intensity API.

pub const DEFAULT_MANAGED_PROFILE: &str = "neutral";
pub const DEFAULT_MANAGED_INTENSITY: &str = "normal";
pub const MANAGED_PROFILES: &[&str] = &[
    "neutral", "warm", "cheerful", "excited", "concerned",
    "angry", "gentle", "whisper", "serious", "sad",
];
pub const MANAGED_INTENSITIES: &[&str] = &["subtle", "normal", "strong"];

pub fn managed_profile_semantics(profile: &str) -> Option<&'static str> {
    match profile {
        "neutral" => Some("natural-linguistic-prosody-without-imposed-affect-not-flatness"),
        "warm" => Some("low-arousal-affiliative-tender"),
        "cheerful" => Some("positive-upbeat-audible-smile-with-conversational-loudness"),
        "excited" => Some("high-arousal-positive-engagement-not-surprise-with-local-release"),
        "concerned" => Some("attentive-alertness-to-caring-reassurance"),
        "angry" => Some("controlled-cold-anger-tension-timing-not-loudness"),
        "gentle" => Some("low-vocal-effort-low-projection-emotion-preserving"),
        "whisper" => Some("low-effort-near-whisper-airflow-reduced-periodic-voicing"),
        "serious" => Some("committed-attentional-stance-not-forced-low-pitch"),
        "sad" => Some("low-arousal-lower-narrower-softer-slower-not-grief-or-sleepiness"),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManagedStyleTuning {
    pub cfg_delta: f32,
    pub demo_gain_multiplier: f32,
}

impl Default for ManagedStyleTuning {
    fn default() -> Self {
        Self { cfg_delta: 0.0, demo_gain_multiplier: 1.0 }
    }
}

/// Direct tuning lookup. No text parsing is involved.
pub fn managed_style_tuning(style: &str, intensity: &str) -> Option<ManagedStyleTuning> {
    let style = style.trim();
    let intensity = intensity.trim();
    if !MANAGED_PROFILES.contains(&style) || !MANAGED_INTENSITIES.contains(&intensity) {
        return None;
    }
    let tuning = match (style, intensity) {
        ("neutral", _) => ManagedStyleTuning::default(),
        ("warm", _) => ManagedStyleTuning { cfg_delta: 0.20, demo_gain_multiplier: 1.05 },
        ("cheerful", "subtle") => ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 1.0 },
        ("gentle", "subtle") => ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 0.98 },
        ("gentle", "normal") => ManagedStyleTuning { cfg_delta: 0.15, demo_gain_multiplier: 0.95 },
        ("gentle", "strong") => ManagedStyleTuning { cfg_delta: 0.15, demo_gain_multiplier: 0.92 },
        ("concerned", _) => ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 1.0 },
        ("sad", "subtle") => ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 0.97 },
        ("sad", "normal") => ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 0.94 },
        ("sad", "strong") => ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 0.90 },
        ("serious", "subtle") => ManagedStyleTuning { cfg_delta: 0.05, demo_gain_multiplier: 1.0 },
        ("serious", _) => ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 1.0 },
        ("whisper", _) => ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 0.85 },
        ("angry", _) => ManagedStyleTuning { cfg_delta: 0.0, demo_gain_multiplier: 0.90 },
        _ => ManagedStyleTuning::default(),
    };
    Some(tuning)
}

fn intensity_prefix(intensity: &str) -> Option<&'static str> {
    match intensity.trim() {
        "subtle" => Some("Subtly"),
        "normal" => Some("Clearly"),
        "strong" => Some("Strongly"),
        _ => None,
    }
}

fn short_line_guard(style: &str, text: &str) -> &'static str {
    if text.split_whitespace().count() > 14 { return ""; }
    match style {
        "neutral" => " Keep short-line punctuation linguistic, not dramatic.",
        "warm" | "cheerful" | "excited" => " Keep the short line natural; do not build a climax.",
        "concerned" => " Keep concern attentive and controlled, never panicked.",
        "gentle" => " Keep low effort throughout without becoming whispery.",
        "whisper" => " Stay near-whispered throughout; do not project punctuation.",
        "angry" => " Sharpen timing and attacks, not loudness; never shout.",
        "sad" => " Keep the short line reflective, not melodramatic or sleepy.",
        "serious" => " Keep the short line firm, not ominous, monotone or artificially low.",
        _ => "",
    }
}

/// Build the final control text that Runtime tokenizes. The prompt is kept
/// intentionally compact because every control token participates in prefix
/// prefill before acoustic generation can begin.
pub fn build_style_control(style: &str, intensity: &str, custom: &str, text: &str) -> Option<String> {
    if style == "auto" { return None; }
    if style == "custom" {
        let custom = custom.trim();
        if custom.is_empty() { return None; }
        return match intensity.trim() {
            "subtle" => Some(format!("{custom}. Keep the effect subtle.")),
            "strong" => Some(format!("{custom}. Make the effect clear but natural.")),
            "normal" => Some(custom.to_owned()),
            _ => None,
        };
    }
    let degree = intensity_prefix(intensity)?;
    let cue = match style.trim() {
        "neutral" => "natural conversational speech: ordinary pitch, pace, loudness and lexical stress; no imposed affect or monotone",
        "warm" => "warm and close: low effort, smooth phrasing, soft attacks, mellow resonance and a faint smile; conversational loudness",
        "cheerful" => "cheerful and upbeat: audible smile, brighter resonance, buoyant rhythm and smooth pitch variation; never shouted",
        "excited" => "excited and positively engaged: quicker timing, wider dynamic pitch, crisp articulation and brief energy peaks that release",
        "concerned" => "concerned and attentive: mild tension, responsive pitch and focused articulation, relaxing toward reassurance where appropriate",
        "angry" => "controlled cold anger: firm tension, clean hard attacks, compact purposeful pitch and brief sharp emphasis; do not shout",
        "gentle" => "gentle and low-effort: reduced projection, smooth phrasing, light attacks and relaxed articulation; stay clear, not whispery",
        "whisper" => "near-whispered: very low effort, audible airflow, minimal projection, softened attacks and reduced voicing; remain intelligible",
        "serious" => "serious and deliberate: measured pacing, firm articulation, controlled pitch, selective emphasis and resolved endings",
        "sad" => "sad and reflective: slightly lower and narrower pitch, softer energy, slower phrasing and gentle falling endings; stay intelligible",
        _ => return None,
    };
    Some(format!("{degree} {cue}.{}", short_line_guard(style.trim(), text)))
}

pub(crate) const MIN_MOOD_SPEED_PERCENT: f32 = 50.0;
pub(crate) const MAX_MOOD_SPEED_PERCENT: f32 = 200.0;
pub(crate) const MIN_MEANINGFUL_MOOD_SPEED_DELTA_PERCENT: f32 = 5.0;
pub(crate) const MAX_NATURAL_MOOD_SPEED_DELTA_PERCENT: f32 = 45.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MoodSpeedTransition {
    pub(crate) from_percent: f32,
    pub(crate) to_percent: f32,
}

impl MoodSpeedTransition {
    pub(crate) fn new(from_percent: f32, to_percent: f32) -> Result<Self, String> {
        for (label, value) in [("from", from_percent), ("to", to_percent)] {
            if !value.is_finite() || !(MIN_MOOD_SPEED_PERCENT..=MAX_MOOD_SPEED_PERCENT).contains(&value) {
                return Err(format!(
                    "transition {label}_speed_percent must be finite and between {MIN_MOOD_SPEED_PERCENT:.0} and {MAX_MOOD_SPEED_PERCENT:.0}"
                ));
            }
        }
        let delta = (to_percent - from_percent).abs();
        if delta > 0.0 && delta < MIN_MEANINGFUL_MOOD_SPEED_DELTA_PERCENT {
            return Err(format!("speed transition is too subtle ({delta:.1} percentage points)"));
        }
        if delta > MAX_NATURAL_MOOD_SPEED_DELTA_PERCENT {
            return Err(format!("speed transition is too large for one natural phrase ({delta:.1} percentage points)"));
        }
        Ok(Self { from_percent, to_percent })
    }
}

fn canonical_style(value: &str) -> Result<&str, String> {
    let value = value.trim();
    if MANAGED_PROFILES.contains(&value) { Ok(value) }
    else { Err(format!("unsupported managed style '{value}'")) }
}

fn canonical_intensity(value: &str) -> Result<&str, String> {
    let value = value.trim();
    if MANAGED_INTENSITIES.contains(&value) { Ok(value) }
    else { Err(format!("unsupported managed intensity '{value}'")) }
}

fn transition_endpoint(style: &str, intensity: &str) -> String {
    let degree = match intensity { "subtle" => "light", "strong" => "strong", _ => "clear" };
    let cue = match style {
        "neutral" => "neutral conversational prosody",
        "warm" => "warm low-effort smooth delivery",
        "cheerful" => "cheerful smiling buoyant delivery",
        "excited" => "excited dynamic high-arousal delivery",
        "concerned" => "concerned attentive caring delivery",
        "angry" => "controlled tense cold anger",
        "gentle" => "gentle low-effort low-projection delivery",
        "whisper" => "intelligible near-whisper",
        "serious" => "serious deliberate focused delivery",
        "sad" => "sad soft slower reflective delivery",
        _ => unreachable!(),
    };
    format!("{degree} {cue}")
}

fn transition_motion(from: &str, to: &str) -> &'static str {
    match (from, to) {
        ("angry", "serious") => "Release vocal tension and hard attacks; finish focused and composed.",
        ("concerned", "warm") => "Let tension settle; soften attacks and let reassuring warmth emerge.",
        ("sad", "warm") => "Restore pitch variation and energy gradually; let gentle warmth emerge without cheerfulness.",
        ("excited", "neutral") => "Narrow pitch movement and relax quick energy peaks toward ordinary conversation.",
        ("warm", "serious") => "Reduce the smile and softness; firm articulation and focus.",
        ("serious", "warm") => "Release firmness; soften attacks and introduce mellow warmth.",
        ("neutral", "concerned") => "Introduce mild tension, attentive pitch responses and focused articulation.",
        ("concerned", "neutral") => "Release tension and normalize pitch, timing and articulation.",
        ("neutral", "sad") => "Lower and narrow pitch, soften energy and slow phrasing gradually.",
        ("sad", "neutral") => "Restore ordinary pitch range, timing, energy and lexical prominence.",
        ("neutral", "warm") => "Soften attacks, lower effort and introduce a faint affiliative smile.",
        ("warm", "neutral") => "Remove the smile and interpersonal softness while preserving natural movement.",
        ("neutral", "serious") => "Tighten focus with firmer articulation, deliberate pauses and resolved endings.",
        ("serious", "neutral") => "Relax the deliberate stance toward ordinary conversational movement.",
        ("cheerful", "excited") => "Raise arousal with wider quicker pitch movement and brief energy peaks.",
        ("excited", "cheerful") => "Lower arousal while preserving a light smile and buoyant rhythm.",
        ("concerned", "serious") => "Turn urgency into composed focus and remove worry-like tension.",
        ("serious", "concerned") => "Add mild attentive tension and responsive pitch without panic.",
        ("warm", "gentle") => "Reduce the interpersonal smile while lowering effort and projection.",
        ("gentle", "warm") => "Keep softness and add a faint smile, mellow resonance and personal warmth.",
        ("warm", "sad") => "Let warmth recede as pitch lowers, energy softens and phrasing slows.",
        ("sad", "serious") => "Restore enough energy for firm articulation, deliberate pauses and resolved endings.",
        ("serious", "sad") => "Release firmness; lower and narrow pitch and soften energy.",
        ("concerned", "sad") => "Let urgency drain away into slower, softer reflective delivery.",
        ("sad", "concerned") => "Restore attentiveness, responsive pitch and focused articulation without panic.",
        ("excited", "serious") => "Channel energy into focus; narrow pitch excursions and stabilize timing.",
        ("angry", "neutral") => "Release tension and hard attacks toward ordinary conversation.",
        ("angry", "concerned") => "Reduce hostility while keeping urgency and attentive pitch.",
        ("neutral", "whisper") => "Reduce projection and voicing while increasing airflow gradually.",
        ("whisper", "neutral") => "Restore periodic voicing and ordinary projection gradually.",
        ("serious", "whisper") => "Preserve focus while reducing projection, voicing and vocal effort.",
        ("whisper", "serious") => "Restore voicing, projection, firm articulation and deliberate endings.",
        _ => "Release source cues while introducing destination cues continuously.",
    }
}

fn speed_clause(speed: Option<MoodSpeedTransition>) -> String {
    match speed {
        None => String::new(),
        Some(speed) if (speed.to_percent - speed.from_percent).abs() < 0.001 =>
            format!(" Hold speaking pace near {:.0}%.", speed.to_percent),
        Some(speed) => format!(" Change speaking pace smoothly from {:.0}% to {:.0}%.", speed.from_percent, speed.to_percent),
    }
}

fn transition_short_guard(text: &str) -> &'static str {
    if text.split_whitespace().count() <= 10 {
        " Keep this short transition compact and continuous; do not split it into two takes."
    } else {
        ""
    }
}

/// One compact, single-take transition instruction. Automatic continuity has
/// one behavior: continuous evolution. The removed quick/alias/legacy variants
/// were not part of the current public destination API.
pub(crate) fn build_transition_control_with_speed(
    from_style: &str,
    from_intensity: &str,
    to_style: &str,
    to_intensity: &str,
    speed: Option<MoodSpeedTransition>,
    text: &str,
) -> Result<String, String> {
    let from = canonical_style(from_style)?;
    let to = canonical_style(to_style)?;
    let from_intensity = canonical_intensity(from_intensity)?;
    let to_intensity = canonical_intensity(to_intensity)?;
    if from == to && from_intensity == to_intensity && speed.is_none() {
        return Err("transition endpoints are identical".to_string());
    }
    let start = transition_endpoint(from, from_intensity);
    let end = transition_endpoint(to, to_intensity);
    let motion = if from == to {
        if from_intensity == to_intensity {
            "Keep the same style and intensity while changing only speaking pace."
        } else {
            "Keep the same style and change only its strength continuously."
        }
    } else {
        transition_motion(from, to)
    };
    Ok(format!(
        "Begin {start}; smoothly become {end} across the phrase. {motion}{} Keep one speaker and continuous phrase timing.{}",
        speed_clause(speed), transition_short_guard(text)
    ))
}

pub(crate) fn managed_transition_cfg_delta(
    from_style: &str,
    from_intensity: &str,
    to_style: &str,
    to_intensity: &str,
) -> Result<f32, String> {
    let a = managed_style_tuning(canonical_style(from_style)?, canonical_intensity(from_intensity)?)
        .ok_or_else(|| "invalid source managed style".to_string())?.cfg_delta;
    let b = managed_style_tuning(canonical_style(to_style)?, canonical_intensity(to_intensity)?)
        .ok_or_else(|| "invalid destination managed style".to_string())?.cfg_delta;
    Ok(((a + b) * 0.5).clamp(0.0, 0.20))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_controls_are_direct_and_compact() {
        for style in MANAGED_PROFILES {
            for intensity in MANAGED_INTENSITIES {
                let control = build_style_control(style, intensity, "", "A representative sentence.").unwrap();
                assert!(control.split_whitespace().count() <= 32, "{style}/{intensity}: {control}");
                assert!(!control.to_ascii_lowercase().contains("phrase-level variation"));
            }
        }
    }

    #[test]
    fn tuning_is_structured_not_text_parsed() {
        assert_eq!(managed_style_tuning("warm", "normal").unwrap(), ManagedStyleTuning { cfg_delta: 0.20, demo_gain_multiplier: 1.05 });
        assert_eq!(managed_style_tuning("angry", "strong").unwrap().cfg_delta, 0.0);
        assert!(managed_style_tuning("banana", "normal").is_none());
    }

    #[test]
    fn transition_is_compact_and_continuous() {
        let control = build_transition_control_with_speed(
            "angry", "strong", "serious", "normal",
            Some(MoodSpeedTransition::new(105.0, 95.0).unwrap()),
            "We need to change course now.",
        ).unwrap();
        assert!(control.contains("Release vocal tension"));
        assert!(control.contains("105% to 95%"));
        assert!(control.contains("one speaker"));
        assert!(control.split_whitespace().count() <= 55);
    }
}
