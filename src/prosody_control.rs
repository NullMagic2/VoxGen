//! Research-guided compilation of VoxCPM2 style-control instructions.
//!
//! VoxCPM2 accepts free-form text in the `(instruction)target text` form.  The
//! applications around VoxGen historically generated broad labels such as
//! "lightly cheerful" plus a generic request for phrase-level variation.  Those
//! labels are valid, but they do not tell the model which low-arousal acoustic
//! cues distinguish subtle positive affect from neutral or serious speech.
//!
//! This module keeps arbitrary user-authored controls untouched.  It only
//! recognizes VoxGen-managed recipes (identified by the legacy managed suffix)
//! and compiles selected managed variants into concise acoustic goals.

const LEGACY_MANAGED_SUFFIX: &str =
    "with natural phrase-level variation in emphasis and emotion rather than a fixed tone";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManagedStyleTuning {
    /// Small additive CFG adjustment for managed low-arousal styles.
    pub cfg_delta: f32,
    /// Demo-only request gain multiplier. The engine HTTP API never applies this
    /// implicitly; external clients retain explicit control of their gain.
    pub demo_gain_multiplier: f32,
}

impl Default for ManagedStyleTuning {
    fn default() -> Self {
        Self { cfg_delta: 0.0, demo_gain_multiplier: 1.0 }
    }
}

/// Return conservative steering metadata for VoxGen-owned style recipes.
/// Arbitrary/custom instructions are never tuned.
pub fn managed_style_tuning(control: Option<&str>) -> ManagedStyleTuning {
    let Some(control) = control.map(str::trim).filter(|x| !x.is_empty()) else {
        return ManagedStyleTuning::default();
    };
    let lower = control.to_ascii_lowercase();
    if !lower.contains(LEGACY_MANAGED_SUFFIX) {
        return ManagedStyleTuning::default();
    }

    if lower.starts_with("subtly neutral and conversational")
        || lower.starts_with("neutral and conversational")
        || lower.starts_with("deliberately affect-neutral")
        || lower.starts_with("natural and conversational, emotionally restrained")
        || lower.starts_with("natural, conversational and emotionally balanced")
        || lower.starts_with("clear, composed and deliberately neutral")
    {
        // Neutral is the untouched acoustic reference level. Its cleanup is
        // entirely in the conditioning language, never hidden CFG or gain.
        return ManagedStyleTuning::default();
    }

    if lower.starts_with("slightly warm and friendly")
        || lower.starts_with("warm and friendly")
        || lower.starts_with("very warm, affectionate and welcoming")
    {
        return ManagedStyleTuning { cfg_delta: 0.20, demo_gain_multiplier: 1.05 };
    }
    if lower.starts_with("lightly cheerful and optimistic") {
        return ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 1.0 };
    }
    if lower.starts_with("slightly softer and gentler than normal") {
        return ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 0.98 };
    }
    if lower.starts_with("gentle, low-effort and calm")
        || lower.starts_with("gentle, calm and reassuring")
    {
        return ManagedStyleTuning { cfg_delta: 0.15, demo_gain_multiplier: 0.95 };
    }
    if lower.starts_with("very gentle, low-effort and unprojected")
        || lower.starts_with("very gentle, tender and soothing")
    {
        return ManagedStyleTuning { cfg_delta: 0.15, demo_gain_multiplier: 0.92 };
    }
    if lower.starts_with("slightly concerned and attentive")
        || lower.starts_with("quietly concerned at first")
        || lower.starts_with("clearly worried at first")
    {
        // Concern is a mixed low/moderate-arousal state. A small guidance lift
        // helps the model retain the alert->reassuring contour without pushing
        // it toward panic or sustained high arousal.
        return ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 1.0 };
    }
    if lower.starts_with("slightly subdued and reflective") {
        return ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 0.97 };
    }
    if lower.starts_with("subdued and reflective") {
        return ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 0.94 };
    }
    if lower.starts_with("deeply saddened and reflective") {
        return ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 0.90 };
    }
    if lower.starts_with("slightly serious and focused") {
        return ManagedStyleTuning { cfg_delta: 0.05, demo_gain_multiplier: 1.0 };
    }
    if lower.starts_with("serious, composed and focused") {
        return ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 1.0 };
    }
    if lower.starts_with("strongly serious, deliberate and focused")
        || lower.starts_with("grave, authoritative and focused")
    {
        return ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 1.0 };
    }
    if lower.starts_with("soft, intimate and breathy")
        || lower.starts_with("whisper-like, soft and intimate")
        || lower.starts_with("very soft and whisper-like")
    {
        // Whisper-like delivery is a low-vocal-effort phonation target rather
        // than an emotion. A small guidance lift helps the model preserve
        // airflow/near-whisper cues; the demo level reduction reflects the
        // lower acoustic intensity of whispered speech without changing the
        // engine/API gain chosen by external clients.
        return ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 0.85 };
    }
    if lower.starts_with("restrained irritation")
        || lower.starts_with("clearly angry but controlled")
        || lower.starts_with("strong controlled anger")
    {
        // Controlled/cold anger must remain perceptible through tension, timing
        // and articulation rather than a hidden CFG or loudness escalation.
        return ManagedStyleTuning { cfg_delta: 0.0, demo_gain_multiplier: 0.90 };
    }

    ManagedStyleTuning::default()
}

/// Apply the managed CFG delta to a base value while staying in VoxCPM2's
/// supported/recommended demo range.
pub fn apply_managed_cfg(base_cfg: f32, control: Option<&str>) -> f32 {
    let tuning = managed_style_tuning(control);
    (base_cfg + tuning.cfg_delta).clamp(1.0, 3.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManagedIntensity {
    Subtle,
    Normal,
    Strong,
}

fn managed_intensity(lower: &str) -> ManagedIntensity {
    if lower.contains("slightly ")
        || lower.contains("lightly ")
        || lower.contains("subtle")
        || lower.contains("restrained")
        || lower.starts_with("pleasantly surprised")
        || lower.starts_with("mildly excited and interested")
        || lower.starts_with("soft, intimate and breathy")
    {
        ManagedIntensity::Subtle
    } else if lower.contains("very ")
        || lower.contains("strong ")
        || lower.contains("clearly delighted")
        || lower.contains("genuinely pleased")
        || lower.contains("genuinely excited")
        || lower.contains("clearly worried")
        || lower.contains("deeply saddened")
        || lower.starts_with("strongly serious")
        || lower.starts_with("grave, authoritative")
        || lower.starts_with("deliberately affect-neutral")
        || lower.starts_with("clear, composed and deliberately neutral")
        || lower.starts_with("very soft and whisper-like")
        || lower.contains("lively")
    {
        ManagedIntensity::Strong
    } else {
        ManagedIntensity::Normal
    }
}

fn short_friendly_guard(text: &str) -> &'static str {
    let words = text.split_whitespace().count();
    if words <= 14 && text.contains('!') {
        " On this short line, treat the exclamation mark as friendly brightness, not extra loudness or a sharp pitch jump."
    } else if words <= 14 {
        " On this short line, keep the positive tone continuous and do not build to a dramatic climax."
    } else {
        ""
    }
}

fn short_concerned_guard(text: &str) -> &'static str {
    let words = text.split_whitespace().count();
    if words <= 14 && (text.contains('!') || text.contains('?')) {
        " On this short line, let punctuation signal attentive concern, not panic, shouting, pleading, or a dramatic pitch spike."
    } else if words <= 14 {
        " On this short line, keep concern controlled and perceptible without inventing a dramatic escalation."
    } else {
        ""
    }
}

fn short_neutral_guard(text: &str) -> &'static str {
    let words = text.split_whitespace().count();
    if words <= 14 && (text.contains('!') || text.contains('?')) {
        " On this short line, let punctuation shape ordinary linguistic emphasis or question intonation only; do not turn it into excitement, anger, concern, warmth, or theatricality."
    } else if words <= 14 {
        " On this short line, keep ordinary conversational movement and lexical stress; do not over-interpret the wording as a distinct emotion or flatten it into monotone."
    } else {
        ""
    }
}

fn neutral_profile(intensity: ManagedIntensity, text: &str) -> String {
    // Neutral is a reference baseline, not a request for flat or emotionless
    // speech. Preserve the cloned speaker's habitual prosody and linguistic
    // prominence while avoiding a deliberately imposed affective stance.
    let base = match intensity {
        ManagedIntensity::Subtle => {
            "Natural conversational baseline with minimal imposed affect. Preserve the speaker's habitual pitch centre, ordinary pitch range, normal speaking rate and loudness, clear lexical stress, and small spontaneous timing variations. Let syntax and punctuation shape normal rises, falls, and prominence without adding a distinct emotional colour. Sound relaxed and human rather than controlled or blank; do not deliberately become warm, cheerful, sad, angry, concerned, serious, sleepy, or theatrical."
        }
        ManagedIntensity::Normal => {
            "Neutral, natural conversational speech with no deliberately imposed emotional colour. Preserve the speaker's habitual pitch centre, ordinary pitch variability, normal rate and loudness, clear lexical stress, syntax-driven phrase contours, and subtle human timing variation. Emphasize words for meaning rather than emotion, and let sentence type determine ordinary rises and falls. Do not flatten the melody, suppress prosody, or drift intentionally toward warmth, cheerfulness, sadness, anger, concern, seriousness, sleepiness, or theatrical delivery."
        }
        ManagedIntensity::Strong => {
            "Deliberately affect-neutral while still fully natural and human. Keep the speaker's habitual pitch centre, normal conversational loudness and rate, and a moderate living pitch range; reduce emotion-specific exaggeration while preserving lexical stress, contrastive emphasis, syntax-driven intonation, and small timing variation. Sound composed without becoming formal, cold, robotic, flat, or monotonous, and do not force a deep register or any identifiable emotional stance."
        }
    };
    format!("{base}{}", short_neutral_guard(text))
}

fn warm_profile(intensity: ManagedIntensity, text: &str) -> String {
    // Warmth is treated as low-arousal affiliative/tender speech, not as a
    // weaker version of joy.  Preserve the cloned speaker's habitual pitch
    // centre and make warmth audible mainly through low vocal effort, smooth
    // timing, softened attacks, mellow resonance and close social delivery.
    let base = match intensity {
        ManagedIntensity::Subtle => {
            "Gently warm, kind and close, like speaking to someone familiar. Keep the speaker's natural pitch centre; speak slightly slower and softer than neutral, with smooth connected phrasing, soft attacks, gently lengthened stressed vowels, rounded intonation, and a mellow full resonance with only a faint smile. Clearly caring, never cheerful, solemn, sleepy, breathy, or flat."
        }
        ManagedIntensity::Normal => {
            "Warm, friendly and reassuring, with personal closeness rather than generic politeness. Keep the speaker's natural pitch centre and low vocal effort; use a slightly unhurried pace, smooth phrasing, soft attacks, gently lengthened important vowels, rounded intonation, low-to-moderate loudness, mellow full resonance, and a subtle smile. Sound caring, not brightly cheerful, formal, or sales-like."
        }
        ManagedIntensity::Strong => {
            "Affectionate, tender and welcoming while remaining calm and natural. Sound genuinely caring and close: slightly slower than ordinary conversation, with smooth phrasing, soft attacks, gently sustained stressed vowels, low-to-moderate loudness, a natural pitch centre, rounded expressive intonation, mellow full resonance, and a gentle smile. Avoid excitement, sing-song delivery, sentimentality, theatricality, or shouting."
        }
    };
    format!("{base}{}", short_friendly_guard(text))
}

fn cheerful_profile(intensity: ManagedIntensity, text: &str) -> String {
    let base = match intensity {
        ManagedIntensity::Subtle => {
            "Mildly cheerful and optimistic, clearly positive but not excited. Keep loudness and the speaker's natural pitch centre near neutral; use a light audible smile, slightly brighter resonance, a small but definite increase in pitch variation, tiny buoyant rises on selected words, and a gently quicker lighter rhythm. Keep it understated and friendly, never flat, shouted, announcer-like, or theatrical."
        }
        ManagedIntensity::Normal => {
            "Cheerful and upbeat with moderate positive energy. Use an audible smile, brighter resonance, clearly varied but smooth pitch, a light rhythmic lift, and clear relaxed articulation. Let positive emphasis appear on selected words instead of raising the entire sentence. Keep loudness conversational; avoid shouting or an announcer-like delivery."
        }
        ManagedIntensity::Strong => {
            "Clearly delighted and lively. Use a bright audible smile, wider but smooth pitch movement, energetic rhythm, and crisp relaxed articulation. Let enthusiasm arrive in brief phrase-level peaks instead of making every syllable high or loud. Keep loudness peaks controlled and never shout or become shrill or theatrical."
        }
    };
    format!("{base}{}", short_friendly_guard(text))
}

fn excited_profile(intensity: ManagedIntensity, text: &str) -> String {
    // High-arousal positive speech is identified most reliably by a raised and
    // wider F0 contour plus faster temporal transitions. Loudness is useful,
    // but sustained loudness is neither necessary nor desirable for natural
    // excitement, so emphasis is phrased as short local peaks with release.
    let base = match intensity {
        ManagedIntensity::Subtle => {
            "Mildly excited and interested, with positive anticipation rather than surprise. Keep the pitch centre only slightly above neutral but make pitch movement perceptibly more variable, use a modestly wider range, a lightly quicker rhythm, crisp relaxed articulation, and small buoyant accents on selected words. Let intensity and pitch rise briefly and settle again within each phrase; keep sustained loudness conversational and never sound startled, shout, squeal, or become announcer-like."
        }
        ManagedIntensity::Normal => {
            "Excited and energized with natural high-arousal positive prosody. Use a moderately raised pitch centre, clearly wider and more variable pitch movement, quicker transitions between stressed syllables, slightly shorter pauses, and crisp articulate consonants. Let both pitch and intensity form brief local peaks on important words, then release toward conversational level so the delivery stays dynamic rather than continuously heightened. Sound genuinely engaged and animated, not startled, continuously loud, frantic, shrill, or theatrical."
        }
        ManagedIntensity::Strong => {
            "Strong genuine excitement with vivid but controlled high-arousal positive prosody. Use a clearly raised pitch centre, wide smooth and highly variable pitch excursions, fast responsive timing, short pauses, and energetic crisp articulation. Build distinct local pitch-and-energy peaks on important words and immediately release between them so the sentence breathes. Keep the voice intelligible and speaker-like; do not turn excitement into surprise, sustain maximum loudness, scream, squeal, or become frantic."
        }
    };
    format!("{base}{}", short_friendly_guard(text))
}

fn concerned_profile(intensity: ManagedIntensity, text: &str) -> String {
    // Concern is not simply fear. The useful target for narration is an
    // attentive alert phase followed, where semantically appropriate, by the
    // lower/slower/softer prosody listeners associate with caring reassurance.
    // Anxiety findings are inconsistent across studies, so avoid rigid global
    // pitch/rate shifts and instead ask for a controlled phrase-level contour.
    let base = match intensity {
        ManagedIntensity::Subtle => {
            "Mildly concerned and attentive, as if noticing something that deserves care. On concern-bearing words use a small increase in pitch responsiveness, slight vocal tension, focused articulation, and restrained intensity. Where the wording becomes reassuring, release that tension: return the pitch centre toward neutral, slow slightly, soften attacks, and use smooth gentle falling contours. Sound caring and alert, never fearful, breathless, gloomy, or dramatic."
        }
        ManagedIntensity::Normal => {
            "Clearly concerned, attentive, and caring rather than frightened. Begin concern-bearing phrases with mild vocal tension, a slightly raised and more responsive pitch contour, focused articulation, and modest urgency. As reassurance becomes appropriate, audibly relax: lower the pitch centre toward neutral, slow the rate a little, soften consonant attacks, reduce energy, and use smoother falling phrase endings. Preserve calm control throughout; never panic, plead, rush, or sound ominous."
        }
        ManagedIntensity::Strong => {
            "Strong but controlled concern, emotionally engaged without panic. Let urgent words carry firmer articulation, somewhat higher local pitch peaks, tighter timing, and brief increases in energy, while avoiding a globally high or loud voice. When the sentence turns toward reassurance, clearly release the tension with a lower pitch centre, slower pace, softer attacks, quieter energy, and longer smooth falling contours. Sound protective and deeply attentive, not frightened, hysterical, or theatrical."
        }
    };
    format!("{base}{}", short_concerned_guard(text))
}

fn short_gentle_guard(text: &str) -> &'static str {
    let words = text.split_whitespace().count();
    if words <= 14 && text.contains('!') {
        " On this short line, keep the exclamation unprojected: express emphasis with clean timing and articulation, not extra loudness or a sudden pitch jump."
    } else if words <= 14 {
        " On this short line, keep vocal effort low and even; do not drift into whispering, sleepiness, sentimentality, or exaggerated tenderness."
    } else {
        ""
    }
}

fn gentle_profile(intensity: ManagedIntensity, text: &str) -> String {
    // Gentle is a low-vocal-effort speaking style, not an affiliative emotion.
    // Research on vocal effort links lower perceived effort with lower SPL and
    // reduced aerodynamic/laryngeal drive. Preserve the text's own affect and
    // distinguish gentle delivery from Warm, breathy speech, and whispering.
    let base = match intensity {
        ManagedIntensity::Subtle => {
            "Slightly gentle and low-effort while remaining ordinary conversational speech. Reduce projection and loudness only a little, use smooth connected phrasing, light clean consonant attacks, relaxed articulation, and slightly restrained pitch excursions. Preserve the emotional meaning of the text instead of adding warmth or tenderness. Stay clear and awake; do not become breathy, whisper-like, sleepy, sentimental, or flat."
        }
        ManagedIntensity::Normal => {
            "Gentle, low-effort and calm without imposing a separate emotion. Use modestly reduced projection and loudness, smooth connected phrasing, light clean attacks, relaxed precise articulation, an unhurried but natural pace, and restrained yet living pitch movement. Let emphasis remain selective and soft-edged. Preserve whatever emotional valence the wording already carries; do not automatically sound warm, happy, sad, sleepy, breathy, or whisper-like."
        }
        ManagedIntensity::Strong => {
            "Very gentle, low-effort and unprojected while staying fully intelligible. Keep vocal effort and loudness clearly below ordinary conversation, with very smooth phrasing, soft clean attacks, relaxed articulation, slightly longer transitions, and compact natural pitch movement. Maintain clarity and alertness and preserve the text's own emotion. Do not turn the voice into a whisper, breathy murmur, lullaby, sentimental tenderness, or monotone."
        }
    };
    format!("{base}{}", short_gentle_guard(text))
}

fn short_sad_guard(text: &str) -> &'static str {
    let words = text.split_whitespace().count();
    if words <= 14 && text.contains('!') {
        " On this short line, keep the exclamation emotionally subdued: do not create a loud burst, high-pitch spike, sob, or dramatic climax."
    } else if words <= 14 && text.contains('?') {
        " On this short line, preserve the linguistic question contour while keeping pitch range narrow, intensity soft, and the overall delivery subdued."
    } else if words <= 14 {
        " On this short line, keep sadness perceptible through slightly lower/narrower pitch, softer energy, and slower timing without collapsing into sleepy or flat neutral speech."
    } else {
        ""
    }
}

fn sad_profile(intensity: ManagedIntensity, text: &str) -> String {
    // Sadness is a comparatively robust low-arousal profile in emotional
    // prosody research: lower mean F0, narrower F0 range, lower intensity and
    // slower rate than neutral. Keep those cues relative to the cloned speaker
    // and explicitly separate sadness from boredom, sleepiness and grief/wailing.
    let base = match intensity {
        ManagedIntensity::Subtle => {
            "Slightly sad and reflective, with a clear but understated low-arousal shift from neutral. Lower the speaker's pitch centre only a little, modestly narrow pitch excursions, slow the pace slightly, and reduce intensity a little. Use soft clean attacks, slightly longer transitions between thoughts, and gentle downward phrase endings while keeping articulation clear and the intonation alive. Sound quietly affected, not tired, bored, depressed, whispery, or flat."
        }
        ManagedIntensity::Normal => {
            "Clearly sad and reflective without becoming theatrical. Use a moderately lower pitch centre than the speaker's neutral voice, a noticeably narrower pitch range, slower phrasing, softer intensity, light clean attacks, and somewhat longer pauses at meaningful boundaries. Let important phrases settle into gentle falling endings and reduced energy rather than dramatic pitch movement. Keep the voice intelligible and emotionally present; do not sound sleepy, bored, clinically depressed, breathy, monotone, or like you are crying."
        }
        ManagedIntensity::Strong => {
            "Deeply sad and emotionally affected, but controlled and low-arousal rather than grief-stricken. Use a clearly lower pitch centre, compact pitch range, distinctly slower timing, low steady intensity, softened attacks, longer meaningful pauses, and sustained gentle falling contours. Allow weight on important words through duration and reduced energy, not loudness or large pitch excursions. Preserve clarity and speaker identity; never wail, sob, break the voice, whisper, become ominous, or collapse into a lifeless monotone."
        }
    };
    format!("{base}{}", short_sad_guard(text))
}

fn short_serious_guard(text: &str) -> &'static str {
    let words = text.split_whitespace().count();
    if words <= 14 && text.contains('!') {
        " On this short line, make the exclamation firm and consequential through timing and articulation, not extra loudness, anger, or a dramatic pitch jump."
    } else if words <= 14 && text.contains('?') {
        " On this short line, preserve the linguistic question contour while keeping the delivery composed, focused, and free of uncertainty or melodrama."
    } else if words <= 14 {
        " On this short line, keep seriousness audible through deliberate timing, stable energy, and selective emphasis without becoming monotone or artificially low-pitched."
    } else {
        ""
    }
}

fn serious_profile(intensity: ManagedIntensity, text: &str) -> String {
    // Seriousness is a communicative stance rather than a basic emotion, so do
    // not invent an unsupported fixed acoustic signature. Target commitment and
    // attentional control using stable energy, deliberate temporal grouping,
    // clear articulation and decisive terminal contours. Confidence research
    // supports falling intonation and firm/stable delivery, but not a blanket
    // rule that a serious speaker must use an artificially low pitch.
    let base = match intensity {
        ManagedIntensity::Subtle => {
            "Slightly serious and focused while remaining conversational. Keep the speaker's natural pitch centre, modestly restrain unnecessary pitch excursions, use a measured but normal pace, clean articulation, stable conversational intensity, and selective emphasis on semantically important words. Favor calm, resolved phrase endings where linguistically appropriate. Sound attentive and committed, not gloomy, stern, uncertain, monotone, or artificially deep."
        }
        ManagedIntensity::Normal => {
            "Serious, composed and focused, conveying commitment rather than a separate negative emotion. Keep the speaker's natural pitch centre or only very slightly lower, use moderately controlled pitch excursions, measured pacing, deliberate short pauses at meaningful boundaries, firm clean articulation, and stable conversational intensity. Emphasize only key words and use decisive falling phrase endings where the sentence permits. Sound consequential and attentive, not angry, sad, ominous, pompous, monotone, or artificially low-pitched."
        }
        ManagedIntensity::Strong => {
            "Strongly serious, deliberate and consequential while remaining natural. Use controlled but still expressive pitch movement around the speaker's natural range, steady moderate intensity, precise firm articulation, purposeful pacing, and clearly deliberate pauses that organize important ideas. Give selected words confident prominence and use resolved falling endings where linguistically appropriate. Do not imitate a movie-trailer voice, force a deep register, become grave or ominous, sound angry, or flatten the voice into monotone."
        }
    };
    format!("{base}{}", short_serious_guard(text))
}

fn short_whisper_guard(text: &str) -> &'static str {
    let words = text.split_whitespace().count();
    if words <= 14 && text.contains('!') {
        " On this short line, express the exclamation through timing and articulation only; do not project, raise loudness, or turn the near-whisper into a normal voiced exclamation."
    } else if words <= 14 {
        " On this short line, keep the low-effort near-whisper continuous from beginning to end instead of drifting back into ordinary soft speech."
    } else {
        ""
    }
}

fn short_angry_guard(text: &str) -> &'static str {
    let words = text.split_whitespace().count();
    if words <= 14 && (text.contains('!') || text.contains('?')) {
        " On this short line, let punctuation sharpen timing and attack, not loudness; keep the anger controlled and never shout."
    } else if words <= 14 {
        " On this short line, make tension audible through firm attacks and compact timing without building a loud dramatic climax."
    } else {
        ""
    }
}

fn whisper_profile(intensity: ManagedIntensity, text: &str) -> String {
    // A true whisper removes ordinary periodic vocal-fold vibration and F0.
    // For cloned TTS, target a controlled near-whisper: clearly lower vocal
    // effort and more airflow/noise than modal speech while retaining enough
    // residual voicing and articulation for intelligibility and speaker identity.
    let base = match intensity {
        ManagedIntensity::Subtle => {
            "Soft intimate near-whisper, clearly lower in vocal effort than ordinary soft speech. Add a modest audible airflow component, soften glottal and consonant attacks, reduce projection and periodic voicing, and slightly lengthen careful articulation while preserving intelligibility and speaker identity. Keep some natural residual voicing; do not merely lower the volume of a normal modal voice, and do not sound hoarse, sleepy, or theatrical."
        }
        ManagedIntensity::Normal => {
            "Intimate controlled near-whisper with very low vocal effort and minimal projection. Use clearly audible airflow/noise, strongly softened attacks, reduced periodic voicing, slightly lengthened vowels and deliberate articulation, while keeping words crisp enough to understand and the cloned speaker recognizable. The result should sound phonationally whisper-like, not simply like normal speech played quietly; avoid hoarseness, hissy exaggeration, or dramatic secrecy."
        }
        ManagedIntensity::Strong => {
            "As close to a true whisper as the cloned voice can naturally sustain while remaining intelligible. Minimize periodic voicing and projection, make airflow/noise clearly dominant, use very soft attacks and carefully lengthened articulation, and keep intensity low and even. Preserve lexical clarity and as much speaker identity as possible. Do not compensate by speaking louder, do not become harsh or hissy, and do not revert to a breathy normal voice."
        }
    };
    format!("{base}{}", short_whisper_guard(text))
}

fn angry_profile(intensity: ManagedIntensity, text: &str) -> String {
    // The application deliberately targets controlled/cold anger rather than
    // explosive hot anger. Research distinguishes cold anger from hot anger by
    // more moderate/lower global F0 and range. Preserve anger through vocal
    // tension, firm attacks, compact timing and selective emphasis, not volume.
    let base = match intensity {
        ManagedIntensity::Subtle => {
            "Restrained irritation with controlled cold-anger cues. Keep the pitch centre near neutral or slightly lower and the pitch range fairly compact; use firmer vocal tension, clipped phrase timing, shorter pauses, crisp hard attacks on selected stressed words, and a slightly brighter/tighter resonance. Keep loudness moderate and mostly steady. Sound unmistakably irritated but contained, never loud, growled, breathless, or theatrical."
        }
        ManagedIntensity::Normal => {
            "Clearly angry but controlled, closer to cold anger than explosive hot anger. Keep the global pitch centre near neutral or slightly lower with compact purposeful pitch movement; use firm vocal tension, hard clean attacks, quicker transitions, shorter pauses, and brief sharp emphasis on key words. Maintain moderate sustained loudness and release pressure between emphatic phrases. Carry anger through tension, timing, articulation and a tight bright voice quality rather than shouting or continuous volume."
        }
        ManagedIntensity::Strong => {
            "Strong controlled anger with high tension but disciplined delivery. Use very firm clean attacks, fast compact timing, short pauses, a tight bright voice quality, and forceful local accents on important words; allow only brief local pitch and energy peaks, then immediately release. Keep overall loudness below a shout and the global pitch range controlled rather than wildly high. Sound forceful and unmistakably angry without screaming, growling, rasping, or becoming continuously loud."
        }
    };
    format!("{base}{}", short_angry_guard(text))
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
            return Err(format!(
                "speed transition is too subtle ({delta:.1} percentage points); omit it or use at least {MIN_MEANINGFUL_MOOD_SPEED_DELTA_PERCENT:.0} percentage points"
            ));
        }
        if delta > MAX_NATURAL_MOOD_SPEED_DELTA_PERCENT {
            return Err(format!(
                "speed transition is too large for one natural phrase ({delta:.1} percentage points); keep it within {MAX_NATURAL_MOOD_SPEED_DELTA_PERCENT:.0} percentage points or split the change across phrases"
            ));
        }
        Ok(Self { from_percent, to_percent })
    }

    pub(crate) fn changed(self) -> bool {
        (self.to_percent - self.from_percent).abs() >= MIN_MEANINGFUL_MOOD_SPEED_DELTA_PERCENT
    }
}

pub(crate) fn parse_mood_speed_transition(
    from_percent: Option<f32>,
    to_percent: Option<f32>,
) -> Result<Option<MoodSpeedTransition>, String> {
    match (from_percent, to_percent) {
        (None, None) => Ok(None),
        (Some(from), Some(to)) => {
            let speed = MoodSpeedTransition::new(from, to)?;
            if speed.changed() { Ok(Some(speed)) } else { Ok(None) }
        }
        _ => Err("from_speed_percent and to_speed_percent must be supplied together".to_string()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MoodTransitionMode {
    Gradual,
    Quick,
}

impl MoodTransitionMode {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "gradual" | "smooth" => Ok(Self::Gradual),
            "quick" | "fast" => Ok(Self::Quick),
            other => Err(format!("unsupported transition mode '{other}'; use gradual or quick")),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Gradual => "gradual",
            Self::Quick => "quick",
        }
    }
}

fn normalize_transition_intensity(value: &str) -> Result<&'static str, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "subtle" => Ok("subtle"),
        "normal" | "medium" => Ok("normal"),
        "strong" => Ok("strong"),
        other => Err(format!("unsupported transition intensity '{other}'; use subtle, normal, or strong")),
    }
}

fn normalize_transition_style(value: &str) -> Result<&'static str, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "neutral" => Ok("neutral"),
        "warm" => Ok("warm"),
        "cheerful" => Ok("cheerful"),
        "excited" => Ok("excited"),
        "sad" => Ok("sad"),
        "concerned" => Ok("concerned"),
        "angry" => Ok("angry"),
        "gentle" => Ok("gentle"),
        "serious" => Ok("serious"),
        "whisper" | "whisper-like" | "whisper_like" => Ok("whisper"),
        other => Err(format!(
            "unsupported transition style '{other}'; use neutral, warm, cheerful, excited, sad, concerned, angry, gentle, serious, or whisper"
        )),
    }
}

fn transition_intensity_word(intensity: &str) -> &'static str {
    match intensity {
        "subtle" => "lightly",
        "strong" => "strongly",
        _ => "clearly",
    }
}

fn transition_signature(style: &str, intensity: &str) -> String {
    let degree = transition_intensity_word(intensity);
    match style {
        "neutral" => format!("{degree} affect-neutral, with natural conversational pitch, timing, lexical stress, and ordinary sentence intonation"),
        "warm" => format!("{degree} warm and affiliative, with low vocal effort, smooth connected phrasing, soft attacks, mellow resonance, and a faint friendly smile"),
        "cheerful" => format!("{degree} cheerful, with a light audible smile, brighter resonance, buoyant rhythm, and wider but smooth positive pitch movement"),
        "excited" => format!("{degree} excited and positively engaged, with quicker timing, wider dynamic pitch movement, crisp articulation, and brief local energy peaks that release"),
        "sad" => format!("{degree} sad and reflective, with a lower pitch centre, narrower pitch range, softer energy, slower phrasing, and gently falling endings"),
        "concerned" => format!("{degree} concerned and attentive, with mild local tension and responsive pitch that can soften into slower, lower, reassuring contours"),
        "angry" => format!("{degree} controlled cold anger, with firm vocal tension, hard clean attacks, compact purposeful pitch movement, quicker timing, and moderate sustained loudness"),
        "gentle" => format!("{degree} gentle and low-effort, with reduced projection, smooth phrasing, light attacks, relaxed articulation, and restrained but living pitch movement"),
        "serious" => format!("{degree} serious and deliberate, with controlled pitch movement, measured pacing, stable intensity, firm articulation, selective emphasis, and resolved endings"),
        "whisper" => format!("{degree} near-whispered, with very low vocal effort, audible airflow, minimal projection, softened attacks, and reduced periodic voicing while remaining intelligible"),
        _ => unreachable!(),
    }
}

fn transition_pair_motion(from: &str, to: &str) -> Option<&'static str> {
    match (from, to) {
        ("angry", "serious") => Some(
            "Retain firmness at first, then progressively release vocal tension, soften the hard attacks, lengthen pauses slightly, stabilize pitch and intensity, and finish composed and consequential rather than angry."
        ),
        ("concerned", "warm") => Some(
            "Begin attentive and mildly tense; then let the pitch settle toward the speaker's natural centre, slow slightly, soften attacks, smooth the phrasing, and let caring warmth emerge as the concern resolves."
        ),
        ("sad", "warm") => Some(
            "Keep the subdued pace and narrower pitch early, then gradually restore pitch variability and vocal energy, soften the heaviness of falling contours, and let gentle affiliative warmth appear without suddenly becoming cheerful."
        ),
        ("excited", "neutral") => Some(
            "Start animated, then steadily narrow the pitch excursions, release the quicker rhythm and local energy peaks, lengthen pauses toward normal, and settle into ordinary conversational prosody without sounding deflated."
        ),
        ("warm", "serious") => Some(
            "Begin approachable and smooth, then reduce the smile and affiliative softness, make articulation firmer, stabilize intensity, restrain pitch excursions, and finish focused and consequential without becoming cold or angry."
        ),
        ("serious", "warm") => Some(
            "Begin focused and composed, then gradually release firmness and pitch restraint, soften attacks, smooth the phrasing, and introduce mellow affiliative warmth while preserving clarity and speaker identity."
        ),
        ("neutral", "concerned") => Some(
            "Begin in ordinary conversational prosody, then introduce mild local tension, more attentive pitch responses, and focused articulation without turning the phrase into fear or panic."
        ),
        ("concerned", "neutral") => Some(
            "Begin attentive and mildly tense, then release the concern cues: settle pitch, normalize timing, soften local tension, and finish in ordinary conversational prosody rather than reassurance acting."
        ),
        ("neutral", "sad") => Some(
            "Begin naturally neutral, then lower and narrow the pitch gradually, soften the energy, slow the phrasing, and allow more reflective falling endings without becoming sleepy or monotone."
        ),
        ("sad", "neutral") => Some(
            "Begin subdued, then gradually restore the speaker's ordinary pitch range, conversational timing, energy, and lexical prominence without jumping into cheerfulness."
        ),
        ("neutral", "warm") => Some(
            "Begin naturally neutral, then soften attacks and timing, reduce vocal effort slightly, smooth the phrasing, and let a faint affiliative smile and mellow warmth emerge progressively."
        ),
        ("warm", "neutral") => Some(
            "Begin warm and affiliative, then gradually remove the smile and interpersonal softness while preserving natural conversational movement, ending emotionally unmarked rather than cold."
        ),
        ("neutral", "serious") => Some(
            "Begin in ordinary conversational prosody, then progressively tighten focus: make pauses more deliberate, articulation firmer, pitch excursions more controlled, and phrase endings more resolved while keeping the speaker's natural pitch centre and normal loudness."
        ),
        ("serious", "neutral") => Some(
            "Begin deliberate and focused, then release the controlled stance: relax articulation and pause placement, restore ordinary pitch movement and conversational timing, and finish naturally unmarked without sounding casual or dismissive."
        ),
        ("cheerful", "excited") => Some(
            "Begin lightly positive and buoyant, then raise arousal rather than simply volume: widen and quicken pitch movement, shorten transitions slightly, sharpen articulation, and introduce brief local energy peaks that still release between accents."
        ),
        ("excited", "cheerful") => Some(
            "Begin highly animated, then lower arousal while keeping positive valence: narrow the largest pitch excursions, relax the quick timing and energy peaks, and settle into a lighter smile and buoyant conversational rhythm rather than dropping to neutral."
        ),
        ("concerned", "serious") => Some(
            "Begin attentive with mild vocal tension, then convert that urgency into composed focus: settle responsive pitch, remove worry-like tension, make pauses deliberate and articulation firm, and finish consequential rather than anxious."
        ),
        ("serious", "concerned") => Some(
            "Begin composed and deliberate, then introduce attentive concern without panic: add mild local tension and responsive pitch around important words while preserving controlled loudness and clear reasoning."
        ),
        ("warm", "gentle") => Some(
            "Begin affiliative and warmly connected, then reduce the interpersonal smile while lowering vocal effort and projection, keeping smooth phrasing and soft attacks so the ending is gentle rather than merely less warm."
        ),
        ("gentle", "warm") => Some(
            "Begin low-effort and unprojected, then keep the softness while introducing affiliative cues: a faint smile, mellow resonance, slightly more connected phrasing, and a sense of personal warmth without increasing loudness."
        ),
        ("warm", "sad") => Some(
            "Begin warm and connected, then gradually let the smile and affiliative lift recede as pitch lowers and narrows, energy softens, and phrasing slows into reflective sadness without becoming abruptly bleak."
        ),
        ("sad", "serious") => Some(
            "Begin subdued and reflective, then restore enough energy and pitch control for composure: make articulation firmer, pauses more deliberate, and endings more resolved while leaving sadness behind rather than turning stern."
        ),
        ("serious", "sad") => Some(
            "Begin focused and composed, then gradually release firmness, lower and narrow the pitch, soften energy, lengthen reflective pauses, and let the delivery become sad without becoming sleepy or theatrically mournful."
        ),
        ("concerned", "sad") => Some(
            "Begin actively attentive with mild tension, then let urgency drain away: lower and narrow pitch, slow the phrasing, soften attacks, and settle into reflective sadness rather than fear or resignation."
        ),
        ("sad", "concerned") => Some(
            "Begin subdued, then restore attentiveness and a little local tension: make pitch more responsive, timing slightly more active, and articulation more focused while avoiding panic or a sudden energy jump."
        ),
        ("excited", "serious") => Some(
            "Begin animated, then channel the energy into focus: progressively narrow pitch excursions, release quick rhythmic peaks, stabilize intensity, lengthen pauses slightly, and finish deliberate and consequential rather than deflated."
        ),
        ("angry", "neutral") => Some(
            "Begin with controlled anger, then progressively release vocal tension and hard attacks, lengthen pauses toward normal, normalize pitch movement and articulation, and finish naturally unmarked without sounding defeated."
        ),
        ("angry", "concerned") => Some(
            "Begin firm and angry, then reduce hostility while retaining urgency: soften hard attacks, release tight vocal tension, make pitch more attentive and responsive, and finish concerned rather than submissive or fearful."
        ),
        ("neutral", "whisper") => Some(
            "Begin with ordinary voiced conversation, then progressively reduce projection and vocal effort, soften attacks, increase audible airflow, and reduce periodic voicing over a natural span until the ending is an intelligible near-whisper."
        ),
        ("whisper", "neutral") => Some(
            "Begin near-whispered, then progressively restore periodic voicing and ordinary projection, reduce excess airflow, strengthen clean attacks, and return to natural conversational energy without a sudden voiced onset."
        ),
        ("serious", "whisper") => Some(
            "Begin focused and fully voiced, then preserve the serious intent while gradually reducing projection and vocal effort, softening attacks and adding airflow until the delivery becomes an intimate near-whisper rather than losing focus."
        ),
        ("whisper", "serious") => Some(
            "Begin near-whispered, then gradually restore periodic voicing, projection, firm articulation and deliberate phrase endings so the speaker becomes fully voiced and serious without an abrupt phonation jump."
        ),
        _ => None,
    }
}

fn transition_speed_clause(mode: MoodTransitionMode, speed: Option<MoodSpeedTransition>) -> String {
    let Some(speed) = speed else { return String::new(); };
    if (speed.to_percent - speed.from_percent).abs() < 0.001 {
        return format!(
            " Pace target: remain around {:.0}% of ordinary conversational pace throughout; preserve that tempo while the other prosodic cues evolve. Treat this percentage as a speaking-rate target, not as a pitch shift.",
            speed.to_percent
        );
    }
    let direction = if speed.to_percent > speed.from_percent { "accelerate" } else { "decelerate" };
    let shape = match mode {
        MoodTransitionMode::Gradual =>
            "Spread the pace change across most of the phrase so several prosodic units participate and the listener can hear a continuous intermediate tempo.",
        MoodTransitionMode::Quick =>
            "Make the pace change early and clearly, but take at least two stressed words or one short prosodic unit to complete it; never jump tempo on a single syllable or word.",
    };
    format!(
        " Pace transition: begin around {:.0}% of ordinary conversational pace and {direction} toward {:.0}% by the destination state. {shape} Treat these percentages as speaking-rate targets, not as a pitch shift.",
        speed.from_percent, speed.to_percent
    )
}

fn transition_short_guard(text: &str) -> &'static str {
    let words = text.split_whitespace().count();
    if words <= 10 {
        " Because this is a short line, make the evolution perceptible but compact; do not perform two separate voices or insert an artificial pause at the midpoint."
    } else {
        ""
    }
}

/// Build one evolving VoxCPM2 control instruction for an intra-utterance mood
/// transition. This is prosody conditioning, not waveform crossfading: the
/// entire phrase is synthesized once with one speaker/reference identity.
pub(crate) fn build_transition_control(
    from_style: &str,
    from_intensity: &str,
    to_style: &str,
    to_intensity: &str,
    mode: MoodTransitionMode,
    text: &str,
) -> Result<String, String> {
    build_transition_control_with_speed(
        from_style, from_intensity, to_style, to_intensity, mode, None, text,
    )
}

/// Extended managed transition compiler with an optional speaking-rate envelope.
/// The speed envelope is realized by VoxCPM2 prosody conditioning, not by
/// repeatedly reconfiguring WSOLA mid-phrase; the latter would create avoidable
/// discontinuities and metallic/phasey artifacts.
pub(crate) fn build_transition_control_with_speed(
    from_style: &str,
    from_intensity: &str,
    to_style: &str,
    to_intensity: &str,
    mode: MoodTransitionMode,
    speed: Option<MoodSpeedTransition>,
    text: &str,
) -> Result<String, String> {
    let from = normalize_transition_style(from_style)?;
    let to = normalize_transition_style(to_style)?;
    let from_intensity = normalize_transition_intensity(from_intensity)?;
    let to_intensity = normalize_transition_intensity(to_intensity)?;

    if from == to && from_intensity == to_intensity && speed.is_none() {
        return Err("transition endpoints are identical; use the ordinary managed style instead".to_string());
    }

    let start = transition_signature(from, from_intensity);
    let end = transition_signature(to, to_intensity);
    let timing = match mode {
        MoodTransitionMode::Gradual => {
            "Let the change unfold continuously across the phrase, with the midpoint sounding like a believable intermediate state rather than a preset switch."
        }
        MoodTransitionMode::Quick => {
            "Make the change early and clearly over a short natural span, then spend the remainder of the phrase established in the destination state; do not make an instantaneous acoustic cut."
        }
    };
    let same_style_motion = "Keep the same emotional/style identity throughout and change only its strength: scale the defining cues continuously from the source intensity toward the destination intensity without resetting the voice or phrase.";
    let speed_only_motion = "Keep the same emotional/style identity and intensity throughout; change only the speaking pace continuously, without resetting the voice, articulation, or phrase.";
    let pair = if from == to {
        if from_intensity == to_intensity { speed_only_motion } else { same_style_motion }
    } else {
        transition_pair_motion(from, to).unwrap_or(
            "Continuously release the source style's distinctive cues while introducing the destination style's cues; preserve one speaker identity, one acoustic space, and continuous phrase timing throughout."
        )
    };
    let speed_clause = transition_speed_clause(mode, speed);

    Ok(format!(
        "Begin {start}. End {end}. {timing} {pair}{speed_clause} Do not crossfade, double consonants, change speaker identity, or sound like two separately recorded takes.{}",
        transition_short_guard(text)
    ))
}

/// Conservative automatic CFG delta for a managed transition when the client
/// leaves CFG unspecified. It averages the endpoint guidance so an expressive
/// source such as Angry cannot become louder/more theatrical merely because a
/// transition was requested.
pub(crate) fn managed_transition_cfg_delta(
    from_style: &str,
    from_intensity: &str,
    to_style: &str,
    to_intensity: &str,
) -> Result<f32, String> {
    fn endpoint(style: &str, intensity: &str) -> Result<f32, String> {
        let style = normalize_transition_style(style)?;
        let intensity = normalize_transition_intensity(intensity)?;
        let raw = build_style_control(style, intensity, "")
            .ok_or_else(|| format!("style '{style}' has no managed control recipe"))?;
        Ok(managed_style_tuning(Some(&raw)).cfg_delta)
    }
    let a = endpoint(from_style, from_intensity)?;
    let b = endpoint(to_style, to_intensity)?;
    Ok(((a + b) * 0.5).clamp(0.0, 0.20))
}

/// Transition synthesis should normally use the neutral speaker reference when
/// a client owns a bank of per-style references. The API cannot select that
/// file by itself, so this is exposed as engine policy for clients/demos.
pub(crate) fn recommended_transition_reference_style() -> &'static str {
    "neutral"
}

/// Compile a VoxGen-managed free-form instruction into a more acoustically
/// specific prompt. Arbitrary custom controls are returned verbatim.
pub fn refine_control_instruction(control: &str, text: &str) -> String {
    let trimmed = control.trim();
    let lower = trimmed.to_ascii_lowercase();

    // Only rewrite exact VoxGen-managed recipe families emitted by
    // VoxGen/Dynamic Dictionary. A person who supplied a custom control through
    // the CLI or API retains exact control even if it happens to mention warmth.
    if !lower.contains(LEGACY_MANAGED_SUFFIX) {
        return trimmed.to_owned();
    }

    let managed_neutral = lower.starts_with("subtly neutral and conversational")
        || lower.starts_with("neutral and conversational")
        || lower.starts_with("deliberately affect-neutral")
        || lower.starts_with("natural and conversational, emotionally restrained")
        || lower.starts_with("natural, conversational and emotionally balanced")
        || lower.starts_with("clear, composed and deliberately neutral");
    let managed_cheerful = lower.starts_with("lightly cheerful and optimistic")
        || lower.starts_with("cheerful and warm")
        || lower.starts_with("very cheerful and lively");
    let managed_warm = lower.starts_with("slightly warm and friendly")
        || lower.starts_with("warm and friendly")
        || lower.starts_with("very warm, affectionate and welcoming");
    let managed_excited = lower.starts_with("mildly excited and interested")
        || lower.starts_with("pleasantly surprised, gradually becoming more animated")
        || lower.starts_with("excited and energetic")
        || lower.starts_with("genuinely excited and energetic");
    let managed_concerned = lower.starts_with("slightly concerned and attentive")
        || lower.starts_with("quietly concerned at first")
        || lower.starts_with("clearly worried at first");
    let managed_angry = lower.starts_with("restrained irritation")
        || lower.starts_with("clearly angry but controlled")
        || lower.starts_with("strong controlled anger");
    let managed_gentle = lower.starts_with("slightly softer and gentler than normal")
        || lower.starts_with("gentle, low-effort and calm")
        || lower.starts_with("gentle, calm and reassuring")
        || lower.starts_with("very gentle, low-effort and unprojected")
        || lower.starts_with("very gentle, tender and soothing");
    let managed_sad = lower.starts_with("slightly subdued and reflective")
        || lower.starts_with("subdued and reflective")
        || lower.starts_with("deeply saddened and reflective");
    let managed_serious = lower.starts_with("slightly serious and focused")
        || lower.starts_with("serious, composed and focused")
        || lower.starts_with("strongly serious, deliberate and focused")
        || lower.starts_with("grave, authoritative and focused");
    let managed_whisper = lower.starts_with("soft, intimate and breathy")
        || lower.starts_with("whisper-like, soft and intimate")
        || lower.starts_with("very soft and whisper-like");

    let intensity = managed_intensity(&lower);
    if managed_neutral {
        return neutral_profile(intensity, text);
    }
    if managed_cheerful {
        return cheerful_profile(intensity, text);
    }
    if managed_warm {
        return warm_profile(intensity, text);
    }
    if managed_excited {
        return excited_profile(intensity, text);
    }
    if managed_concerned {
        return concerned_profile(intensity, text);
    }
    if managed_angry {
        return angry_profile(intensity, text);
    }
    if managed_gentle {
        return gentle_profile(intensity, text);
    }
    if managed_sad {
        return sad_profile(intensity, text);
    }
    if managed_serious {
        return serious_profile(intensity, text);
    }
    if managed_whisper {
        return whisper_profile(intensity, text);
    }

    trimmed.to_owned()
}

/// Shared style recipe builder used by the VoxGen demo. HTTP/CLI clients may
/// still provide a raw `control`; the runtime compiler above recognizes the
/// managed suffix and applies the same final rendering rules.
pub fn build_style_control(preset: &str, intensity: &str, custom: &str) -> Option<String> {
    if preset == "auto" {
        return None;
    }
    let natural = LEGACY_MANAGED_SUFFIX;
    let prompt = match preset {
        "neutral" => match intensity {
            "subtle" => format!("subtly neutral and conversational, with minimal imposed affect, {natural}"),
            "strong" => format!("deliberately affect-neutral but still natural and conversational, {natural}"),
            _ => format!("neutral and conversational, with no deliberately imposed emotional colour, {natural}"),
        },
        "warm" => match intensity {
            "subtle" => format!("slightly warm and friendly, conversational, with a subtle smile, {natural}"),
            "strong" => format!("very warm, affectionate and welcoming, genuinely pleased, {natural}"),
            _ => format!("warm and friendly, conversational, gently smiling, {natural}"),
        },
        "cheerful" => match intensity {
            "subtle" => format!("lightly cheerful and optimistic, relaxed and conversational, {natural}"),
            "strong" => format!("very cheerful and lively, clearly delighted but still natural, {natural}"),
            _ => format!("cheerful and warm, naturally expressive and upbeat, {natural}"),
        },
        "excited" => match intensity {
            "subtle" => format!("mildly excited and interested, positive anticipation without surprise, {natural}"),
            "strong" => format!("genuinely excited and energetic, enthusiasm rising on important phrases without shouting, {natural}"),
            _ => format!("excited and energetic, with believable changes in enthusiasm across the sentence, {natural}"),
        },
        "sad" => match intensity {
            "subtle" => format!("slightly subdued and reflective, speaking a little more softly, {natural}"),
            "strong" => format!("deeply saddened and reflective, emotionally affected but restrained rather than theatrical, {natural}"),
            _ => format!("subdued and reflective, speaking softly with restrained sadness, {natural}"),
        },
        "concerned" => match intensity {
            "subtle" => format!("slightly concerned and attentive, becoming gently reassuring where the wording allows, {natural}"),
            "strong" => format!("clearly worried at first, emotionally attentive, becoming reassuring when appropriate, {natural}"),
            _ => format!("quietly concerned at first, becoming gently reassuring where the wording allows, {natural}"),
        },
        "angry" => match intensity {
            "subtle" => format!("restrained irritation, firm and clipped but still natural, moderate loudness, with tension carried by timing and emphasis rather than shouting, {natural}"),
            "strong" => format!("strong controlled anger, tense and forceful with short bursts of emphasis on key words, clean and intelligible rather than screamed or continuously loud, {natural}"),
            _ => format!("clearly angry but controlled, tense and direct, with sharper phrase-level emphasis and moderate loudness rather than a constant shouted delivery, {natural}"),
        },
        "gentle" => match intensity {
            "subtle" => format!("slightly softer and gentler than normal, low-effort and conversational, {natural}"),
            "strong" => format!("very gentle, low-effort and unprojected, clear and calm rather than whisper-like, {natural}"),
            _ => format!("gentle, low-effort and calm, softly projected without imposing warmth, {natural}"),
        },
        "serious" => match intensity {
            "subtle" => format!("slightly serious and focused, measured but conversational, {natural}"),
            "strong" => format!("strongly serious, deliberate and focused, consequential without becoming ominous or monotone, {natural}"),
            _ => format!("serious, composed and focused, with measured emphasis, {natural}"),
        },
        "whisper" => match intensity {
            "subtle" => format!("soft, intimate and breathy, close to a whisper while remaining clear, {natural}"),
            "strong" => format!("very soft and whisper-like, intimate and breathy but still intelligible, {natural}"),
            _ => format!("whisper-like, soft and intimate, breathy but intelligible, {natural}"),
        },
        "custom" => {
            let custom = custom.trim();
            if custom.is_empty() {
                return None;
            }
            match intensity {
                "subtle" => format!("{custom}; keep the effect subtle and restrained; {natural}"),
                "strong" => format!("{custom}; make the requested delivery clearly perceptible but still believable; {natural}"),
                _ => format!("{custom}; keep the delivery natural and believable; {natural}"),
            }
        }
        _ => return None,
    };
    Some(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_controls_are_untouched() {
        let custom = format!("A close, friendly documentary voice; {LEGACY_MANAGED_SUFFIX}");
        assert_eq!(refine_control_instruction(&custom, "Hello."), custom);
    }

    #[test]
    fn subtle_cheerful_gets_low_arousal_positive_cues() {
        let raw = build_style_control("cheerful", "subtle", "").unwrap();
        let effective = refine_control_instruction(&raw, "Select any text and ask me about it!");
        assert!(effective.contains("light audible smile"));
        assert!(effective.contains("slightly brighter resonance"));
        assert!(effective.contains("small but definite increase in pitch variation"));
        assert!(effective.contains("friendly brightness"));
        assert!(!effective.contains(LEGACY_MANAGED_SUFFIX));
    }

    #[test]
    fn subtle_warm_gets_stable_social_cues() {
        let raw = build_style_control("warm", "subtle", "").unwrap();
        let effective = refine_control_instruction(&raw, "Welcome back.");
        assert!(effective.contains("slightly slower and softer than neutral"));
        assert!(effective.contains("mellow full resonance"));
        assert!(effective.contains("natural pitch centre"));
        assert!(effective.contains("Clearly caring"));
    }

    #[test]
    fn excited_uses_dynamic_arousal_not_sustained_loudness() {
        let raw = build_style_control("excited", "normal", "").unwrap();
        let effective = refine_control_instruction(&raw, "This is incredible!");
        assert!(effective.contains("moderately raised pitch centre"));
        assert!(effective.contains("clearly wider and more dynamic pitch movement"));
        assert!(effective.contains("brief phrase-level peaks"));
        assert!(effective.contains("not continuously loud"));
    }

    #[test]
    fn strong_excited_is_recognized_as_strong() {
        let raw = build_style_control("excited", "strong", "").unwrap();
        let effective = refine_control_instruction(&raw, "We actually did it!");
        assert!(effective.starts_with("Strong genuine excitement"));
        assert!(effective.contains("wide smooth pitch excursions"));
    }

    #[test]
    fn concerned_has_alert_then_reassuring_contour() {
        let raw = build_style_control("concerned", "normal", "").unwrap();
        let effective = refine_control_instruction(&raw, "I noticed a problem, but we can fix it.");
        assert!(effective.contains("Begin concern-bearing phrases with mild vocal tension"));
        assert!(effective.contains("As reassurance becomes appropriate, audibly relax"));
        assert!(effective.contains("lower the pitch centre toward neutral"));
        assert!(effective.contains("never panic"));
    }

    #[test]
    fn strong_concerned_is_recognized_as_strong() {
        let raw = build_style_control("concerned", "strong", "").unwrap();
        let effective = refine_control_instruction(&raw, "Something is wrong, but stay with me.");
        assert!(effective.starts_with("Strong but controlled concern"));
        assert!(effective.contains("somewhat higher local pitch peaks"));
        assert!(effective.contains("protective and deeply attentive"));
    }


    #[test]
    fn whisper_like_targets_phonation_not_just_quiet_volume() {
        let raw = build_style_control("whisper", "normal", "").unwrap();
        let effective = refine_control_instruction(&raw, "Please keep this between us.");
        assert!(effective.contains("Intimate controlled near-whisper"));
        assert!(effective.contains("audible airflow/noise"));
        assert!(effective.contains("reduced periodic voicing"));
        assert!(effective.contains("not simply like normal speech played quietly"));
    }

    #[test]
    fn strong_whisper_is_recognized_as_strong() {
        let raw = build_style_control("whisper", "strong", "").unwrap();
        let effective = refine_control_instruction(&raw, "Don't wake anyone!");
        assert!(effective.starts_with("As close to a true whisper"));
        assert!(effective.contains("airflow/noise clearly dominant"));
        assert!(effective.contains("do not project"));
    }

    #[test]
    fn controlled_angry_uses_cold_anger_cues_not_sustained_loudness() {
        let raw = build_style_control("angry", "normal", "").unwrap();
        let effective = refine_control_instruction(&raw, "I asked you not to do that!");
        assert!(effective.contains("closer to cold anger than explosive hot anger"));
        assert!(effective.contains("compact purposeful pitch movement"));
        assert!(effective.contains("hard clean attacks"));
        assert!(effective.contains("moderate sustained loudness"));
        assert!(effective.contains("rather than shouting"));
    }

    #[test]
    fn subtle_angry_is_recognized_as_subtle() {
        let raw = build_style_control("angry", "subtle", "").unwrap();
        let effective = refine_control_instruction(&raw, "That's enough.");
        assert!(effective.starts_with("Restrained irritation"));
        assert!(effective.contains("pitch range fairly compact"));
        assert!(effective.contains("mostly steady"));
    }

    #[test]
    fn excited_is_not_conflated_with_surprise() {
        let raw = build_style_control("excited", "subtle", "").unwrap();
        assert!(raw.starts_with("mildly excited and interested"));
        let effective = refine_control_instruction(&raw, "This is going to be great!");
        assert!(effective.contains("positive anticipation rather than surprise"));
        assert!(effective.contains("pitch movement perceptibly more variable"));
        assert!(effective.contains("sustained loudness conversational"));
    }

    #[test]
    fn gentle_is_low_effort_not_warm_or_whisper() {
        let raw = build_style_control("gentle", "normal", "").unwrap();
        let effective = refine_control_instruction(&raw, "Here is what we need to do.");
        assert!(effective.starts_with("Gentle, low-effort and calm"));
        assert!(effective.contains("modestly reduced projection and loudness"));
        assert!(effective.contains("Preserve whatever emotional valence"));
        assert!(effective.contains("do not automatically sound warm"));
        assert!(effective.contains("whisper-like"));
    }

    #[test]
    fn gentle_strength_scales_level_without_hidden_engine_gain() {
        let subtle = build_style_control("gentle", "subtle", "").unwrap();
        let normal = build_style_control("gentle", "normal", "").unwrap();
        let strong = build_style_control("gentle", "strong", "").unwrap();
        assert_eq!(managed_style_tuning(Some(&subtle)).demo_gain_multiplier, 0.98);
        assert_eq!(managed_style_tuning(Some(&normal)).demo_gain_multiplier, 0.95);
        assert_eq!(managed_style_tuning(Some(&strong)).demo_gain_multiplier, 0.92);
    }

    #[test]
    fn neutral_preserves_linguistic_prosody_without_imposing_emotion() {
        let raw = build_style_control("neutral", "normal", "").unwrap();
        let effective = refine_control_instruction(&raw, "Are we still meeting at six?");
        assert!(effective.starts_with("Neutral, natural conversational speech"));
        assert!(effective.contains("habitual pitch centre"));
        assert!(effective.contains("clear lexical stress"));
        assert!(effective.contains("syntax-driven phrase contours"));
        assert!(effective.contains("do not flatten the melody"));
        assert!(!effective.contains(LEGACY_MANAGED_SUFFIX));
    }

    #[test]
    fn strong_neutral_stays_human_not_flat_or_deep() {
        let raw = build_style_control("neutral", "strong", "").unwrap();
        let effective = refine_control_instruction(&raw, "This is the final result.");
        assert!(effective.starts_with("Deliberately affect-neutral while still fully natural and human"));
        assert!(effective.contains("preserving lexical stress"));
        assert!(effective.contains("small timing variation"));
        assert!(effective.contains("do not force a deep register"));
    }

    #[test]
    fn legacy_strong_neutral_is_upgraded_as_strong() {
        let raw = format!("clear, composed and deliberately neutral, but still human and expressive, {LEGACY_MANAGED_SUFFIX}");
        let effective = refine_control_instruction(&raw, "This is the final result.");
        assert!(effective.starts_with("Deliberately affect-neutral while still fully natural and human"));
    }

    #[test]
    fn neutral_has_no_hidden_cfg_or_gain_tuning() {
        for intensity in ["subtle", "normal", "strong"] {
            let raw = build_style_control("neutral", intensity, "").unwrap();
            assert_eq!(managed_style_tuning(Some(&raw)), ManagedStyleTuning::default());
            assert!((apply_managed_cfg(2.0, Some(&raw)) - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn sad_uses_low_arousal_cues_without_becoming_sleepy_or_flat() {
        let raw = build_style_control("sad", "normal", "").unwrap();
        let effective = refine_control_instruction(&raw, "I thought we would have more time.");
        assert!(effective.starts_with("Clearly sad and reflective"));
        assert!(effective.contains("moderately lower pitch centre"));
        assert!(effective.contains("noticeably narrower pitch range"));
        assert!(effective.contains("slower phrasing"));
        assert!(effective.contains("softer intensity"));
        assert!(effective.contains("do not sound sleepy"));
    }

    #[test]
    fn strong_sad_is_recognized_as_strong_and_not_grief() {
        let raw = build_style_control("sad", "strong", "").unwrap();
        let effective = refine_control_instruction(&raw, "I will miss you very much.");
        assert!(effective.starts_with("Deeply sad and emotionally affected"));
        assert!(effective.contains("low-arousal rather than grief-stricken"));
        assert!(effective.contains("never wail, sob, break the voice"));
    }

    #[test]
    fn serious_is_stance_not_forced_deep_voice() {
        let raw = build_style_control("serious", "normal", "").unwrap();
        let effective = refine_control_instruction(&raw, "This decision has consequences.");
        assert!(effective.starts_with("Serious, composed and focused"));
        assert!(effective.contains("natural pitch centre or only very slightly lower"));
        assert!(effective.contains("stable conversational intensity"));
        assert!(effective.contains("decisive falling phrase endings"));
        assert!(effective.contains("not angry, sad, ominous"));
    }

    #[test]
    fn strong_serious_is_recognized_without_grave_movie_voice() {
        let raw = build_style_control("serious", "strong", "").unwrap();
        assert!(raw.starts_with("strongly serious, deliberate and focused"));
        let effective = refine_control_instruction(&raw, "We need to act now.");
        assert!(effective.starts_with("Strongly serious, deliberate and consequential"));
        assert!(effective.contains("Do not imitate a movie-trailer voice"));
        assert!(effective.contains("force a deep register"));
    }

    #[test]
    fn sad_and_serious_demo_tuning_is_explicit_and_conservative() {
        let sad_subtle = build_style_control("sad", "subtle", "").unwrap();
        let sad_normal = build_style_control("sad", "normal", "").unwrap();
        let sad_strong = build_style_control("sad", "strong", "").unwrap();
        let serious_subtle = build_style_control("serious", "subtle", "").unwrap();
        let serious_normal = build_style_control("serious", "normal", "").unwrap();
        let serious_strong = build_style_control("serious", "strong", "").unwrap();
        assert_eq!(managed_style_tuning(Some(&sad_subtle)), ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 0.97 });
        assert_eq!(managed_style_tuning(Some(&sad_normal)), ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 0.94 });
        assert_eq!(managed_style_tuning(Some(&sad_strong)), ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 0.90 });
        assert_eq!(managed_style_tuning(Some(&serious_subtle)), ManagedStyleTuning { cfg_delta: 0.05, demo_gain_multiplier: 1.0 });
        assert_eq!(managed_style_tuning(Some(&serious_normal)), ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 1.0 });
        assert_eq!(managed_style_tuning(Some(&serious_strong)), ManagedStyleTuning { cfg_delta: 0.10, demo_gain_multiplier: 1.0 });
    }


    #[test]
    fn serious_to_warm_transition_is_continuous_not_two_takes() {
        let control = build_transition_control(
            "serious", "normal", "warm", "subtle", MoodTransitionMode::Gradual,
            "The report is difficult, but there is still something we can do.",
        ).unwrap();
        assert!(control.contains("Begin clearly serious and deliberate"));
        assert!(control.contains("End lightly warm and affiliative"));
        assert!(control.contains("release firmness and pitch restraint"));
        assert!(control.contains("believable intermediate state"));
        assert!(control.contains("Do not crossfade"));
    }

    #[test]
    fn angry_to_serious_transition_releases_tension_not_volume() {
        let control = build_transition_control(
            "angry", "strong", "serious", "normal", MoodTransitionMode::Quick,
            "This was unacceptable, and now we need to decide what happens next.",
        ).unwrap();
        assert!(control.contains("Strongly controlled cold anger") || control.contains("strongly controlled cold anger"));
        assert!(control.contains("release vocal tension"));
        assert!(control.contains("finish composed and consequential rather than angry"));
        assert!(control.contains("short natural span"));
    }

    #[test]
    fn transition_aliases_and_cfg_are_conservative() {
        let control = build_transition_control(
            "whisper-like", "subtle", "neutral", "normal", MoodTransitionMode::Gradual,
            "You can hear me now, so I can speak normally again.",
        ).unwrap();
        assert!(control.contains("near-whispered"));
        assert!(control.contains("affect-neutral"));
        let delta = managed_transition_cfg_delta("warm", "normal", "serious", "normal").unwrap();
        assert!((delta - 0.15).abs() < 1e-6);
        assert_eq!(recommended_transition_reference_style(), "neutral");
    }

    #[test]
    fn same_style_can_transition_between_intensities() {
        let control = build_transition_control(
            "angry", "normal", "angry", "subtle", MoodTransitionMode::Gradual,
            "I am still upset, but I am getting myself under control.",
        ).unwrap();
        assert!(control.contains("change only its strength"));
        assert!(control.contains("without resetting the voice or phrase"));
    }

    #[test]
    fn identical_transition_is_rejected() {
        let err = build_transition_control(
            "neutral", "normal", "neutral", "normal", MoodTransitionMode::Gradual, "Hello."
        ).unwrap_err();
        assert!(err.contains("endpoints are identical"));
    }

    #[test]
    fn managed_style_strength_is_conservative_and_explicit() {
        let warm = build_style_control("warm", "normal", "").unwrap();
        let subtle_cheerful = build_style_control("cheerful", "subtle", "").unwrap();
        let gentle = build_style_control("gentle", "normal", "").unwrap();
        let concerned = build_style_control("concerned", "normal", "").unwrap();
        let angry = build_style_control("angry", "normal", "").unwrap();
        let whisper = build_style_control("whisper", "normal", "").unwrap();
        assert_eq!(managed_style_tuning(Some(&warm)).cfg_delta, 0.20);
        assert_eq!(managed_style_tuning(Some(&warm)).demo_gain_multiplier, 1.05);
        assert_eq!(managed_style_tuning(Some(&subtle_cheerful)).cfg_delta, 0.10);
        assert_eq!(managed_style_tuning(Some(&gentle)).cfg_delta, 0.15);
        assert_eq!(managed_style_tuning(Some(&gentle)).demo_gain_multiplier, 0.95);
        assert_eq!(managed_style_tuning(Some(&concerned)).cfg_delta, 0.10);
        assert_eq!(managed_style_tuning(Some(&angry)).cfg_delta, 0.0);
        assert_eq!(managed_style_tuning(Some(&angry)).demo_gain_multiplier, 0.90);
        assert_eq!(managed_style_tuning(Some(&whisper)).cfg_delta, 0.10);
        assert_eq!(managed_style_tuning(Some(&whisper)).demo_gain_multiplier, 0.85);
        assert!((apply_managed_cfg(2.0, Some(&warm)) - 2.2).abs() < 1e-6);
        assert_eq!(apply_managed_cfg(2.95, Some(&warm)), 3.0);
    }


    #[test]
    fn extended_pair_specific_transitions_have_distinct_motion() {
        let cases = [
            ("neutral", "serious", "tighten focus"),
            ("cheerful", "excited", "raise arousal"),
            ("concerned", "serious", "composed focus"),
            ("warm", "gentle", "lowering vocal effort and projection"),
            ("warm", "sad", "reflective sadness"),
            ("sad", "serious", "restore enough energy"),
            ("concerned", "sad", "urgency drain away"),
            ("excited", "serious", "channel the energy into focus"),
            ("angry", "neutral", "release vocal tension"),
            ("angry", "concerned", "reduce hostility"),
            ("neutral", "whisper", "reduce projection and vocal effort"),
            ("serious", "whisper", "preserve the serious intent"),
        ];
        for (from, to, needle) in cases {
            let control = build_transition_control(
                from, "normal", to, "normal", MoodTransitionMode::Gradual,
                "This sentence is long enough to make a natural transition audible.",
            ).unwrap();
            assert!(control.contains(needle), "missing {needle:?} in {from}->{to}: {control}");
        }
    }

    #[test]
    fn managed_speed_transition_rejects_micro_and_extreme_changes() {
        let micro = MoodSpeedTransition::new(100.0, 103.0).unwrap_err();
        assert!(micro.contains("too subtle"));
        let extreme = MoodSpeedTransition::new(80.0, 130.0).unwrap_err();
        assert!(extreme.contains("too large"));
        assert!(parse_mood_speed_transition(None, Some(110.0)).unwrap_err().contains("supplied together"));
    }

    #[test]
    fn quick_speed_transition_is_fast_but_not_abrupt() {
        let speed = MoodSpeedTransition::new(90.0, 115.0).unwrap();
        let control = build_transition_control_with_speed(
            "serious", "normal", "warm", "normal", MoodTransitionMode::Quick,
            Some(speed),
            "We have considered the problem carefully, and now there is some genuinely good news to share.",
        ).unwrap();
        assert!(control.contains("90%"));
        assert!(control.contains("115%"));
        assert!(control.contains("at least two stressed words"));
        assert!(control.contains("never jump tempo on a single syllable or word"));
    }

    #[test]
    fn speed_only_transition_is_allowed_when_style_is_unchanged() {
        let speed = MoodSpeedTransition::new(88.0, 100.0).unwrap();
        let control = build_transition_control_with_speed(
            "neutral", "normal", "neutral", "normal", MoodTransitionMode::Gradual,
            Some(speed), "The explanation begins slowly and then returns to an ordinary conversational pace.",
        ).unwrap();
        assert!(control.contains("change only the speaking pace"));
        assert!(control.contains("Pace transition"));
    }

}
