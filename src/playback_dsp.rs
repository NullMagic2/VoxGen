pub const DEFAULT_SPEED_PERCENT: f32 = 100.0;
pub const MIN_SPEED_PERCENT: f32 = 50.0;
pub const MAX_SPEED_PERCENT: f32 = 200.0;
pub const DEFAULT_PITCH_SEMITONES: f32 = 0.0;
pub const MIN_PITCH_SEMITONES: f32 = -12.0;
pub const MAX_PITCH_SEMITONES: f32 = 12.0;

const RESAMPLER_HALF_TAPS: usize = 12; // 24-tap Lanczos-windowed sinc

/// Conservative sample-peak ceiling for final speech output.  Keeping a little
/// floating-point headroom prevents the old serializer from flattening expressive
/// peaks at +/-1.0 while remaining effectively transparent at normal levels.
pub const OUTPUT_PEAK_CEILING: f32 = 0.98;

/// Streaming peak-protection release.  Attack is immediate whenever the next
/// complete PCM block would exceed the ceiling; recovery is deliberately slow so
/// gain does not chatter between consecutive acoustic patches.
pub const OUTPUT_GUARD_RELEASE_MS: f32 = 250.0;

/// Below this normalized correlation WSOLA treats the match as unreliable.
/// Breath/noise-dominated and unvoiced regions do not contain a stable periodic
/// waveform to align; jumping to a weak accidental correlation can create
/// warble/metallic coloration.
const WSOLA_MIN_CONFIDENT_NCC: f64 = 0.20;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaybackControls {
    pub speed_percent: f32,
    pub pitch_semitones: f32,
}

impl Default for PlaybackControls {
    fn default() -> Self {
        Self {
            speed_percent: DEFAULT_SPEED_PERCENT,
            pitch_semitones: DEFAULT_PITCH_SEMITONES,
        }
    }
}

impl PlaybackControls {
    pub fn new(speed_percent: f32, pitch_semitones: f32) -> Result<Self, String> {
        if !speed_percent.is_finite()
            || !(MIN_SPEED_PERCENT..=MAX_SPEED_PERCENT).contains(&speed_percent)
        {
            return Err(format!(
                "speed_percent must be finite and between {MIN_SPEED_PERCENT:.0} and {MAX_SPEED_PERCENT:.0}"
            ));
        }
        if !pitch_semitones.is_finite()
            || !(MIN_PITCH_SEMITONES..=MAX_PITCH_SEMITONES).contains(&pitch_semitones)
        {
            return Err(format!(
                "pitch_semitones must be finite and between {MIN_PITCH_SEMITONES:.0} and {MAX_PITCH_SEMITONES:.0}"
            ));
        }
        Ok(Self {
            speed_percent,
            pitch_semitones,
        })
    }

    #[inline]
    pub fn active(self) -> bool {
        (self.speed_percent - DEFAULT_SPEED_PERCENT).abs() > 1.0e-4
            || (self.pitch_semitones - DEFAULT_PITCH_SEMITONES).abs() > 1.0e-4
    }

    #[inline]
    fn pitch_factor(self) -> f32 {
        2.0_f32.powf(self.pitch_semitones / 12.0)
    }

    #[inline]
    fn wsola_tempo(self) -> f32 {
        (self.speed_percent / 100.0 / self.pitch_factor()).clamp(0.25, 4.0)
    }
}

/// Stateful band-limited streaming resampler used only for pitch transposition.
///
/// A pitch factor `p` advances the source position by `p` samples per output
/// sample. The WSOLA stage then uses tempo `speed / p`, restoring the desired
/// duration while leaving the transposed pitch intact. A compact Lanczos-windowed
/// sinc kernel avoids the high-frequency roughness of linear interpolation and
/// applies an anti-alias cutoff when pitching upward (downsampling).
struct StreamingSincResampler {
    buffer: Vec<f32>,
    position: f64,
    factor: f64,
}

impl StreamingSincResampler {
    fn new() -> Self {
        Self {
            buffer: Vec::new(),
            position: 0.0,
            factor: 1.0,
        }
    }

    fn set_factor(&mut self, factor: f32) {
        if factor.is_finite() && factor > 0.0 {
            self.factor = (factor as f64).clamp(0.5, 2.0);
        }
    }

    #[inline]
    fn sinc(x: f64) -> f64 {
        if x.abs() < 1.0e-12 {
            1.0
        } else {
            let px = std::f64::consts::PI * x;
            px.sin() / px
        }
    }

    fn sample_at(&self, index: isize, draining: bool) -> f32 {
        if self.buffer.is_empty() {
            return 0.0;
        }
        if index < 0 {
            return self.buffer[0];
        }
        let index = index as usize;
        if index < self.buffer.len() {
            self.buffer[index]
        } else if draining {
            0.0
        } else {
            self.buffer[self.buffer.len() - 1]
        }
    }

    fn interpolate(&self, position: f64, draining: bool) -> f32 {
        let center = position.floor() as isize;
        let half = RESAMPLER_HALF_TAPS as isize;
        let cutoff = (1.0 / self.factor.max(1.0)).min(1.0);
        let mut sum = 0.0f64;
        let mut weight_sum = 0.0f64;
        for index in (center - half + 1)..=(center + half) {
            let distance = position - index as f64;
            let window_x = distance / RESAMPLER_HALF_TAPS as f64;
            if window_x.abs() >= 1.0 {
                continue;
            }
            let weight = cutoff * Self::sinc(distance * cutoff) * Self::sinc(window_x);
            sum += self.sample_at(index, draining) as f64 * weight;
            weight_sum += weight;
        }
        let sample = if weight_sum.abs() > 1.0e-12 {
            (sum / weight_sum) as f32
        } else {
            0.0
        };
        if sample.is_finite() { sample } else { 0.0 }
    }

    fn compact(&mut self) {
        let consumed = self.position.floor().max(0.0) as usize;
        let retain = RESAMPLER_HALF_TAPS + 2;
        let drain = consumed.saturating_sub(retain).min(self.buffer.len());
        if drain > 0 {
            self.buffer.drain(..drain);
            self.position -= drain as f64;
        }
    }

    fn produce(&mut self, draining: bool) -> Vec<f32> {
        let mut out = Vec::new();
        if self.buffer.is_empty() {
            return out;
        }
        loop {
            let can_emit = if draining {
                self.position < self.buffer.len() as f64
            } else {
                let needed = self.position.floor().max(0.0) as usize + RESAMPLER_HALF_TAPS;
                needed < self.buffer.len()
            };
            if !can_emit {
                break;
            }
            out.push(self.interpolate(self.position, draining));
            self.position += self.factor;
        }
        self.compact();
        out
    }

    fn push(&mut self, input: &[f32]) -> Vec<f32> {
        if (self.factor - 1.0).abs() < 1.0e-9 {
            // Exact pitch-neutral bypass. Speed-only playback must not pass through
            // a resampler at all; this preserves the generated waveform exactly
            // before the WSOLA tempo stage.
            return input
                .iter()
                .map(|&sample| if sample.is_finite() { sample } else { 0.0 })
                .collect();
        }
        self.buffer.extend(input.iter().map(|&sample| {
            if sample.is_finite() { sample } else { 0.0 }
        }));
        self.produce(false)
    }

    fn flush(&mut self) -> Vec<f32> {
        if (self.factor - 1.0).abs() < 1.0e-9 {
            self.buffer.clear();
            self.position = 0.0;
            return Vec::new();
        }
        let out = self.produce(true);
        self.buffer.clear();
        self.position = 0.0;
        out
    }
}

/// Speech-tuned, streaming WSOLA used by VoxGen for pitch-preserving tempo changes.
///
/// This deliberately mirrors the matching behavior of the pre-v0.7.40 VoxGen
/// client DSP that proved perceptually clean on narration:
///
/// - 30 ms analysis window / 15 ms overlap;
/// - 7.5 ms similarity search half-range;
/// - candidate stride of roughly 1/6000 second;
/// - normalized waveform correlation instead of raw dot-product energy.
///
/// Normalized correlation is important for expressive speech. A raw dot product
/// tends to prefer louder candidates even when their waveform phase is a poorer
/// continuation, which can sound like a short echo, doubling, or tense/metallic
/// coloration after repeated overlap-adds.
struct SpeechWsola {
    neutral: bool,
    window: usize,
    overlap: usize,
    synth_hop: usize,
    analysis_hop: f64,
    search: usize,
    search_step: usize,
    buffer: Vec<f32>,
    buffer_start: usize,
    initialized: bool,
    prev_segment: Option<Vec<f32>>,
    analysis_pos: f64,
    finished: bool,
    fade_out: Vec<f32>,
    fade_in: Vec<f32>,
}

impl SpeechWsola {
    fn new(sample_rate: u32, tempo: f32) -> Result<Self, String> {
        if sample_rate == 0 {
            return Err("sample_rate must be greater than zero".to_string());
        }
        let tempo = if tempo.is_finite() && tempo > 0.0 {
            tempo.clamp(0.25, 4.0)
        } else {
            1.0
        };
        let mut window = ((sample_rate as f64 * 0.030).round() as usize).max(256);
        if window % 2 != 0 {
            window += 1;
        }
        let overlap = window / 2;
        let synth_hop = window - overlap;
        let search = ((sample_rate as f64 * 0.0075).round() as usize).max(32);
        let search_step = ((sample_rate as f64 / 6000.0).round() as usize).max(1);
        let mut fade_out = Vec::with_capacity(overlap);
        let mut fade_in = Vec::with_capacity(overlap);
        if overlap <= 1 {
            fade_out.push(1.0);
            fade_in.push(0.0);
        } else {
            // WSOLA intentionally overlaps highly correlated regions of the SAME
            // speech waveform. Equal-power cos/sin fades are wrong here because
            // their gains sum to sqrt(2) at the midpoint and can add ~+3 dB before
            // the caller's requested gain is applied. Use an amplitude-complementary
            // raised-cosine crossfade instead: fade_out + fade_in == 1.0.
            for i in 0..overlap {
                let t = i as f32 / (overlap - 1) as f32;
                let fade_in_value = 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
                fade_in.push(fade_in_value);
                fade_out.push(1.0 - fade_in_value);
            }
        }
        Ok(Self {
            neutral: (tempo - 1.0).abs() < 1.0e-4,
            window,
            overlap,
            synth_hop,
            analysis_hop: synth_hop as f64 * tempo as f64,
            search,
            search_step,
            buffer: Vec::new(),
            buffer_start: 0,
            initialized: false,
            prev_segment: None,
            analysis_pos: 0.0,
            finished: false,
            fade_out,
            fade_in,
        })
    }

    fn append(&mut self, samples: &[f32]) {
        self.buffer.extend(samples.iter().map(|&sample| {
            if sample.is_finite() { sample } else { 0.0 }
        }));
    }

    fn slice_abs(&self, start: usize, len: usize, pad: bool) -> Option<Vec<f32>> {
        if start < self.buffer_start {
            return None;
        }
        let rel = start - self.buffer_start;
        let end = rel.saturating_add(len);
        if end <= self.buffer.len() {
            return Some(self.buffer[rel..end].to_vec());
        }
        if !pad || rel >= self.buffer.len() {
            return None;
        }
        let mut out = vec![0.0; len];
        let available = &self.buffer[rel..];
        out[..available.len()].copy_from_slice(available);
        Some(out)
    }

    fn compact(&mut self) {
        let analysis = self.analysis_pos.floor().max(0.0) as usize;
        let keep_from = self
            .buffer_start
            .max(analysis.saturating_sub(self.search + 8));
        let drop = keep_from
            .saturating_sub(self.buffer_start)
            .min(self.buffer.len());
        if drop > 0 {
            self.buffer.drain(..drop);
            self.buffer_start += drop;
        }
    }

    fn best_candidate(&self, predicted: f64, final_chunk: bool) -> Option<usize> {
        let prev = self.prev_segment.as_ref()?;
        let available_end = self.buffer_start + self.buffer.len();
        if available_end < self.window {
            return None;
        }
        let max_full = available_end - self.window;
        let predicted_rounded = predicted.round().max(0.0) as usize;
        let lo = predicted_rounded
            .saturating_sub(self.search)
            .max(self.buffer_start);
        let hi_target = predicted_rounded.saturating_add(self.search);
        if max_full < lo {
            return None;
        }
        if !final_chunk && max_full < hi_target {
            return None;
        }
        let hi = max_full.min(hi_target);
        if hi < lo || self.synth_hop + self.overlap > prev.len() {
            return None;
        }
        let predicted_fallback = predicted_rounded.clamp(lo, hi);

        let reference = &prev[self.synth_hop..self.synth_hop + self.overlap];
        let ref_energy = reference
            .iter()
            .fold(0.0f64, |sum, &x| sum + (x as f64) * (x as f64));
        let ref_rms = (ref_energy / self.overlap.max(1) as f64).sqrt();
        if ref_rms < 1.0e-4 {
            return Some(predicted_fallback);
        }
        let ref_norm = ref_energy.sqrt() + 1.0e-9;

        let mut best: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut cand = lo;
        loop {
            if let Some(probe) = self.slice_abs(cand, self.overlap, false) {
                let mut dot = 0.0f64;
                let mut probe_energy = 0.0f64;
                for (&a, &b) in reference.iter().zip(probe.iter()) {
                    let a = a as f64;
                    let b = b as f64;
                    dot += a * b;
                    probe_energy += b * b;
                }
                let denom = ref_norm * (probe_energy.sqrt() + 1.0e-9);
                let score = if denom > 1.0e-12 { dot / denom } else { -1.0 };
                // Prefer the candidate nearest the predicted analysis position when
                // correlations are effectively tied. This reduces unnecessary
                // pitch-period hopping on quasi-periodic voiced speech.
                let replace = score > best_score + 1.0e-9
                    || ((score - best_score).abs() <= 1.0e-9
                        && best.map_or(true, |old| {
                            cand.abs_diff(predicted_rounded) < old.abs_diff(predicted_rounded)
                        }));
                if replace {
                    best_score = score;
                    best = Some(cand);
                }
            }
            if cand >= hi {
                break;
            }
            cand = cand.saturating_add(self.search_step).min(hi);
        }
        // A weak NCC maximum on unvoiced/breathy/noise-like material is not a
        // trustworthy waveform synchronization point.  Using it can hop between
        // unrelated noise grains and sound metallic or phasey.  In that case keep
        // the nominal time trajectory and let the long complementary crossfade do
        // the continuity work.
        if best_score < WSOLA_MIN_CONFIDENT_NCC {
            Some(predicted_fallback)
        } else {
            best
        }
    }

    fn push(&mut self, samples: &[f32], final_chunk: bool) -> Vec<f32> {
        if self.finished {
            return Vec::new();
        }
        let clean = samples
            .iter()
            .map(|&sample| if sample.is_finite() { sample } else { 0.0 })
            .collect::<Vec<_>>();
        if self.neutral {
            if final_chunk {
                self.finished = true;
            }
            return clean;
        }
        self.append(&clean);
        let mut out = Vec::new();

        if !self.initialized {
            if self.buffer.len() < self.window && !final_chunk {
                return out;
            }
            let first = match self.slice_abs(self.buffer_start, self.window, final_chunk) {
                Some(segment) => segment,
                None => return out,
            };
            out.extend_from_slice(&first[..self.synth_hop.min(first.len())]);
            self.prev_segment = Some(first);
            self.analysis_pos = self.buffer_start as f64;
            self.initialized = true;
        }

        loop {
            let predicted = self.analysis_pos + self.analysis_hop;
            let Some(best) = self.best_candidate(predicted, final_chunk) else {
                break;
            };
            let Some(segment) = self.slice_abs(best, self.window, false) else {
                break;
            };
            let Some(prev) = self.prev_segment.as_ref() else {
                break;
            };
            let left = &prev[self.synth_hop..self.synth_hop + self.overlap];
            let right = &segment[..self.overlap];
            for i in 0..self.overlap {
                let sample = left[i] * self.fade_out[i] + right[i] * self.fade_in[i];
                out.push(if sample.is_finite() { sample } else { 0.0 });
            }
            self.prev_segment = Some(segment);
            self.analysis_pos = predicted;
            self.compact();
        }

        if final_chunk {
            if let Some(prev) = self.prev_segment.as_ref() {
                out.extend(prev[self.synth_hop..].iter().map(|&sample| {
                    if sample.is_finite() { sample } else { 0.0 }
                }));
            }
            self.finished = true;
            self.buffer.clear();
        }
        out
    }

    fn flush(&mut self) -> Vec<f32> {
        self.push(&[], true)
    }
}

/// VoxGen's authoritative streaming playback DSP.
///
/// Speed is handled by a speech-tuned normalized-correlation WSOLA. Pitch uses
/// a band-limited sinc resampler followed by compensating WSOLA so pitch and
/// duration remain independently controllable:
///
/// ```text
/// pitch_factor = 2^(semitones / 12)
/// resampler     = pitch_factor
/// WSOLA tempo   = speed / pitch_factor
/// ```
///
/// At 100% speed and 0 semitones the original samples are returned unchanged.
/// The DSP graph is kept warm so callers such as the desktop demo may change
/// controls during playback without maintaining a second DSP implementation.
pub struct StreamingPlaybackDsp {
    sample_rate: u32,
    controls: PlaybackControls,
    resampler: StreamingSincResampler,
    stretcher: SpeechWsola,
    effect_was_active: bool,
}

impl StreamingPlaybackDsp {
    pub fn new(sample_rate: u32, controls: PlaybackControls) -> Result<Self, String> {
        if sample_rate == 0 {
            return Err("sample_rate must be greater than zero".to_string());
        }
        let stretcher = SpeechWsola::new(sample_rate, controls.wsola_tempo())?;
        let mut resampler = StreamingSincResampler::new();
        resampler.set_factor(controls.pitch_factor());
        Ok(Self {
            sample_rate,
            controls,
            resampler,
            stretcher,
            effect_was_active: false,
        })
    }

    pub fn controls(&self) -> PlaybackControls {
        self.controls
    }

    /// Update controls for future, not-yet-rendered PCM.
    ///
    /// WSOLA overlap/search history is tempo-specific, so a control change starts
    /// a fresh short-history stretcher. Pitch changes additionally reset sinc
    /// phase/history.
    pub fn set_controls(&mut self, controls: PlaybackControls) -> Result<(), String> {
        if controls == self.controls {
            return Ok(());
        }
        let pitch_changed =
            (controls.pitch_semitones - self.controls.pitch_semitones).abs() > 1.0e-4;
        self.stretcher = SpeechWsola::new(self.sample_rate, controls.wsola_tempo())?;
        if pitch_changed {
            self.resampler = StreamingSincResampler::new();
        }
        self.resampler.set_factor(controls.pitch_factor());
        self.controls = controls;
        self.effect_was_active = false;
        Ok(())
    }

    fn resized_dry(input: &[f32], len: usize) -> Vec<f32> {
        if len == 0 || input.is_empty() {
            return Vec::new();
        }
        if len == input.len() {
            return input.to_vec();
        }
        if len == 1 {
            return vec![input[0]];
        }
        let mut out = Vec::with_capacity(len);
        let scale = (input.len() - 1) as f64 / (len - 1) as f64;
        for i in 0..len {
            let pos = i as f64 * scale;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(input.len() - 1);
            let frac = (pos - lo as f64) as f32;
            out.push(input[lo] + (input[hi] - input[lo]) * frac);
        }
        out
    }

    fn crossfade_in(processed: &mut [f32], dry: &[f32], sample_rate: u32) {
        let transition_samples = ((sample_rate as usize) / 100).max(1); // 10 ms
        let n = transition_samples.min(processed.len()).min(dry.len());
        if n == 0 {
            return;
        }
        for i in 0..n {
            let alpha = (i + 1) as f32 / n as f32;
            processed[i] = dry[i] * (1.0 - alpha) + processed[i] * alpha;
        }
    }

    pub fn push(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let clean = input
            .iter()
            .map(|&sample| if sample.is_finite() { sample } else { 0.0 })
            .collect::<Vec<_>>();
        if clean.is_empty() {
            return Ok(Vec::new());
        }

        let controls = self.controls;
        let pitch_factor = controls.pitch_factor();
        let active = controls.active();

        self.resampler.set_factor(pitch_factor);
        let transposed = self.resampler.push(&clean);
        let mut processed = if transposed.is_empty() {
            Vec::new()
        } else {
            self.stretcher.push(&transposed, false)
        };

        let out = if active {
            if !self.effect_was_active && !processed.is_empty() {
                let dry = Self::resized_dry(&clean, processed.len());
                Self::crossfade_in(&mut processed, &dry, self.sample_rate);
            }
            processed
        } else if self.effect_was_active && !processed.is_empty() {
            let transition_samples = ((self.sample_rate as usize) / 100).max(1);
            let n = transition_samples.min(clean.len()).min(processed.len());
            let mut dry = clean.clone();
            for i in 0..n {
                let alpha = (i + 1) as f32 / n as f32;
                dry[i] = processed[i] * (1.0 - alpha) + dry[i] * alpha;
            }
            dry
        } else {
            // Neutral bypass: bit-for-bit original finite PCM, while the native
            // graph remains available for possible live control changes.
            clean
        };
        self.effect_was_active = active;
        Ok(out)
    }

    pub fn finish(&mut self) -> Result<Vec<f32>, String> {
        let active = self.controls.active();
        let tail = self.resampler.flush();
        let mut out = if tail.is_empty() {
            Vec::new()
        } else {
            self.stretcher.push(&tail, false)
        };
        out.extend(self.stretcher.flush());
        if active {
            Ok(out)
        } else {
            // Neutral PCM was already returned through the dry bypass.
            Ok(Vec::new())
        }
    }

    pub fn process_all(
        sample_rate: u32,
        controls: PlaybackControls,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if !controls.active() {
            return Ok(input
                .iter()
                .map(|&sample| if sample.is_finite() { sample } else { 0.0 })
                .collect());
        }
        let mut dsp = Self::new(sample_rate, controls)?;
        let mut out = dsp.push(input)?;
        out.extend(dsp.finish()?);
        Ok(out)
    }
}

/// Style-agnostic final output protection.
///
/// VoxGen keeps floating-point headroom through synthesis, resampling and WSOLA.
/// This guard is the *only* level-safety stage before serialization.  It never
/// boosts audio and never changes pitch/timbre; it applies a single gain factor to
/// a complete streaming block and recovers slowly after hot peaks.
#[derive(Debug, Clone)]
pub struct OutputPeakGuard {
    sample_rate: u32,
    attenuation: f32,
}

impl OutputPeakGuard {
    pub fn new(sample_rate: u32) -> Result<Self, String> {
        if sample_rate == 0 {
            return Err("sample_rate must be greater than zero".to_string());
        }
        Ok(Self { sample_rate, attenuation: 1.0 })
    }

    #[inline]
    fn required_attenuation(input: &[f32], requested_gain: f32) -> f32 {
        if requested_gain <= 0.0 || input.is_empty() {
            return 1.0;
        }
        let mut peak = 0.0f32;
        for &sample in input {
            if sample.is_finite() {
                peak = peak.max((sample * requested_gain).abs());
            }
        }
        if peak > OUTPUT_PEAK_CEILING {
            (OUTPUT_PEAK_CEILING / peak).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Protect one already-rendered streaming block.  The whole block is known
    /// before it is sent, which gives the guard block look-ahead: attenuation can
    /// attack before the block's hottest sample rather than clipping that sample.
    pub fn process(&mut self, input: &[f32], requested_gain: f32) -> Result<Vec<f32>, String> {
        if !requested_gain.is_finite() || requested_gain < 0.0 {
            return Err("speech gain must be a finite value >= 0.0".to_string());
        }
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let required = Self::required_attenuation(input, requested_gain);
        if required < self.attenuation {
            // Instant attack: never let a newly hot block overshoot the ceiling.
            self.attenuation = required;
        } else if required > self.attenuation {
            // Slow monotonic release prevents patch-to-patch gain chatter/pumping.
            let seconds = input.len() as f32 / self.sample_rate as f32;
            let tau = OUTPUT_GUARD_RELEASE_MS / 1000.0;
            let alpha = 1.0 - (-seconds / tau.max(1.0e-3)).exp();
            self.attenuation += (required - self.attenuation) * alpha.clamp(0.0, 1.0);
            self.attenuation = self.attenuation.min(required);
        }
        let total_gain = requested_gain * self.attenuation;
        Ok(input
            .iter()
            .map(|&sample| {
                let value = if sample.is_finite() { sample * total_gain } else { 0.0 };
                // Numerical last line of defence.  Normal operation reaches this
                // below the ceiling, so this is not used as a loudness processor.
                value.clamp(-OUTPUT_PEAK_CEILING, OUTPUT_PEAK_CEILING)
            })
            .collect())
    }

    /// Protect a complete utterance with one uniform attenuation factor.  This is
    /// maximally transparent: dynamics inside the utterance are preserved exactly.
    pub fn process_all(sample_rate: u32, input: &[f32], requested_gain: f32) -> Result<Vec<f32>, String> {
        if sample_rate == 0 {
            return Err("sample_rate must be greater than zero".to_string());
        }
        if !requested_gain.is_finite() || requested_gain < 0.0 {
            return Err("speech gain must be a finite value >= 0.0".to_string());
        }
        let attenuation = Self::required_attenuation(input, requested_gain);
        let total_gain = requested_gain * attenuation;
        Ok(input
            .iter()
            .map(|&sample| {
                let value = if sample.is_finite() { sample * total_gain } else { 0.0 };
                value.clamp(-OUTPUT_PEAK_CEILING, OUTPUT_PEAK_CEILING)
            })
            .collect())
    }

    pub fn attenuation(&self) -> f32 {
        self.attenuation
    }
}

