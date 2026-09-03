use voxgen::{
    playback_dsp::{OutputPeakGuard, PlaybackControls as NativePlaybackControls, StreamingPlaybackDsp},
    prosody_control::{
        apply_managed_cfg, build_style_control, managed_style_tuning, refine_control_instruction,
    },
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    env,
    fs,
    io::{Read, Write},
    net::{Shutdown, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
#[cfg(windows)]
use std::{collections::VecDeque, ffi::c_void, io::{BufRead, BufReader}};
use wxdragon::{
    dialogs::file_dialog::FileDialog,
    prelude::*,
    sound::{Sound, SoundFlags},
    widgets::{
        combobox::{ComboBox, ComboBoxStyle},
        textctrl::TextCtrlStyle,
    },
};

const HOST: &str = "127.0.0.1";
const PORT: u16 = 8091;
const SERVER_ADDR: &str = "127.0.0.1:8091";
const HTTP_TIMEOUT: Duration = Duration::from_secs(300);
// Preserve the previous demo loudness as the portable default, but route it
// through VoxGen's explicit gain control instead of hard-coding playback boost.
const DEFAULT_GAIN_PERCENT: u32 = 100;
const MIN_GAIN_PERCENT: u32 = 0;
const MAX_GAIN_PERCENT: u32 = 400;
// A short fade suppresses the one-shot/reference onset chirp reported by
// VoxCPM2 users without deleting the first acoustic patch/word.
const DEMO_ONSET_FADE_SAMPLES: usize = 1_440; // 30 ms at 48 kHz
// Greek can sound unnaturally rushed with some cloned voices.  Rather than
// resampling the complete waveform (which changes pitch), the demo can extend
// short low-energy gaps that are likely to be inter-word boundaries.
const DEFAULT_WORD_SPACING_MS: u32 = 30;
const WORD_GAP_MIN_MS: u32 = 8;
const WORD_GAP_MAX_MS: u32 = 120;
const WORD_GAP_MIN_VOICED_MS: u32 = 35;

// Speech-oriented real-time playback DSP. Speed is handled by WSOLA, which
// aligns overlapping waveform segments in the time domain and avoids the
// characteristic metallic/phasey coloration of a small-window phase vocoder.
// Pitch uses a lightweight windowed-sinc resampler followed by compensating
// WSOLA time stretch, so speed and pitch remain independent.
const DEFAULT_SPEED_PERCENT: u32 = 100;
const MIN_SPEED_PERCENT: u32 = 50;
const MAX_SPEED_PERCENT: u32 = 200;
const DEFAULT_PITCH_SEMITONES: i32 = 0;
const MIN_PITCH_SEMITONES: i32 = -12;
const MAX_PITCH_SEMITONES: i32 = 12;
const DSP_SAMPLE_RATE: u32 = 48_000;
// Keep at most this many already-rendered live blocks pending in WinMM.
// This bounds how long a previous speed setting can remain audible after a live change.
const STREAM_MAX_PENDING_BLOCKS: usize = 2;
// At normal/slow playback speed, start from the first 160-ms patch: the first
// audible response matters more than stockpiling an extra patch before WinMM.
// Faster playback still raises the reserve because it consumes PCM more quickly.
const STREAM_PREBUFFER_MIN_PATCHES: usize = 1;
const STREAM_PREBUFFER_MAX_PATCHES: usize = 4;
const STREAM_PREBUFFER_RISK_RATIO: f64 = 0.85;

const STYLE_PRESETS: [(&str, &str); 12] = [
    ("auto", "Auto / text prosody"),
    ("neutral", "Neutral"),
    ("warm", "Warm"),
    ("cheerful", "Cheerful"),
    ("excited", "Excited"),
    ("sad", "Sad"),
    ("concerned", "Concerned"),
    ("angry", "Angry"),
    ("gentle", "Gentle"),
    ("serious", "Serious"),
    ("whisper", "Whisper-like"),
    ("custom", "Custom"),
];
const INTENSITIES: [(&str, &str); 3] = [
    ("subtle", "Subtle"),
    ("normal", "Normal"),
    ("strong", "Strong"),
];
const CLONE_MODES: [(&str, &str); 2] = [
    ("reference", "Controllable reference"),
    ("ultimate", "Ultimate cloning"),
];
const ENGINE_MODES: [(&str, &str); 2] = [
    ("normal", "Normal"),
    ("xtx7900", "XTX 7900"),
];
const DEFAULT_CFG_PERCENT: u32 = 200;
const DEFAULT_TEMPERATURE_PERCENT: u32 = 100;
const DEFAULT_INFERENCE_TIMESTEPS: u32 = 10;
const DEFAULT_VARIATIONS: u32 = 1;
const MAX_VARIATIONS: u32 = 3;

fn table_index(table: &[(&str, &str)], key: &str) -> u32 {
    table.iter().position(|(k, _)| *k == key).unwrap_or(0) as u32
}

fn table_key(table: &[(&'static str, &'static str)], index: Option<u32>) -> &'static str {
    index.and_then(|i| table.get(i as usize).map(|x| x.0)).unwrap_or(table[0].0)
}

fn table_label(table: &[(&'static str, &'static str)], index: Option<u32>) -> &'static str {
    index.and_then(|i| table.get(i as usize).map(|x| x.1)).unwrap_or(table[0].1)
}

fn update_emotion_sample_button(button: &Button, panel: &Panel, selection: Option<u32>) {
    let emotion = table_label(&STYLE_PRESETS, selection);
    button.set_label(&format!("Select {emotion} sample..."));
    // Clear any previous minimum first so the button can both grow and shrink.
    // wxWidgets then computes the native best size for the new label.
    button.set_min_size(Size::new(-1, -1));
    let best = button.get_best_size();
    button.set_min_size(best);
    button.set_size(best);
    panel.layout();
}


#[derive(Debug, Clone)]
struct DemoSettings {
    base_model: Option<PathBuf>,
    acoustic_model: Option<PathBuf>,
    voice_sample: Option<PathBuf>,
    word_spacing_ms: u32,
    speed_percent: u32,
    pitch_semitones: i32,
    gain_percent: u32,
    stream: bool,
    engine_mode: String,
    style_preset: String,
    style_intensity: String,
    custom_control: String,
    clone_mode: String,
    prompt_text: String,
    variations: u32,
    cfg_percent: u32,
    temperature_percent: u32,
    inference_timesteps: u32,
    emotion_references: BTreeMap<String, PathBuf>,
}

impl Default for DemoSettings {
    fn default() -> Self {
        Self {
            base_model: None,
            acoustic_model: None,
            voice_sample: None,
            word_spacing_ms: DEFAULT_WORD_SPACING_MS,
            speed_percent: DEFAULT_SPEED_PERCENT,
            pitch_semitones: DEFAULT_PITCH_SEMITONES,
            gain_percent: DEFAULT_GAIN_PERCENT,
            stream: true,
            engine_mode: "normal".to_string(),
            style_preset: "auto".to_string(),
            style_intensity: "normal".to_string(),
            custom_control: String::new(),
            clone_mode: "reference".to_string(),
            prompt_text: String::new(),
            variations: DEFAULT_VARIATIONS,
            cfg_percent: DEFAULT_CFG_PERCENT,
            temperature_percent: DEFAULT_TEMPERATURE_PERCENT,
            inference_timesteps: DEFAULT_INFERENCE_TIMESTEPS,
            emotion_references: BTreeMap::new(),
        }
    }
}

impl DemoSettings {
    fn load(path: &Path) -> Result<Self, String> {
        if !path.is_file() { return Ok(Self::default()); }
        let text = fs::read_to_string(path).map_err(|e| format!("read settings {}: {e}", path.display()))?;
        let mut out = Self::default();
        for (line_no, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with(';') { continue; }
            let Some((key, value)) = line.split_once('=') else {
                return Err(format!("invalid settings.cfg line {}: expected key=value", line_no + 1));
            };
            let key = key.trim();
            let value = value.trim();
            if let Some(style) = key.strip_prefix("emotion_reference.") {
                if STYLE_PRESETS.iter().any(|(k, _)| *k == style) {
                    if let Some(path) = nonempty_path(value) { out.emotion_references.insert(style.to_string(), path); }
                }
                continue;
            }
            match key {
                "base_model" => out.base_model = nonempty_path(value),
                "acoustic_model" => out.acoustic_model = nonempty_path(value),
                "voice_sample" => out.voice_sample = nonempty_path(value),
                "word_spacing_ms" => out.word_spacing_ms = value.parse::<u32>().map_err(|_| format!("invalid word_spacing_ms on line {}", line_no + 1))?.clamp(0, 100),
                "speed_percent" => out.speed_percent = value.parse::<u32>().map_err(|_| format!("invalid speed_percent on line {}", line_no + 1))?.clamp(MIN_SPEED_PERCENT, MAX_SPEED_PERCENT),
                "pitch_semitones" => out.pitch_semitones = value.parse::<i32>().map_err(|_| format!("invalid pitch_semitones on line {}", line_no + 1))?.clamp(MIN_PITCH_SEMITONES, MAX_PITCH_SEMITONES),
                "gain" => {
                    let gain = value.parse::<f32>().map_err(|_| format!("invalid gain on line {}", line_no + 1))?;
                    if !gain.is_finite() || gain < 0.0 { return Err(format!("invalid gain on line {}: use a finite value >= 0.0", line_no + 1)); }
                    out.gain_percent = ((gain * 100.0).round() as u32).clamp(MIN_GAIN_PERCENT, MAX_GAIN_PERCENT);
                }
                "stream" => out.stream = match value.to_ascii_lowercase().as_str() {
                    "on" | "true" | "1" | "yes" => true,
                    "off" | "false" | "0" | "no" => false,
                    _ => return Err(format!("invalid stream value on line {}: use on or off", line_no + 1)),
                },
                "mode" => if ENGINE_MODES.iter().any(|(k, _)| *k == value) { out.engine_mode = value.to_string(); },
                "style_preset" => if STYLE_PRESETS.iter().any(|(k, _)| *k == value) { out.style_preset = value.to_string(); },
                "style_intensity" => if INTENSITIES.iter().any(|(k, _)| *k == value) { out.style_intensity = value.to_string(); },
                "custom_control" => out.custom_control = value.to_string(),
                "clone_mode" => if CLONE_MODES.iter().any(|(k, _)| *k == value) { out.clone_mode = value.to_string(); },
                "prompt_text" => out.prompt_text = value.to_string(),
                "variations" => out.variations = value.parse::<u32>().map_err(|_| format!("invalid variations on line {}", line_no + 1))?.clamp(1, MAX_VARIATIONS),
                "cfg" => {
                    let x=value.parse::<f32>().map_err(|_| format!("invalid cfg on line {}",line_no+1))?;
                    out.cfg_percent=((x*100.0).round() as u32).clamp(100,300);
                }
                "temperature" => {
                    let x=value.parse::<f32>().map_err(|_| format!("invalid temperature on line {}",line_no+1))?;
                    out.temperature_percent=((x*100.0).round() as u32).clamp(50,150);
                }
                "inference_timesteps" => out.inference_timesteps=value.parse::<u32>().map_err(|_| format!("invalid inference_timesteps on line {}",line_no+1))?.clamp(4,30),
                _ => {}
            }
        }
        // v0.7.36 migration: an explicitly saved Neutral preset is the canonical
        // fallback identity. Mirror it into the legacy voice_sample field only
        // when that older field was empty, while resolution still prefers the
        // explicit neutral entry directly.
        if out.voice_sample.is_none() {
            if let Some(neutral) = out.emotion_references.get("neutral").cloned() {
                out.voice_sample = Some(neutral);
            }
        }
        Ok(out)
    }

    fn save(&self, path: &Path) -> Result<(), String> {
        let parent = path.parent().ok_or_else(|| format!("settings path has no parent: {}", path.display()))?;
        if !parent.is_dir() { return Err(format!("settings directory does not exist: {}", parent.display())); }
        let clean = |s: &str| s.replace('\r', " ").replace('\n', " ");
        let mut text = format!(
            "# VoxGen Demo portable settings\n# This file intentionally lives beside voxgen-demo/voxgen-demo.exe.\nbase_model={}\nacoustic_model={}\nvoice_sample={}\nword_spacing_ms={}\nspeed_percent={}\npitch_semitones={}\ngain={:.2}\nstream={}\nmode={}\nstyle_preset={}\nstyle_intensity={}\ncustom_control={}\nclone_mode={}\nprompt_text={}\nvariations={}\ncfg={:.2}\ntemperature={:.2}\ninference_timesteps={}\n",
            path_value(self.base_model.as_deref()),
            path_value(self.acoustic_model.as_deref()),
            path_value(self.voice_sample.as_deref()),
            self.word_spacing_ms.clamp(0,100),
            self.speed_percent.clamp(MIN_SPEED_PERCENT,MAX_SPEED_PERCENT),
            self.pitch_semitones.clamp(MIN_PITCH_SEMITONES,MAX_PITCH_SEMITONES),
            self.gain_percent.clamp(MIN_GAIN_PERCENT,MAX_GAIN_PERCENT) as f32/100.0,
            if self.stream{"on"}else{"off"},
            self.engine_mode,
            self.style_preset,
            self.style_intensity,
            clean(&self.custom_control),
            self.clone_mode,
            clean(&self.prompt_text),
            self.variations.clamp(1,MAX_VARIATIONS),
            self.cfg_percent.clamp(100,300) as f32/100.0,
            self.temperature_percent.clamp(50,150) as f32/100.0,
            self.inference_timesteps.clamp(4,30),
        );
        for (key, value) in &self.emotion_references {
            text.push_str(&format!("emotion_reference.{key}={}\n", path_value(Some(value.as_path()))));
        }
        fs::write(path, text).map_err(|e| format!("write settings {}: {e}", path.display()))?;
        Ok(())
    }
}

type SharedSettings = Arc<Mutex<DemoSettings>>;

fn nonempty_path(value: &str) -> Option<PathBuf> {
    if value.is_empty() {
        None
    } else {
        Some(PathBuf::from(value))
    }
}

fn path_value(path: Option<&Path>) -> String {
    path.map(|p| p.to_string_lossy().into_owned()).unwrap_or_default()
}

fn demo_settings_path() -> PathBuf {
    env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join("settings.cfg")))
        .unwrap_or_else(|| PathBuf::from("settings.cfg"))
}

fn save_shared_settings(settings: &SharedSettings) -> Result<PathBuf, String> {
    let path = demo_settings_path();
    let snapshot = settings
        .lock()
        .map_err(|_| "demo settings lock poisoned".to_string())?
        .clone();
    snapshot.save(&path)?;
    Ok(path)
}

fn existing_file(path: Option<PathBuf>) -> Option<PathBuf> {
    path.filter(|p| p.is_file())
}

#[derive(Clone)]
struct LivePlaybackControls {
    speed_percent: Arc<AtomicU32>,
    pitch_semitones: Arc<AtomicI32>,
}

impl Default for LivePlaybackControls {
    fn default() -> Self {
        Self::new(DEFAULT_SPEED_PERCENT, DEFAULT_PITCH_SEMITONES)
    }
}

impl LivePlaybackControls {
    fn new(speed_percent: u32, pitch_semitones: i32) -> Self {
        Self {
            speed_percent: Arc::new(AtomicU32::new(
                speed_percent.clamp(MIN_SPEED_PERCENT, MAX_SPEED_PERCENT),
            )),
            pitch_semitones: Arc::new(AtomicI32::new(
                pitch_semitones.clamp(MIN_PITCH_SEMITONES, MAX_PITCH_SEMITONES),
            )),
        }
    }

    fn speed_percent(&self) -> u32 {
        self.speed_percent
            .load(Ordering::Relaxed)
            .clamp(MIN_SPEED_PERCENT, MAX_SPEED_PERCENT)
    }

    fn pitch_semitones(&self) -> i32 {
        self.pitch_semitones
            .load(Ordering::Relaxed)
            .clamp(MIN_PITCH_SEMITONES, MAX_PITCH_SEMITONES)
    }

    fn set_speed_percent(&self, value: i32) {
        self.speed_percent.store(
            value.clamp(MIN_SPEED_PERCENT as i32, MAX_SPEED_PERCENT as i32) as u32,
            Ordering::Relaxed,
        );
    }

    fn set_pitch_semitones(&self, value: i32) {
        self.pitch_semitones.store(
            value.clamp(MIN_PITCH_SEMITONES, MAX_PITCH_SEMITONES),
            Ordering::Relaxed,
        );
    }
}

#[derive(Clone)]
struct SynthesisCancel {
    cancelled: Arc<AtomicBool>,
    request_id: Arc<AtomicU64>,
    #[cfg(windows)]
    active_waveout: Arc<Mutex<Option<usize>>>,
}

impl Default for SynthesisCancel {
    fn default() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            request_id: Arc::new(AtomicU64::new(0)),
            #[cfg(windows)]
            active_waveout: Arc::new(Mutex::new(None)),
        }
    }
}

impl SynthesisCancel {
    fn begin(&self) -> u64 {
        let request_id = next_demo_seed();
        self.request_id.store(request_id, Ordering::Release);
        self.cancelled.store(false, Ordering::Release);
        request_id
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        #[cfg(windows)]
        if let Ok(active) = self.active_waveout.lock() {
            if let Some(raw) = *active {
                // waveOutReset is the WinMM "stop now" primitive: it returns all
                // queued headers immediately instead of waiting for buffered PCM.
                unsafe { let _ = waveOutReset(raw as HWaveOut); }
            }
        }
    }

    fn request_id(&self) -> u64 {
        self.request_id.load(Ordering::Acquire)
    }

    #[cfg(windows)]
    fn register_waveout(&self, handle: HWaveOut) {
        if let Ok(mut active) = self.active_waveout.lock() {
            *active = Some(handle as usize);
            if self.is_cancelled() {
                unsafe { let _ = waveOutReset(handle); }
            }
        }
    }

    #[cfg(windows)]
    fn unregister_waveout(&self, handle: HWaveOut) {
        if let Ok(mut active) = self.active_waveout.lock() {
            if active.as_ref().copied() == Some(handle as usize) {
                *active = None;
            }
        }
    }
}

/// Thin demo adapter over VoxGen's authoritative native playback DSP.
///
/// The algorithm itself lives in the engine crate (`src/playback_dsp.rs`) so the
/// desktop demo and HTTP clients cannot drift into different speed/pitch math.
struct RealtimeVoiceProcessor {
    dsp: StreamingPlaybackDsp,
    peak_guard: OutputPeakGuard,
}

impl RealtimeVoiceProcessor {
    fn new() -> Self {
        let controls = NativePlaybackControls::new(
            DEFAULT_SPEED_PERCENT as f32,
            DEFAULT_PITCH_SEMITONES as f32,
        ).expect("fixed VoxGen playback controls must be valid");
        Self {
            dsp: StreamingPlaybackDsp::new(DSP_SAMPLE_RATE, controls)
                .expect("fixed VoxGen native playback DSP configuration must be valid"),
            peak_guard: OutputPeakGuard::new(DSP_SAMPLE_RATE)
                .expect("fixed VoxGen output peak guard configuration must be valid"),
        }
    }

    fn sync_controls(&mut self, controls: &LivePlaybackControls) {
        let native = NativePlaybackControls::new(
            controls.speed_percent() as f32,
            controls.pitch_semitones() as f32,
        ).expect("demo sliders clamp to VoxGen native DSP limits");
        self.dsp
            .set_controls(native)
            .expect("VoxGen native playback DSP control update must be valid");
    }

    fn push(&mut self, input: &[f32], controls: &LivePlaybackControls) -> Vec<f32> {
        self.sync_controls(controls);
        let rendered = self.dsp
            .push(input)
            .expect("VoxGen native playback DSP stream processing failed");
        self.peak_guard
            .process(&rendered, 1.0)
            .expect("VoxGen output peak guard failed")
    }

    fn finish(&mut self, controls: &LivePlaybackControls) -> Vec<f32> {
        self.sync_controls(controls);
        let rendered = self.dsp
            .finish()
            .expect("VoxGen native playback DSP flush failed");
        self.peak_guard
            .process(&rendered, 1.0)
            .expect("VoxGen output peak guard failed")
    }
}

#[derive(Default)]
struct DemoState {
    voice_sample: Option<PathBuf>,
    base_model: Option<PathBuf>,
    acoustic_model: Option<PathBuf>,
    child: Option<Child>,
    owns_server: bool,
    playback_file: Option<PathBuf>,
}

type SharedState = Arc<Mutex<DemoState>>;

fn append_log(log: TextCtrl, message: &str) {
    log.append_text(message);
    if !message.ends_with('\n') {
        log.append_text("\n");
    }
    log.set_insertion_point_end();
}


/// Streaming pause expander used by the demo playback path.
///
/// It does not alter voiced samples or their sample rate.  It measures
/// low-energy runs as they stream and, when a short gap ends after a meaningful
/// voiced run, inserts a small amount of silence before speech resumes.  That makes
/// rapid languages/voices easier to follow without lowering pitch or stretching
/// vowels.  Long punctuation/sentence pauses are intentionally left alone.
struct WordSpacingProcessor {
    extra_samples: usize,
    min_gap_samples: usize,
    max_gap_samples: usize,
    min_voiced_samples: usize,
    envelope: f32,
    peak_envelope: f32,
    heard_voiced: bool,
    quiet_run_samples: usize,
    voiced_since_gap: usize,
}

impl WordSpacingProcessor {
    fn new(extra_ms: u32, sample_rate: u32) -> Self {
        let samples_for_ms = |ms: u32| -> usize {
            ((sample_rate as u64 * ms as u64) / 1000) as usize
        };
        Self {
            extra_samples: samples_for_ms(extra_ms),
            min_gap_samples: samples_for_ms(WORD_GAP_MIN_MS),
            max_gap_samples: samples_for_ms(WORD_GAP_MAX_MS),
            min_voiced_samples: samples_for_ms(WORD_GAP_MIN_VOICED_MS),
            envelope: 0.0,
            peak_envelope: 0.0,
            heard_voiced: false,
            quiet_run_samples: 0,
            voiced_since_gap: 0,
        }
    }

    fn push(&mut self, input: &[f32]) -> Vec<f32> {
        if self.extra_samples == 0 {
            return input.to_vec();
        }

        let mut out = Vec::with_capacity(input.len() + self.extra_samples);
        for &sample in input {
            let amplitude = sample.abs();
            // ~4 ms envelope plus a much slower recent-peak estimate.  The
            // relative threshold adapts to quiet/loud cloned voices.
            self.envelope = self.envelope * 0.995 + amplitude * 0.005;
            self.peak_envelope = (self.peak_envelope * 0.99995).max(self.envelope);
            let quiet_threshold = (self.peak_envelope * 0.10).clamp(0.0015, 0.018);
            let quiet = self.heard_voiced && self.envelope < quiet_threshold;

            if quiet {
                // Emit the original gap immediately.  We only need its length
                // to decide whether to add extra spacing when speech resumes;
                // this keeps the streaming player from stalling on pauses.
                self.quiet_run_samples = self.quiet_run_samples.saturating_add(1);
                out.push(sample);
                continue;
            }

            if self.quiet_run_samples > 0 {
                let looks_like_word_gap = self.quiet_run_samples >= self.min_gap_samples
                    && self.quiet_run_samples <= self.max_gap_samples
                    && self.voiced_since_gap >= self.min_voiced_samples;
                if looks_like_word_gap {
                    out.resize(out.len() + self.extra_samples, 0.0);
                    self.voiced_since_gap = 0;
                }
                self.quiet_run_samples = 0;
            }

            out.push(sample);
            if self.envelope >= quiet_threshold {
                self.heard_voiced = true;
                self.voiced_since_gap = self.voiced_since_gap.saturating_add(1);
            }
        }
        out
    }

    fn finish(&mut self) -> Vec<f32> {
        // The original trailing silence has already been emitted; never append
        // extra word spacing at the end of an utterance.
        Vec::new()
    }
}

fn is_voxgen_root(path: &Path) -> bool {
    path.join("Cargo.toml").is_file()
        && path.join("src").join("main.rs").is_file()
        && path.join("shaders").is_dir()
        && path.join("build_voxgen.bat").is_file()
        && path.join("build_voxgen.sh").is_file()
}

fn search_up_for_voxgen_root(start: &Path) -> Option<PathBuf> {
    let mut cursor = if start.is_file() {
        start.parent()?.to_path_buf()
    } else {
        start.to_path_buf()
    };
    loop {
        if is_voxgen_root(&cursor) {
            return Some(cursor);
        }
        if !cursor.pop() {
            break;
        }
    }
    None
}

fn project_root() -> PathBuf {
    // An explicit override wins, but only when it really identifies a VoxGen
    // source tree. This catches stale/accidental VOXGEN_ROOT values early.
    if let Some(value) = env::var_os("VOXGEN_ROOT") {
        let p = PathBuf::from(value);
        if is_voxgen_root(&p) {
            return p;
        }
    }

    // First search upward from the process working directory. run_demo.bat/.sh
    // deliberately start us at the project root, but direct launches from
    // demo/ or demo/target/release must work too.
    if let Ok(cwd) = env::current_dir() {
        if let Some(root) = search_up_for_voxgen_root(&cwd) {
            return root;
        }
    }

    // A GUI can also be launched by double-clicking its binary, in which case
    // the working directory is arbitrary. Search upward from the executable
    // itself: .../VoxGen/demo/target/release/voxgen-demo.exe -> .../VoxGen.
    if let Ok(exe) = env::current_exe() {
        if let Some(root) = search_up_for_voxgen_root(&exe) {
            return root;
        }
    }

    // Preserve a useful fallback for diagnostics. find_voxgen_binary() will
    // report the exact PathBuf candidates rather than fabricating separators.
    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn model_dir(root: &Path) -> PathBuf {
    env::var_os("VOXGEN_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("models"))
}

fn find_base_model(root: &Path) -> Result<PathBuf, String> {
    if let Some(value) = env::var_os("VOXGEN_BASE_MODEL") {
        let p = PathBuf::from(value);
        if p.is_file() {
            return Ok(p);
        }
        return Err(format!("VOXGEN_BASE_MODEL does not exist: {}", p.display()));
    }

    let dir = model_dir(root);
    let q8 = dir.join("VoxCPM2-BaseLM-Q8_0.gguf");
    if q8.is_file() {
        return Ok(q8);
    }
    let f16 = dir.join("VoxCPM2-BaseLM-F16.gguf");
    if f16.is_file() {
        return Ok(f16);
    }
    Err(format!(
        "Could not find VoxCPM2 BaseLM. Expected {} or {}. Set VOXGEN_MODEL_DIR or VOXGEN_BASE_MODEL.",
        q8.display(),
        f16.display()
    ))
}

fn find_acoustic_model(root: &Path) -> Result<PathBuf, String> {
    if let Some(value) = env::var_os("VOXGEN_ACOUSTIC") {
        let p = PathBuf::from(value);
        if p.is_file() {
            return Ok(p);
        }
        return Err(format!("VOXGEN_ACOUSTIC does not exist: {}", p.display()));
    }
    let p = model_dir(root).join("VoxCPM2-Acoustic-F16.gguf");
    if p.is_file() {
        Ok(p)
    } else {
        Err(format!(
            "Could not find acoustic model at {}. Set VOXGEN_MODEL_DIR or VOXGEN_ACOUSTIC.",
            p.display()
        ))
    }
}

fn find_voxgen_binary(root: &Path) -> Result<PathBuf, String> {
    if let Some(value) = env::var_os("VOXGEN_BIN") {
        let p = PathBuf::from(value);
        if p.is_file() {
            return Ok(p);
        }
        return Err(format!("VOXGEN_BIN does not exist: {}", p.display()));
    }

    #[cfg(windows)]
    let name = "voxgen.exe";
    #[cfg(not(windows))]
    let name = "voxgen";

    // Portable deployment: if voxgen lives beside voxgen-demo, prefer that
    // before consulting a source-tree target directory. This lets users copy
    // the two release binaries into one folder and run the demo without a
    // VoxGen checkout or VOXGEN_ROOT.
    let adjacent = env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|dir| dir.join(name)));
    if let Some(p) = adjacent.as_ref() {
        if p.is_file() {
            return Ok(p.clone());
        }
    }

    for p in [
        root.join("target").join("release").join(name),
        root.join("target").join("debug").join(name),
    ] {
        if p.is_file() {
            return Ok(p);
        }
    }
    let release = root.join("target").join("release").join(name);
    let debug = root.join("target").join("debug").join(name);
    let adjacent_display = adjacent
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| format!("<demo executable directory>{}{}", std::path::MAIN_SEPARATOR, name));
    Err(format!(
        "VoxGen binary not found. Expected it beside the demo at {}, or at {} or {}. Build VoxGen first, copy it next to the demo, or set VOXGEN_BIN.",
        adjacent_display,
        release.display(),
        debug.display()
    ))
}

fn connect() -> Result<TcpStream, String> {
    let stream = TcpStream::connect_timeout(
        &SERVER_ADDR
            .parse()
            .map_err(|e| format!("invalid server address: {e}"))?,
        Duration::from_secs(2),
    )
    .map_err(|e| format!("connect {SERVER_ADDR}: {e}"))?;
    stream
        .set_read_timeout(Some(HTTP_TIMEOUT))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(10)))
        .map_err(|e| e.to_string())?;
    Ok(stream)
}

#[derive(Debug)]
struct HttpResponse {
    body: Vec<u8>,
    headers: Vec<(String, String)>,
}

impl HttpResponse {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

fn http_request_full(method: &str, path: &str, body: &[u8]) -> Result<HttpResponse, String> {
    let mut stream = connect()?;
    let content_type = if body.is_empty() {
        "application/octet-stream"
    } else {
        "application/json"
    };
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {HOST}:{PORT}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .and_then(|_| stream.write_all(body))
        .and_then(|_| stream.flush())
        .map_err(|e| format!("write HTTP request: {e}"))?;
    let _ = stream.shutdown(Shutdown::Write);

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|e| format!("read HTTP response: {e}"))?;

    let split = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| "VoxGen returned malformed HTTP response".to_string())?;
    let head = std::str::from_utf8(&response[..split])
        .map_err(|e| format!("invalid HTTP header: {e}"))?;
    let status = head.lines().next().unwrap_or("");
    if !status.contains(" 200 ") {
        let payload = String::from_utf8_lossy(&response[split + 4..]);
        return Err(format!("VoxGen request failed ({status}): {payload}"));
    }
    let headers = head
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once(':'))
        .map(|(k,v)| (k.trim().to_string(),v.trim().to_string()))
        .collect();
    Ok(HttpResponse { body: response[split + 4..].to_vec(), headers })
}

fn http_request(method: &str, path: &str, body: &[u8]) -> Result<Vec<u8>, String> {
    Ok(http_request_full(method, path, body)?.body)
}

fn cancel_active_server_speech(request_id: u64) -> Result<(), String> {
    let body = serde_json::to_vec(&json!({"request_id": request_id})).map_err(|e| e.to_string())?;
    let _ = http_request("POST", "/v1/audio/speech/cancel", &body)?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HealthState {
    Ready,
    EngineOnly,
    Unavailable,
}

fn health_state() -> HealthState {
    let Ok(body) = http_request("GET", "/health", &[]) else {
        return HealthState::Unavailable;
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&body) else {
        return HealthState::Unavailable;
    };
    if value.get("engine").and_then(|v| v.as_str()) != Some("VoxGen") {
        return HealthState::Unavailable;
    }
    if value
        .get("speech_inference_ready")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        HealthState::Ready
    } else {
        HealthState::EngineOnly
    }
}

fn health_check() -> bool {
    health_state() == HealthState::Ready
}

fn engine_check() -> bool {
    health_state() != HealthState::Unavailable
}

fn server_version() -> Option<String> {
    let body = http_request("GET", "/health", &[]).ok()?;
    let value = serde_json::from_slice::<serde_json::Value>(&body).ok()?;
    if value.get("engine").and_then(|v| v.as_str()) != Some("VoxGen") {
        return None;
    }
    value.get("version").and_then(|v| v.as_str()).map(str::to_owned)
}

fn server_streaming_enabled() -> Option<bool> {
    let body = http_request("GET", "/health", &[]).ok()?;
    let value = serde_json::from_slice::<serde_json::Value>(&body).ok()?;
    if value.get("engine").and_then(|v| v.as_str()) != Some("VoxGen") {
        return None;
    }
    value.get("streaming_enabled").and_then(|v| v.as_bool())
}

fn server_execution_mode() -> Option<String> {
    let body = http_request("GET", "/health", &[]).ok()?;
    let value = serde_json::from_slice::<serde_json::Value>(&body).ok()?;
    if value.get("engine").and_then(|v| v.as_str()) != Some("VoxGen") { return None; }
    // VoxGen builds predating --mode did not expose a mode field. Those builds
    // necessarily used the portable/generic Vulkan path, so treat them as Normal
    // instead of reporting a mismatch for every mode selection.
    Some(
        value
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("normal")
            .to_string(),
    )
}

fn server_xtx_stream_safe() -> Option<bool> {
    let body = http_request("GET", "/health", &[]).ok()?;
    let value = serde_json::from_slice::<serde_json::Value>(&body).ok()?;
    if value.get("engine").and_then(|v| v.as_str()) != Some("VoxGen") { return None; }
    // v0.7.31 exposes both flags. Missing fields mean an older XTX server whose
    // profiler/coopmat policy is unknown, so require a restart before streaming.
    let profiling = value.get("gpu_profile").and_then(|v| v.as_bool());
    let coopmat = value.get("xtx_coopmat").and_then(|v| v.as_bool());
    Some(profiling == Some(false) && coopmat == Some(false))
}

fn server_benchmark_profile() -> Option<bool> {
    let body = http_request("GET", "/health", &[]).ok()?;
    let value = serde_json::from_slice::<serde_json::Value>(&body).ok()?;
    if value.get("engine").and_then(|v| v.as_str()) != Some("VoxGen") { return None; }
    value.get("benchmark_profile").and_then(|v| v.as_bool())
}

fn wait_for_server_down() -> bool {
    for _ in 0..50 {
        if !engine_check() {
            return true;
        }
        thread::sleep(Duration::from_millis(100));
    }
    false
}

#[cfg(windows)]
fn windows_listener_pid(port: u16) -> Result<u32, String> {
    let output = Command::new("netstat")
        .args(["-ano", "-p", "tcp"])
        .output()
        .map_err(|e| format!("run netstat: {e}"))?;
    if !output.status.success() {
        return Err(format!("netstat failed with {}", output.status));
    }
    let needle = format!(":{port}");
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() < 5 || !fields[0].eq_ignore_ascii_case("TCP") {
            continue;
        }
        if !fields[1].ends_with(&needle) || !fields[3].eq_ignore_ascii_case("LISTENING") {
            continue;
        }
        if let Ok(pid) = fields[4].parse::<u32>() {
            return Ok(pid);
        }
    }
    Err(format!("could not resolve the process listening on TCP port {port}"))
}

#[cfg(windows)]
fn terminate_legacy_voxgen_server_windows() -> Result<(), String> {
    // Only reach this fallback after /health has positively identified the listener
    // as VoxGen. Resolve the owning PID, then verify its image name before killing it.
    let pid = windows_listener_pid(PORT)?;
    let filter = format!("PID eq {pid}");
    let output = Command::new("tasklist")
        .args(["/FI", filter.as_str(), "/FO", "CSV", "/NH"])
        .output()
        .map_err(|e| format!("run tasklist: {e}"))?;
    let listing = String::from_utf8_lossy(&output.stdout).to_lowercase();
    if !listing.contains("voxgen") {
        return Err(format!(
            "TCP port {PORT} belongs to PID {pid}, but Windows does not identify that process as VoxGen; refusing to terminate it"
        ));
    }
    let pid_text = pid.to_string();
    let status = Command::new("taskkill")
        .args(["/PID", pid_text.as_str(), "/T", "/F"])
        .status()
        .map_err(|e| format!("run taskkill for VoxGen PID {pid}: {e}"))?;
    if !status.success() {
        return Err(format!("taskkill failed for VoxGen PID {pid} with {status}"));
    }
    Ok(())
}

fn stop_existing_voxgen_server(state: &SharedState) -> Result<(), String> {
    {
        let mut guard = state.lock().map_err(|_| "demo state poisoned".to_string())?;
        if guard.owns_server {
            if let Some(child) = guard.child.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
            guard.child = None;
            guard.owns_server = false;
            if wait_for_server_down() {
                return Ok(());
            }
        }
    }

    if !engine_check() {
        return Ok(());
    }

    // Never terminate an arbitrary listener. The existing endpoint must first
    // positively identify itself as VoxGen through /health.
    if server_execution_mode().is_none() {
        return Err(format!(
            "TCP port {PORT} is occupied, but the listener is not a recognized VoxGen server"
        ));
    }

    // v0.7.25+ supports a loopback-only graceful shutdown endpoint. This also
    // handles a server left behind by an earlier demo process without relying on
    // the current demo's Child handle.
    if http_request("POST", "/v1/server/shutdown", b"{}").is_ok() && wait_for_server_down() {
        return Ok(());
    }

    // Compatibility takeover for older Windows VoxGen servers, which do not
    // expose the graceful shutdown endpoint. /health above has already confirmed
    // that the listener is VoxGen; verify the owning image before taskkill.
    #[cfg(windows)]
    {
        terminate_legacy_voxgen_server_windows()?;
        if wait_for_server_down() {
            return Ok(());
        }
        return Err("VoxGen was terminated but TCP port 8091 did not become available in time".to_string());
    }

    #[cfg(not(windows))]
    {
        Err("The running VoxGen server predates graceful mode switching. Stop that legacy server once, then click Load VoxCPM2 again.".to_string())
    }
}

fn ensure_server_with_profile(
    state: &SharedState,
    stream_enabled: bool,
    default_gain: f32,
    engine_mode: &str,
    benchmark_profile: bool,
) -> Result<String, String> {
    let stream_enabled = if benchmark_profile { false } else { stream_enabled };
    if benchmark_profile && engine_mode != "xtx7900" {
        return Err("Offline GPU profiling requires XTX 7900 mode.".to_string());
    }
    // Never reuse an older listener merely because its port/mode looks compatible.
    // Startup/cache/prefill behavior changes across releases, so an old server would
    // silently defeat the demo's current low-latency path. Builds predating the
    // health version field return None and are restarted as well.
    if engine_check() && server_version().as_deref() != Some(env!("CARGO_PKG_VERSION")) {
        stop_existing_voxgen_server(state)?;
    }

    // Do not reuse an older XTX listener merely because its mode string matches.
    // v0.7.26 enabled synchronous GPU timing and cooperative matrices by default;
    // later stream-safe releases must restart it to obtain the live tuning flags.
    if !benchmark_profile
        && engine_mode == "xtx7900"
        && engine_check()
        && server_execution_mode().as_deref() == Some("xtx7900")
        && server_xtx_stream_safe() != Some(true)
    {
        stop_existing_voxgen_server(state)?;
    }
    match health_state() {
        HealthState::Ready => {
            if stream_enabled && server_streaming_enabled() == Some(false) {
                return Err("A VoxGen server is already running with streaming disabled. Restart it with --stream on, or stop it so the demo can launch its own streaming server.".to_string());
            }
            if server_execution_mode().as_deref() != Some(engine_mode) {
                return Err(format!("A VoxGen server is already running in a different execution mode. Requested {engine_mode}; stop/restart the server or let the demo relaunch it."));
            }
            if benchmark_profile && server_benchmark_profile() != Some(true) {
                return Err("The running XTX server is stream-safe rather than offline profiling mode; restart it for profiling.".to_string());
            }
            if !benchmark_profile && engine_mode == "xtx7900" && server_xtx_stream_safe() != Some(true) {
                return Err("The running XTX server is not using the stream-safe tuning profile; restart it before live playback.".to_string());
            }
            return Ok("Connected to VoxGen; a speech model is already loaded.".to_string());
        }
        HealthState::EngineOnly => {
            if stream_enabled && server_streaming_enabled() == Some(false) {
                return Err("A VoxGen server is already running with streaming disabled. Restart it with --stream on, or stop it so the demo can launch its own streaming server.".to_string());
            }
            if server_execution_mode().as_deref() != Some(engine_mode) {
                return Err(format!("A VoxGen server is already running in a different execution mode. Requested {engine_mode}; stop/restart the server or let the demo relaunch it."));
            }
            if benchmark_profile && server_benchmark_profile() != Some(true) {
                return Err("The running XTX server is stream-safe rather than offline profiling mode; restart it for profiling.".to_string());
            }
            if !benchmark_profile && engine_mode == "xtx7900" && server_xtx_stream_safe() != Some(true) {
                return Err("The running XTX server is not using the stream-safe tuning profile; restart it before live playback.".to_string());
            }
            return Ok("Connected to VoxGen; select/load model paths before speaking.".to_string());
        }
        HealthState::Unavailable => {}
    }

    let root = project_root();
    let exe = find_voxgen_binary(&root)?;
    let mut command = Command::new(&exe);
    command
        .arg("--server")
        .arg("--host")
        .arg(HOST)
        .arg("--port")
        .arg(PORT.to_string())
        .arg("--stream")
        .arg(if stream_enabled { "on" } else { "off" })
        .arg("--gain")
        .arg(format!("{default_gain:.2}"))
        .arg("--mode")
        .arg(engine_mode)
        // Real-time demo playback uses the stream-safe XTX profile. Per-submit
        // timestamp readback and cooperative matrices remain opt-in benchmarking
        // experiments because either can increase 160-ms patch jitter.
        .arg("--gpu-profile")
        .arg(if benchmark_profile { "on" } else { "off" })
        .arg("--xtx-coopmat")
        .arg("off");
    if benchmark_profile {
        command.arg("--benchmark-profile");
    }
    let child = command
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("failed to start {}: {e}", exe.display()))?;

    {
        let mut guard = state.lock().map_err(|_| "demo state poisoned".to_string())?;
        guard.child = Some(child);
        guard.owns_server = true;
    }

    for _ in 0..240 {
        if engine_check()
            && server_execution_mode().as_deref() == Some(engine_mode)
            && (!benchmark_profile || server_benchmark_profile() == Some(true))
            && (benchmark_profile || engine_mode != "xtx7900" || server_xtx_stream_safe() == Some(true))
        {
            return Ok(format!(
                "Started VoxGen model-lifecycle server at http://127.0.0.1:8091 in {engine_mode} mode{}",
                if benchmark_profile { " (offline GPU profile)" } else { "" }
            ));
        }
        {
            let mut guard = state.lock().map_err(|_| "demo state poisoned".to_string())?;
            if let Some(child) = guard.child.as_mut() {
                if let Ok(Some(status)) = child.try_wait() {
                    return Err(format!("VoxGen exited during startup with {status}"));
                }
            }
        }
        thread::sleep(Duration::from_millis(500));
    }
    Err("VoxGen did not start its HTTP server. See the console for diagnostics.".to_string())
}

fn ensure_server(state: &SharedState, stream_enabled: bool, default_gain: f32, engine_mode: &str) -> Result<String, String> {
    ensure_server_with_profile(state, stream_enabled, default_gain, engine_mode, false)
}

fn ensure_offline_profile_server(state: &SharedState, default_gain: f32) -> Result<String, String> {
    ensure_server_with_profile(state, false, default_gain, "xtx7900", true)
}

fn current_model_paths() -> Result<Option<(PathBuf, Option<PathBuf>)>, String> {
    let body = http_request("GET", "/v1/models/current", &[])?;
    let value: serde_json::Value = serde_json::from_slice(&body).map_err(|e| e.to_string())?;
    if !value.get("loaded").and_then(|v| v.as_bool()).unwrap_or(false) {
        return Ok(None);
    }
    let base = value
        .get("base_lm")
        .and_then(|v| v.as_str())
        .map(PathBuf::from)
        .ok_or_else(|| "VoxGen reports a loaded model but no base_lm path".to_string())?;
    let acoustic = value
        .get("acoustic")
        .and_then(|v| v.as_str())
        .map(PathBuf::from);
    Ok(Some((base, acoustic)))
}

fn reset_server_gpu_profile() -> Result<(), String> {
    let body = http_request("POST", "/v1/profile/gpu/reset", b"{}")?;
    let value: serde_json::Value = serde_json::from_slice(&body).map_err(|e| format!("parse GPU profile reset response: {e}"))?;
    if value.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        return Err(format!("VoxGen rejected GPU profile reset: {}", String::from_utf8_lossy(&body)));
    }
    if value.get("enabled").and_then(|v| v.as_bool()) != Some(true) {
        return Err("VoxGen GPU profiling is not enabled on the offline profile server.".to_string());
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct GpuProfileRow {
    name: String,
    total_ms: f64,
    calls: u64,
    avg_ms: f64,
}

fn fetch_server_gpu_profile() -> Result<Vec<GpuProfileRow>, String> {
    let body = http_request("GET", "/v1/profile/gpu", &[])?;
    let value: serde_json::Value = serde_json::from_slice(&body).map_err(|e| format!("parse GPU profile response: {e}"))?;
    let profile = value.get("profile").ok_or_else(|| "GPU profile response has no profile object".to_string())?;
    if profile.get("enabled").and_then(|v| v.as_bool()) != Some(true) {
        return Err("VoxGen GPU profile is disabled; launch the offline profile server first.".to_string());
    }
    let timings = profile.get("timings").and_then(|v| v.as_object()).ok_or_else(|| "GPU profile response has no timings map".to_string())?;
    let mut rows = Vec::new();
    for (name, stat) in timings {
        let total_ms = stat.get("total_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let calls = stat.get("calls").and_then(|v| v.as_u64()).unwrap_or(0);
        let avg_ms = stat.get("avg_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
        if total_ms.is_finite() && total_ms >= 0.0 {
            rows.push(GpuProfileRow { name: name.clone(), total_ms, calls, avg_ms });
        }
    }
    rows.sort_by(|a, b| b.total_ms.partial_cmp(&a.total_ms).unwrap_or(std::cmp::Ordering::Equal));
    Ok(rows)
}

fn load_models(base: &Path, acoustic: &Path) -> Result<String, String> {
    if !base.is_file() {
        return Err(format!("BaseLM model does not exist: {}", base.display()));
    }
    if !acoustic.is_file() {
        return Err(format!("Acoustic model does not exist: {}", acoustic.display()));
    }
    let request = json!({
        "base_lm": base,
        "acoustic": acoustic,
        "base_format": "auto"
    });
    let body = serde_json::to_vec(&request).map_err(|e| e.to_string())?;
    let response = http_request("POST", "/v1/models/load", &body)?;
    let value: serde_json::Value = serde_json::from_slice(&response).map_err(|e| e.to_string())?;
    let ready = value
        .get("model")
        .and_then(|m| m.get("speech_inference_ready"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !ready {
        return Err("VoxGen loaded the selection but speech_inference_ready is false".to_string());
    }
    Ok(format!(
        "Loaded VoxCPM2 bundle: BaseLM={} | Acoustic={}",
        base.file_name().and_then(|s| s.to_str()).unwrap_or("BaseLM"),
        acoustic.file_name().and_then(|s| s.to_str()).unwrap_or("Acoustic")
    ))
}

fn warm_reference_audio(path: &Path) -> Result<String, String> {
    if !path.is_file() {
        return Err(format!("reference WAV does not exist: {}", path.display()));
    }
    let request = json!({"reference_audio_path": path});
    let body = serde_json::to_vec(&request).map_err(|e| e.to_string())?;
    let response = http_request("POST", "/v1/audio/conditioning/warm", &body)?;
    let value: serde_json::Value = serde_json::from_slice(&response).map_err(|e| e.to_string())?;
    if value.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        return Err(value.get("error").and_then(|v| v.as_str()).unwrap_or("conditioning warm-up failed").to_string());
    }
    let patches = value.get("latent_patches").and_then(|v| v.as_u64()).unwrap_or(0);
    let encode_ms = value.get("encode_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
    Ok(format!("Reference conditioning ready: {patches} latent patches (warm-up encode {encode_ms:.0} ms)."))
}

fn selected_models(state: &SharedState) -> Result<(PathBuf, PathBuf), String> {
    let guard = state.lock().map_err(|_| "demo state poisoned".to_string())?;
    let base = guard
        .base_model
        .clone()
        .ok_or_else(|| "Select a BaseLM GGUF first.".to_string())?;
    let acoustic = guard
        .acoustic_model
        .clone()
        .ok_or_else(|| "Select the Acoustic GGUF first.".to_string())?;
    Ok((base, acoustic))
}

fn ensure_models_ready(state: &SharedState, stream_enabled: bool, gain: f32, engine_mode: &str) -> Result<(), String> {
    ensure_server(state, stream_enabled, gain, engine_mode)?;
    if health_check() {
        return Ok(());
    }
    let (base, acoustic) = selected_models(state)?;
    load_models(&base, &acoustic)?;
    Ok(())
}

fn initialize_engine_and_models(state: &SharedState, stream_enabled: bool, gain: f32, engine_mode: &str) -> Result<(String, bool), String> {
    let server_message = ensure_server(state, stream_enabled, gain, engine_mode)?;
    if let Some((base, acoustic)) = current_model_paths()? {
        let mut guard = state.lock().map_err(|_| "demo state poisoned".to_string())?;
        guard.base_model = Some(base.clone());
        guard.acoustic_model = acoustic.clone();
        return Ok((
            format!(
                "{server_message}\nCurrent BaseLM component: {}\nCurrent Acoustic component: {}",
                base.display(),
                acoustic
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "(not loaded)".to_string())
            ),
            health_check(),
        ));
    }

    // Prefer paths persisted beside the demo executable before source-tree
    // discovery. This is what makes a portable demo folder self-contained.
    if let Ok((base, acoustic)) = selected_models(state) {
        if base.is_file() && acoustic.is_file() {
            let loaded = load_models(&base, &acoustic)?;
            return Ok((
                format!(
                    "{server_message}\nSaved BaseLM component: {}\nSaved Acoustic component: {}\n{loaded}",
                    base.display(), acoustic.display()
                ),
                true,
            ));
        }
    }

    let root = project_root();
    let base = find_base_model(&root).ok();
    let acoustic = find_acoustic_model(&root).ok();
    {
        let mut guard = state.lock().map_err(|_| "demo state poisoned".to_string())?;
        guard.base_model = base.clone();
        guard.acoustic_model = acoustic.clone();
    }

    match (base, acoustic) {
        (Some(base), Some(acoustic)) => {
            let loaded = load_models(&base, &acoustic)?;
            Ok((
                format!(
                    "{server_message}\nAuto-selected BaseLM component: {}\nAuto-selected Acoustic component: {}\n{loaded}",
                    base.display(), acoustic.display()
                ),
                true,
            ))
        }
        (base, acoustic) => Ok((
            format!(
                "{server_message}\nNo complete VoxCPM2 bundle was found. Select the BaseLM and Acoustic components, then click Load VoxCPM2.\nBaseLM component candidate: {}\nAcoustic component candidate: {}",
                base.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "(none)".to_string()),
                acoustic.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "(none)".to_string())
            ),
            false,
        )),
    }
}

fn validate_voice_wav(path: &Path) -> Result<(), String> {
    let bytes = fs::read(path)
        .map_err(|e| format!("read voice sample {}: {e}", path.display()))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("Voice sample must be a RIFF/WAVE file.".to_string());
    }
    Ok(())
}

#[derive(Debug)]
struct SpeechWav {
    wav: Vec<u8>,
    generated_patches: Option<usize>,
    stopped_by_predictor: Option<bool>,
    rtf: Option<f64>,
    engine_elapsed_ms: Option<f64>,
    first_pcm_ms: Option<f64>,
}

fn next_demo_seed() -> u64 {
    use std::sync::atomic::AtomicU64;
    static NEXT_SEED: AtomicU64 = AtomicU64::new(1);

    if let Ok(value) = env::var("VOXGEN_DEMO_SEED") {
        if let Ok(seed) = value.trim().parse::<u64>() {
            return seed;
        }
    }

    // Upstream VoxCPM2 does not pin normal generation to one fixed seed.  The
    // old demo inherited the server's seed 42 on every click, so a seed-specific
    // bad first patch (very noticeable on short openings such as "Oi!") was
    // reproduced forever.  Rotate the demo seed while retaining an env override
    // for reproducibility.
    let serial = NEXT_SEED.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    now ^ serial.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[derive(Debug, Clone)]
struct ExpressiveRequest {
    control: Option<String>,
    clone_mode: String,
    prompt_text: String,
    base_cfg_value: f32,
    cfg_value: f32,
    managed_gain_multiplier: f32,
    temperature: f32,
    inference_timesteps: u32,
}

impl ExpressiveRequest {
    fn effective_gain(&self, base_gain: f32) -> f32 {
        (base_gain * self.managed_gain_multiplier).clamp(0.0, MAX_GAIN_PERCENT as f32 / 100.0)
    }
}

fn build_demo_expressive_request(
    text: &str,
    preset: &str,
    intensity: &str,
    custom: &str,
    clone_mode: String,
    prompt_text: String,
    base_cfg_value: f32,
    temperature: f32,
    inference_timesteps: u32,
) -> ExpressiveRequest {
    let raw_control = if clone_mode == "ultimate" {
        None
    } else {
        build_style_control(preset, intensity, custom)
    };
    let tuning = managed_style_tuning(raw_control.as_deref());
    // Resolve the exact control text in the demo using the same engine-owned
    // compiler used by Runtime. Sending the resolved form makes the log and the
    // actual tokenized instruction identical; Runtime leaves the resolved custom
    // text untouched because the managed legacy marker is no longer present.
    let control = raw_control
        .as_deref()
        .map(|raw| refine_control_instruction(raw, text));
    let cfg_value = if clone_mode == "ultimate" {
        base_cfg_value
    } else {
        apply_managed_cfg(base_cfg_value, raw_control.as_deref())
    };
    ExpressiveRequest {
        control,
        clone_mode,
        prompt_text,
        base_cfg_value,
        cfg_value,
        managed_gain_multiplier: tuning.demo_gain_multiplier,
        temperature,
        inference_timesteps,
    }
}

fn speech_request_json(
    text: &str,
    voice_sample: Option<&Path>,
    gain: f32,
    seed: u64,
    expressive: &ExpressiveRequest,
) -> Result<serde_json::Value, String> {
    let effective_gain = expressive.effective_gain(gain);
    let mut request = json!({
        "input": text,
        "response_format": "wav",
        "seed": seed,
        "gain": effective_gain,
        "cfg_value": expressive.cfg_value,
        "temperature": expressive.temperature,
        "inference_timesteps": expressive.inference_timesteps,
        "clone_mode": expressive.clone_mode,
    });
    if let Some(control) = expressive.control.as_deref().map(str::trim).filter(|x| !x.is_empty()) {
        request["control"] = json!(control);
    }
    if let Some(path) = voice_sample {
        // The bundled demo always talks to localhost, so pass the stable local path.
        // This avoids per-click WAV read + base64 + JSON expansion + temporary-file
        // creation and lets the engine's AudioVAE conditioning cache actually hit.
        request["reference_audio_path"] = json!(path);
        if expressive.clone_mode == "ultimate" {
            let transcript = expressive.prompt_text.trim();
            if transcript.is_empty() {
                return Err("Ultimate cloning requires the exact transcript of the reference audio.".to_string());
            }
            request["prompt_audio_path"] = json!(path);
            request["prompt_text"] = json!(transcript);
            request["control"] = serde_json::Value::Null;
        }
    } else if expressive.clone_mode == "ultimate" {
        return Err("Ultimate cloning requires a voice/reference WAV.".to_string());
    }
    Ok(request)
}

fn speech_wav(
    text: &str,
    voice_sample: Option<&Path>,
    gain: f32,
    seed: u64,
    expressive: &ExpressiveRequest,
    request_id: Option<u64>,
) -> Result<SpeechWav, String> {
    let mut request = speech_request_json(text, voice_sample, gain, seed, expressive)?;
    if let Some(request_id) = request_id {
        request["request_id"] = json!(request_id);
    }
    let body = serde_json::to_vec(&request).map_err(|e| e.to_string())?;
    let response = http_request_full("POST", "/v1/audio/speech", &body)?;
    let generated_patches = response.header("X-VoxGen-Generated-Patches").and_then(|v| v.parse().ok());
    let stopped_by_predictor = response.header("X-VoxGen-Stopped-By-Predictor").and_then(|v| v.parse().ok());
    let rtf = response.header("X-VoxGen-RTF").and_then(|v| v.parse().ok());
    let engine_elapsed_ms = response.header("X-VoxGen-Elapsed-Ms").and_then(|v| v.parse().ok());
    let first_pcm_ms = response.header("X-VoxGen-First-PCM-Ms").and_then(|v| v.parse().ok());
    Ok(SpeechWav { wav: response.body, generated_patches, stopped_by_predictor, rtf, engine_elapsed_ms, first_pcm_ms })
}

fn pcm16_samples_from_f32(samples: &[f32], start_sample: usize) -> Result<Vec<i16>, String> {
    let mut out = Vec::with_capacity(samples.len());
    for (i, &value) in samples.iter().enumerate() {
        if !value.is_finite() {
            return Err("VoxGen returned a non-finite PCM sample".to_string());
        }
        let absolute = start_sample + i;
        let fade = if DEMO_ONSET_FADE_SAMPLES > 1 && absolute < DEMO_ONSET_FADE_SAMPLES {
            absolute as f32 / (DEMO_ONSET_FADE_SAMPLES - 1) as f32
        } else {
            1.0
        };
        let shaped = (value * fade).clamp(-1.0, 1.0);
        out.push((shaped * 32767.0).round() as i16);
    }
    Ok(out)
}

fn le_u16(buf: &[u8], off: usize) -> Result<u16, String> {
    let b = buf
        .get(off..off + 2)
        .ok_or_else(|| "truncated WAV".to_string())?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

fn le_u32(buf: &[u8], off: usize) -> Result<u32, String> {
    let b = buf
        .get(off..off + 4)
        .ok_or_else(|| "truncated WAV".to_string())?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn pcm16_wav_from_voxgen(
    wav: &[u8],
    word_spacing_ms: u32,
    live_controls: &LivePlaybackControls,
) -> Result<Vec<u8>, String> {
    if wav.len() < 12 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return Err("VoxGen response is not a RIFF/WAVE file".to_string());
    }

    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut data: Option<&[u8]> = None;
    let mut p = 12usize;
    while p + 8 <= wav.len() {
        let id = &wav[p..p + 4];
        let size = le_u32(wav, p + 4)? as usize;
        let start = p + 8;
        let end = start
            .checked_add(size)
            .ok_or_else(|| "WAV chunk size overflow".to_string())?;
        if end > wav.len() {
            return Err("truncated WAV chunk".to_string());
        }
        if id == b"fmt " && size >= 16 {
            fmt = Some((
                le_u16(wav, start)?,
                le_u16(wav, start + 2)?,
                le_u32(wav, start + 4)?,
                le_u16(wav, start + 14)?,
            ));
        } else if id == b"data" {
            data = Some(&wav[start..end]);
        }
        p = end + (size & 1);
    }

    let (format, channels, sample_rate, bits) =
        fmt.ok_or_else(|| "WAV has no fmt chunk".to_string())?;
    let data = data.ok_or_else(|| "WAV has no data chunk".to_string())?;
    if channels != 1 {
        return Err(format!("demo expects mono VoxGen audio, got {channels} channels"));
    }

    let floats = if format == 1 && bits == 16 && data.len() % 2 == 0 {
        data.chunks_exact(2)
            .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0)
            .collect::<Vec<_>>()
    } else if format == 3 && bits == 32 && data.len() % 4 == 0 {
        data.chunks_exact(4)
            .map(|sample| f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]))
            .collect::<Vec<_>>()
    } else {
        return Err(format!(
            "unsupported VoxGen WAV format: format={format}, bits={bits}"
        ));
    };

    let mut spacing = WordSpacingProcessor::new(word_spacing_ms, sample_rate);
    let mut paced = spacing.push(&floats);
    paced.extend(spacing.finish());

    let mut realtime = RealtimeVoiceProcessor::new();
    let mut rendered = realtime.push(&paced, live_controls);
    rendered.extend(realtime.finish(live_controls));
    let samples = pcm16_samples_from_f32(&rendered, 0)?;
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        pcm.extend_from_slice(&sample.to_le_bytes());
    }

    let data_bytes = u32::try_from(pcm.len()).map_err(|_| "audio is too large".to_string())?;
    let byte_rate = sample_rate
        .checked_mul(2)
        .ok_or_else(|| "WAV byte-rate overflow".to_string())?;
    let riff_size = data_bytes
        .checked_add(36)
        .ok_or_else(|| "WAV size overflow".to_string())?;
    let mut out = Vec::with_capacity(44 + pcm.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    out.extend_from_slice(&pcm);
    Ok(out)
}

#[cfg(windows)]
type HWaveOut = *mut c_void;

#[cfg(windows)]
#[repr(C)]
struct WaveFormatEx {
    format_tag: u16,
    channels: u16,
    samples_per_sec: u32,
    avg_bytes_per_sec: u32,
    block_align: u16,
    bits_per_sample: u16,
    cb_size: u16,
}

#[cfg(windows)]
#[repr(C)]
struct WaveHdr {
    data: *mut i8,
    buffer_length: u32,
    bytes_recorded: u32,
    user: usize,
    flags: u32,
    loops: u32,
    next: *mut WaveHdr,
    reserved: usize,
}

#[cfg(windows)]
#[link(name = "winmm")]
extern "system" {
    fn waveOutOpen(
        handle: *mut HWaveOut,
        device_id: u32,
        format: *const WaveFormatEx,
        callback: usize,
        instance: usize,
        flags: u32,
    ) -> u32;
    fn waveOutPrepareHeader(handle: HWaveOut, header: *mut WaveHdr, size: u32) -> u32;
    fn waveOutUnprepareHeader(handle: HWaveOut, header: *mut WaveHdr, size: u32) -> u32;
    fn waveOutWrite(handle: HWaveOut, header: *mut WaveHdr, size: u32) -> u32;
    fn waveOutReset(handle: HWaveOut) -> u32;
    fn waveOutClose(handle: HWaveOut) -> u32;
}

#[cfg(windows)]
const WAVE_MAPPER: u32 = u32::MAX;
#[cfg(windows)]
const WHDR_DONE: u32 = 0x0000_0001;

#[cfg(windows)]
struct WaveBlock {
    // Keep both allocations alive and stationary until WinMM marks the header done.
    samples: Vec<i16>,
    header: Box<WaveHdr>,
}

#[cfg(windows)]
struct WaveOutPlayer {
    handle: HWaveOut,
    blocks: VecDeque<WaveBlock>,
    submitted_samples: usize,
    cancel: SynthesisCancel,
}

#[cfg(windows)]
impl WaveOutPlayer {
    fn new(cancel: SynthesisCancel) -> Result<Self, String> {
        let format = WaveFormatEx {
            format_tag: 1, // WAVE_FORMAT_PCM
            channels: 1,
            samples_per_sec: 48_000,
            avg_bytes_per_sec: 96_000,
            block_align: 2,
            bits_per_sample: 16,
            cb_size: 0,
        };
        let mut handle: HWaveOut = std::ptr::null_mut();
        let code = unsafe {
            waveOutOpen(
                &mut handle,
                WAVE_MAPPER,
                &format,
                0,
                0,
                0,
            )
        };
        if code != 0 || handle.is_null() {
            return Err(format!("WinMM waveOutOpen failed with code {code}"));
        }
        cancel.register_waveout(handle);
        Ok(Self { handle, blocks: VecDeque::new(), submitted_samples: 0, cancel })
    }

    fn reap_done(&mut self) {
        let header_size = std::mem::size_of::<WaveHdr>() as u32;
        loop {
            let done = self.blocks.front().map(|block| unsafe {
                std::ptr::read_volatile(&block.header.flags) & WHDR_DONE != 0
            }).unwrap_or(false);
            if !done {
                break;
            }
            if let Some(mut block) = self.blocks.pop_front() {
                unsafe {
                    let _ = waveOutUnprepareHeader(self.handle, &mut *block.header, header_size);
                }
            }
        }
    }

    fn wait_for_live_capacity(&mut self) {
        // Do not let generation/rendering run arbitrarily far ahead of the audio
        // device. Otherwise lowering Speed can enqueue seconds of slow PCM, and a
        // later increase cannot affect those blocks because WinMM already owns them.
        while self.blocks.len() >= STREAM_MAX_PENDING_BLOCKS {
            if self.cancel.is_cancelled() {
                break;
            }
            self.reap_done();
            if self.blocks.len() >= STREAM_MAX_PENDING_BLOCKS {
                thread::sleep(Duration::from_millis(2));
            }
        }
    }

    fn queue_f32(&mut self, chunk: &[f32]) -> Result<(), String> {
        if self.cancel.is_cancelled() {
            return Ok(());
        }
        self.reap_done();
        let mut samples = pcm16_samples_from_f32(chunk, self.submitted_samples)?;
        self.submitted_samples = self.submitted_samples.saturating_add(samples.len());
        if samples.is_empty() {
            return Ok(());
        }
        let byte_len = samples.len().checked_mul(2)
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| "streaming PCM block is too large".to_string())?;
        let mut header = Box::new(WaveHdr {
            data: samples.as_mut_ptr() as *mut i8,
            buffer_length: byte_len,
            bytes_recorded: 0,
            user: 0,
            flags: 0,
            loops: 0,
            next: std::ptr::null_mut(),
            reserved: 0,
        });
        let header_size = std::mem::size_of::<WaveHdr>() as u32;
        let prep = unsafe { waveOutPrepareHeader(self.handle, &mut *header, header_size) };
        if prep != 0 {
            return Err(format!("WinMM waveOutPrepareHeader failed with code {prep}"));
        }
        // Serialize the actual queue operation against Stop's waveOutReset. This
        // closes the tiny race where Stop could reset the device just before a
        // synthesis thread enqueues one more block after cancellation.
        let active_guard = self.cancel.active_waveout.lock()
            .map_err(|_| "stream cancellation lock poisoned".to_string())?;
        if self.cancel.is_cancelled() {
            drop(active_guard);
            unsafe { let _ = waveOutUnprepareHeader(self.handle, &mut *header, header_size); }
            return Ok(());
        }
        let write = unsafe { waveOutWrite(self.handle, &mut *header, header_size) };
        drop(active_guard);
        if write != 0 {
            unsafe { let _ = waveOutUnprepareHeader(self.handle, &mut *header, header_size); }
            return Err(format!("WinMM waveOutWrite failed with code {write}"));
        }
        self.blocks.push_back(WaveBlock { samples, header });
        Ok(())
    }

    fn finish(mut self) {
        while !self.blocks.is_empty() {
            if self.cancel.is_cancelled() {
                return;
            }
            self.reap_done();
            if !self.blocks.is_empty() {
                thread::sleep(Duration::from_millis(2));
            }
        }
        if !self.handle.is_null() {
            self.cancel.unregister_waveout(self.handle);
            unsafe { let _ = waveOutClose(self.handle); }
            self.handle = std::ptr::null_mut();
        }
    }
}

#[cfg(windows)]
impl Drop for WaveOutPlayer {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        self.cancel.unregister_waveout(self.handle);
        unsafe { let _ = waveOutReset(self.handle); }
        let header_size = std::mem::size_of::<WaveHdr>() as u32;
        while let Some(mut block) = self.blocks.pop_front() {
            unsafe { let _ = waveOutUnprepareHeader(self.handle, &mut *block.header, header_size); }
        }
        unsafe { let _ = waveOutClose(self.handle); }
        self.handle = std::ptr::null_mut();
    }
}

#[cfg(windows)]
struct StreamSpeechResult {
    generated_patches: usize,
    rtf: f64,
    seed: u64,
    generation_seconds: f64,
    first_chunk_seconds: Option<f64>,
    time_to_first_audio_seconds: Option<f64>,
    initial_buffer_patches: usize,
    initial_buffer_ms: f64,
    average_patch_interval_ms: Option<f64>,
    max_patch_interval_ms: Option<f64>,
    late_patch_intervals: usize,
    patch_interval_count: usize,
    patch_deadline_ms: f64,
}

#[cfg(windows)]
fn speech_stream_windows(
    text: &str,
    voice_sample: Option<&Path>,
    word_spacing_ms: u32,
    gain: f32,
    seed: u64,
    expressive: &ExpressiveRequest,
    live_controls: &LivePlaybackControls,
    cancel: &SynthesisCancel,
    request_id: u64,
) -> Result<StreamSpeechResult, String> {
    if cancel.is_cancelled() {
        return Err("speech synthesis cancelled".to_string());
    }
    let mut request = speech_request_json(text, voice_sample, gain, seed, expressive)?;
    request["request_id"] = json!(request_id);
    request["streaming_prefix_len"] = json!(6);
    let body = serde_json::to_vec(&request).map_err(|e| e.to_string())?;
    let mut player = WaveOutPlayer::new(cancel.clone())?;
    let mut spacing = WordSpacingProcessor::new(word_spacing_ms, 48_000);
    let mut realtime = RealtimeVoiceProcessor::new();
    let started = std::time::Instant::now();
    let mut generated_patches = 0usize;
    let mut header_left = 44usize;
    let mut remainder = Vec::<u8>::new();
    let mut first_chunk_seconds = None;
    let mut playback_start_seconds = None;
    let mut playback_started = false;
    let mut prebuffer_target = STREAM_PREBUFFER_MIN_PATCHES;
    let mut prebuffer_samples = Vec::<f32>::new();
    let mut initial_buffer_patches = 0usize;
    let mut previous_patch_at: Option<std::time::Instant> = None;
    let mut patch_interval_total_ms = 0.0f64;
    let mut patch_interval_count = 0usize;
    let mut max_patch_interval_ms = 0.0f64;
    let mut late_patch_intervals = 0usize;
    // One acoustic patch represents 160 ms at 100% playback speed. Faster
    // playback consumes the queue sooner, so expose that tighter nominal
    // deadline in the benchmark block. Word-spacing can add some extra
    // playback time, therefore this remains a conservative diagnostic.
    let initial_speed = live_controls.speed_percent().max(1) as f64;
    let patch_deadline_ms = 160.0 * 100.0 / initial_speed;
    if initial_speed > 100.0 {
        prebuffer_target = prebuffer_target.max(2);
    }
    if initial_speed >= 110.0 {
        prebuffer_target = prebuffer_target.max(3);
    }

    let mut stream = connect()?;
    let request_head = format!(
        "POST /v1/audio/speech/stream HTTP/1.1\r\nHost: {HOST}:{PORT}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(request_head.as_bytes())
        .and_then(|_| stream.write_all(&body))
        .and_then(|_| stream.flush())
        .map_err(|e| format!("write streaming HTTP request: {e}"))?;
    let _ = stream.shutdown(Shutdown::Write);
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| format!("read streaming status: {e}"))?;
    if !line.contains(" 200 ") {
        let status = line.trim().to_string();
        let mut payload = String::new();
        let _ = reader.read_to_string(&mut payload);
        if status.is_empty() {
            return Err("VoxGen closed the streaming connection without an HTTP response. Check the engine log for the request error.".to_string());
        }
        return Err(format!("VoxGen streaming request failed ({status}): {payload}"));
    }
    let mut chunked = false;
    loop {
        line.clear();
        let n = reader.read_line(&mut line).map_err(|e| format!("read streaming headers: {e}"))?;
        if n == 0 {
            return Err("VoxGen closed before streaming headers completed".to_string());
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("transfer-encoding") && value.to_ascii_lowercase().contains("chunked") {
                chunked = true;
            }
        }
    }
    if !chunked {
        return Err("VoxGen streaming response is not chunked".to_string());
    }

    loop {
        if cancel.is_cancelled() {
            return Err("speech synthesis cancelled".to_string());
        }
        line.clear();
        let n = reader.read_line(&mut line).map_err(|e| {
            if cancel.is_cancelled() { "speech synthesis cancelled".to_string() } else { format!("read chunk size: {e}") }
        })?;
        if n == 0 {
            if cancel.is_cancelled() {
                return Err("speech synthesis cancelled".to_string());
            }
            return Err("VoxGen streaming response ended before the terminal chunk".to_string());
        }
        let size_text = line.trim().split(';').next().unwrap_or("");
        let size = usize::from_str_radix(size_text, 16)
            .map_err(|e| format!("invalid HTTP chunk size {size_text:?}: {e}"))?;
        if size == 0 {
            // Consume optional trailers.
            loop {
                line.clear();
                if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 || line == "\r\n" || line == "\n" {
                    break;
                }
            }
            break;
        }
        let mut chunk = vec![0u8; size];
        reader.read_exact(&mut chunk).map_err(|e| {
            if cancel.is_cancelled() { "speech synthesis cancelled".to_string() } else { format!("read streaming audio chunk: {e}") }
        })?;
        let mut crlf = [0u8; 2];
        reader.read_exact(&mut crlf).map_err(|e| format!("read chunk terminator: {e}"))?;
        if crlf != *b"\r\n" {
            return Err("malformed chunk terminator from VoxGen".to_string());
        }

        let mut payload = chunk.as_slice();
        if header_left > 0 {
            let skip = header_left.min(payload.len());
            header_left -= skip;
            payload = &payload[skip..];
        }
        if payload.is_empty() {
            continue;
        }
        if cancel.is_cancelled() {
            return Err("speech synthesis cancelled".to_string());
        }
        remainder.extend_from_slice(payload);
        let complete = remainder.len() / 4 * 4;
        if complete == 0 {
            continue;
        }
        let floats = remainder[..complete]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect::<Vec<_>>();
        remainder.drain(..complete);
        let paced = spacing.push(&floats);
        if playback_started {
            player.wait_for_live_capacity();
        }
        if cancel.is_cancelled() {
            return Err("speech synthesis cancelled".to_string());
        }
        let rendered = realtime.push(&paced, live_controls);
        let arrived = std::time::Instant::now();
        if first_chunk_seconds.is_none() {
            first_chunk_seconds = Some(started.elapsed().as_secs_f64());
        }
        if let Some(previous) = previous_patch_at {
            let interval_ms = arrived.duration_since(previous).as_secs_f64() * 1000.0;
            patch_interval_total_ms += interval_ms;
            patch_interval_count += 1;
            max_patch_interval_ms = max_patch_interval_ms.max(interval_ms);
            if interval_ms > patch_deadline_ms {
                late_patch_intervals += 1;
            }
            // Before playback starts, use the observed early cadence to size the
            // reserve. A near-deadline second/third patch gets an extra patch of
            // cushion; an already-late interval gets the maximum startup reserve.
            if !playback_started {
                if interval_ms > patch_deadline_ms {
                    prebuffer_target = STREAM_PREBUFFER_MAX_PATCHES;
                } else if interval_ms > patch_deadline_ms * STREAM_PREBUFFER_RISK_RATIO {
                    prebuffer_target = prebuffer_target.max(3);
                }
            }
        }
        previous_patch_at = Some(arrived);
        generated_patches += 1;
        if playback_started {
            player.queue_f32(&rendered)?;
        } else {
            prebuffer_samples.extend_from_slice(&rendered);
            if generated_patches >= prebuffer_target && !prebuffer_samples.is_empty() {
                initial_buffer_patches = generated_patches;
                player.queue_f32(&prebuffer_samples)?;
                prebuffer_samples.clear();
                playback_started = true;
                playback_start_seconds = Some(started.elapsed().as_secs_f64());
            }
        }
    }
    if cancel.is_cancelled() {
        return Err("speech synthesis cancelled".to_string());
    }
    let pacing_tail = spacing.finish();
    let rendered_tail = realtime.push(&pacing_tail, live_controls);
    if playback_started {
        player.queue_f32(&rendered_tail)?;
    } else {
        prebuffer_samples.extend_from_slice(&rendered_tail);
    }
    let dsp_tail = realtime.finish(live_controls);
    if playback_started {
        player.queue_f32(&dsp_tail)?;
    } else {
        prebuffer_samples.extend_from_slice(&dsp_tail);
        if !prebuffer_samples.is_empty() {
            initial_buffer_patches = generated_patches;
            player.queue_f32(&prebuffer_samples)?;
            prebuffer_samples.clear();
            playback_started = true;
            playback_start_seconds = Some(started.elapsed().as_secs_f64());
        }
    }
    if header_left != 0 || !remainder.is_empty() {
        return Err("truncated f32 WAV stream from VoxGen".to_string());
    }
    let audio_seconds = generated_patches as f64 * 0.160;
    let generation_seconds = started.elapsed().as_secs_f64();
    let rtf = if audio_seconds > 0.0 {
        generation_seconds / audio_seconds
    } else {
        f64::INFINITY
    };
    let average_patch_interval_ms = (patch_interval_count > 0)
        .then_some(patch_interval_total_ms / patch_interval_count as f64);
    let max_patch_interval_ms = (patch_interval_count > 0).then_some(max_patch_interval_ms);
    player.finish();
    Ok(StreamSpeechResult {
        generated_patches,
        rtf,
        seed,
        generation_seconds,
        first_chunk_seconds,
        time_to_first_audio_seconds: playback_start_seconds,
        initial_buffer_patches,
        initial_buffer_ms: initial_buffer_patches as f64 * patch_deadline_ms,
        average_patch_interval_ms,
        max_patch_interval_ms,
        late_patch_intervals,
        patch_interval_count,
        patch_deadline_ms,
    })
}

#[derive(Debug, Clone)]
struct VariationSummary {
    index: u32,
    seed: u64,
    generated_patches: Option<usize>,
    stopped_by_predictor: Option<bool>,
    rtf: Option<f64>,
    generation_seconds: Option<f64>,
    first_chunk_seconds: Option<f64>,
    time_to_first_audio_seconds: Option<f64>,
    initial_buffer_patches: Option<usize>,
    initial_buffer_ms: Option<f64>,
    average_patch_interval_ms: Option<f64>,
    max_patch_interval_ms: Option<f64>,
    late_patch_intervals: Option<usize>,
    patch_interval_count: Option<usize>,
    patch_deadline_ms: Option<f64>,
}

fn append_benchmark_results(
    log: TextCtrl,
    engine_mode: &str,
    stream_enabled: bool,
    text: &str,
    variations: u32,
    summaries: &[VariationSummary],
) {
    let mode_label = if engine_mode == "xtx7900" { "XTX 7900" } else { "Normal" };
    let chars = text.chars().count();
    let words = text.split_whitespace().count();
    for summary in summaries {
        append_log(log, "--- Benchmark results ---");
        append_log(log, &format!("Mode: {mode_label}"));
        append_log(log, &format!("Streaming: {}", if stream_enabled { "on" } else { "off" }));
        append_log(log, &format!("Input: {chars} characters, {words} words"));
        append_log(log, &format!("Variation: {}/{}", summary.index, variations));
        if let Some(patches) = summary.generated_patches {
            append_log(log, &format!("Acoustic patches: {patches}"));
            append_log(log, &format!("Generated audio: {:.2} s", patches as f64 * 0.160));
        }
        if let Some(seconds) = summary.generation_seconds {
            append_log(log, &format!("Generation wall time: {seconds:.3} s"));
        }
        if let Some(first_chunk) = summary.first_chunk_seconds {
            append_log(log, &format!("First PCM ready: {first_chunk:.3} s"));
        }
        if let Some(ttfa) = summary.time_to_first_audio_seconds {
            append_log(log, &format!("Time to playback start: {ttfa:.3} s"));
        } else if stream_enabled {
            append_log(log, "Time to playback start: unavailable");
        }
        if let (Some(patches), Some(ms)) = (summary.initial_buffer_patches, summary.initial_buffer_ms) {
            append_log(log, &format!("Adaptive startup buffer: {patches} patches (~{ms:.0} ms at current speed)"));
        }
        if let Some(rtf) = summary.rtf {
            append_log(log, &format!("RTF: {rtf:.3}"));
            let headroom = (1.0 - rtf) * 100.0;
            if stream_enabled {
                append_log(log, &format!(
                    "End-to-end throughput headroom: {headroom:+.1}% (includes startup latency)"
                ));
            } else {
                let status = if rtf <= 1.0 { "PASS" } else { "SLOWER THAN REAL TIME" };
                append_log(log, &format!("Throughput headroom: {headroom:+.1}% ({status})"));
            }
        }
        if let (Some(avg), Some(max), Some(late), Some(count), Some(deadline)) = (
            summary.average_patch_interval_ms,
            summary.max_patch_interval_ms,
            summary.late_patch_intervals,
            summary.patch_interval_count,
            summary.patch_deadline_ms,
        ) {
            append_log(log, &format!(
                "Patch delivery: avg {avg:.1} ms, max {max:.1} ms, late >{deadline:.1} ms: {late}/{count}"
            ));
            if deadline > 0.0 {
                let avg_headroom = (1.0 - avg / deadline) * 100.0;
                let worst_headroom = (1.0 - max / deadline) * 100.0;
                let cadence_status = if late == 0 { "PASS" } else { "WARN - late patches detected" };
                append_log(log, &format!(
                    "Streaming cadence headroom: avg {avg_headroom:+.1}%, worst {worst_headroom:+.1}% ({cadence_status})"
                ));
            }
        }
        append_log(log, &format!("Seed: {}", summary.seed));
        if engine_mode == "xtx7900" {
            append_log(log, "XTX tuning: shared QKV + targeted barriers + residual-rms/swiglu + wave32 + subgroup reductions + x4 linear + prefill32-live; GPU profile off; coopmat off");
        } else {
            append_log(log, "Engine path: generic Vulkan + shared QKV/targeted barriers + residual-rms/swiglu fusions");
        }
        append_log(log, "-------------------------");
    }
}

enum SynthesisPlayback {
    #[cfg(windows)]
    Streamed {
        cloned: bool,
        summaries: Vec<VariationSummary>,
    },
    File {
        path: PathBuf,
        cloned: bool,
        summaries: Vec<VariationSummary>,
    },
}
fn playback_path() -> PathBuf {
    use std::sync::atomic::AtomicU64;
    static NEXT_PLAYBACK: AtomicU64 = AtomicU64::new(1);
    let serial = NEXT_PLAYBACK.fetch_add(1, Ordering::Relaxed);
    env::temp_dir().join(format!(
        "voxgen-demo-{}-{serial}.wav",
        std::process::id()
    ))
}


fn combine_pcm16_wavs(wavs: &[Vec<u8>], gap_ms: u32) -> Result<Vec<u8>, String> {
    if wavs.is_empty() { return Err("no candidate WAVs to combine".to_string()); }
    if wavs.len() == 1 { return Ok(wavs[0].clone()); }
    let mut pcm = Vec::new();
    let gap_bytes = (48_000usize * gap_ms as usize / 1000) * 2;
    for (i, wav) in wavs.iter().enumerate() {
        if wav.len() < 44 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" || &wav[36..40] != b"data" {
            return Err("candidate playback WAV is not the demo's canonical PCM16 format".to_string());
        }
        let declared = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]) as usize;
        if wav.len() < 44 + declared { return Err("candidate playback WAV is truncated".to_string()); }
        if i > 0 { pcm.resize(pcm.len() + gap_bytes, 0); }
        pcm.extend_from_slice(&wav[44..44 + declared]);
    }
    let data_bytes = u32::try_from(pcm.len()).map_err(|_| "combined candidate audio is too large".to_string())?;
    let mut out = wavs[0][..44].to_vec();
    out[4..8].copy_from_slice(&data_bytes.saturating_add(36).to_le_bytes());
    out[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    out.extend_from_slice(&pcm);
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferenceSource {
    PresetSpecific,
    NeutralAnchor,
    LegacyVoiceSample,
    RuntimeVoiceSample,
    None,
}

#[derive(Debug, Clone)]
struct ResolvedReference {
    path: Option<PathBuf>,
    source: ReferenceSource,
}

impl ResolvedReference {
    fn describe(&self, preset: &str) -> Option<String> {
        let path = self.path.as_deref()?;
        Some(match self.source {
            ReferenceSource::PresetSpecific => {
                format!("Using preset-specific [{preset}] reference: {}", path.display())
            }
            ReferenceSource::NeutralAnchor if preset == "neutral" => {
                format!("Using neutral voice anchor: {}", path.display())
            }
            ReferenceSource::NeutralAnchor => {
                format!("No usable [{preset}] voice clip; anchoring identity to neutral sample: {}", path.display())
            }
            ReferenceSource::LegacyVoiceSample | ReferenceSource::RuntimeVoiceSample => {
                format!("No usable [{preset}] preset reference; using configured voice anchor: {}", path.display())
            }
            ReferenceSource::None => return None,
        })
    }
}

/// Resolve the reference audio for a style without silently abandoning a
/// configured speaker identity.  A dedicated preset clip wins when present,
/// otherwise the explicit Neutral preset is the canonical identity anchor.
/// Older `voice_sample=` settings remain supported after that.
///
/// If the user configured a Neutral anchor but the file is no longer present,
/// return an error instead of falling through to VoxCPM2 zero-shot generation.
/// That prevents a missing/moved WAV from unexpectedly producing a new voice.
fn resolve_reference_sample(
    cfg: &DemoSettings,
    preset: &str,
    runtime_voice_sample: Option<PathBuf>,
) -> Result<ResolvedReference, String> {
    let configured_preset = cfg.emotion_references.get(preset).cloned();
    if let Some(path) = configured_preset.as_ref() {
        if path.is_file() {
            return Ok(ResolvedReference {
                path: Some(path.clone()),
                source: if preset == "neutral" { ReferenceSource::NeutralAnchor } else { ReferenceSource::PresetSpecific },
            });
        }
    }

    // Neutral is authoritative once configured. If it was moved/deleted, do
    // not silently substitute a different voice or fall through to zero-shot.
    if let Some(path) = cfg.emotion_references.get("neutral") {
        if path.is_file() {
            return Ok(ResolvedReference {
                path: Some(path.clone()),
                source: ReferenceSource::NeutralAnchor,
            });
        }
        return Err(format!(
            "Neutral voice anchor is configured but unavailable: {}. Refusing zero-shot voice generation; restore or reselect the neutral WAV.",
            path.display()
        ));
    }

    // Compatibility with pre-v0.7.36 settings, where the neutral/default
    // reference lived only in voice_sample=.
    if let Some(path) = cfg.voice_sample.as_ref() {
        if path.is_file() {
            return Ok(ResolvedReference {
                path: Some(path.clone()),
                source: ReferenceSource::LegacyVoiceSample,
            });
        }
        return Err(format!(
            "Configured voice anchor is unavailable: {}. Refusing zero-shot voice generation; restore or reselect the WAV.",
            path.display()
        ));
    }

    if let Some(path) = runtime_voice_sample {
        if path.is_file() {
            return Ok(ResolvedReference {
                path: Some(path),
                source: ReferenceSource::RuntimeVoiceSample,
            });
        }
    }

    if let Some(path) = configured_preset {
        return Err(format!(
            "Preset reference [{preset}] is configured but unavailable: {}. No neutral voice anchor is available, so VoxGen will not invent a replacement voice.",
            path.display()
        ));
    }

    Ok(ResolvedReference { path: None, source: ReferenceSource::None })
}

fn effective_expressive_for_sample(
    mut expressive: ExpressiveRequest,
    sample: Option<&Path>,
) -> Result<ExpressiveRequest, String> {
    if sample.is_none() && expressive.clone_mode == "reference" {
        expressive.clone_mode = "auto".to_string();
    }
    if sample.is_none() && expressive.clone_mode == "ultimate" {
        return Err("Ultimate cloning requires a voice/reference WAV.".to_string());
    }
    Ok(expressive)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn short_file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_else(|| path.to_str().unwrap_or("(unknown)"))
        .to_string()
}

#[derive(Debug, Clone)]
struct OfflineBenchCase {
    mode: &'static str,
    seed: u64,
    patches: Option<usize>,
    audio_seconds: Option<f64>,
    wall_seconds: f64,
    engine_seconds: Option<f64>,
    first_pcm_seconds: Option<f64>,
    rtf: Option<f64>,
}

fn run_offline_benchmark_case(
    state: &SharedState,
    base: &Path,
    acoustic: &Path,
    gain: f32,
    mode: &'static str,
    text: &str,
    sample: Option<&Path>,
    expressive: &ExpressiveRequest,
    seed: u64,
) -> Result<OfflineBenchCase, String> {
    stop_existing_voxgen_server(state)?;
    // A/B runs are intentionally offline at the server level as well as silent
    // in the demo. This keeps endpoint availability/playback buffering out of
    // the comparison while preserving the same stream-safe XTX kernel profile.
    ensure_server(state, false, gain, mode)?;
    load_models(base, acoustic)?;
    let started = std::time::Instant::now();
    let speech = speech_wav(text, sample, gain, seed, expressive, None)?;
    let wall_seconds = started.elapsed().as_secs_f64();
    let audio_seconds = speech.generated_patches.map(|p| p as f64 * 0.160);
    Ok(OfflineBenchCase {
        mode,
        seed,
        patches: speech.generated_patches,
        audio_seconds,
        wall_seconds,
        engine_seconds: speech.engine_elapsed_ms.map(|ms| ms / 1000.0),
        first_pcm_seconds: speech.first_pcm_ms.map(|ms| ms / 1000.0),
        rtf: speech.rtf,
    })
}

fn restore_selected_server(
    state: &SharedState,
    base: &Path,
    acoustic: &Path,
    stream_enabled: bool,
    gain: f32,
    selected_mode: &str,
) -> Result<(), String> {
    stop_existing_voxgen_server(state)?;
    ensure_server(state, stream_enabled, gain, selected_mode)?;
    load_models(base, acoustic)?;
    Ok(())
}

fn lower_is_better_improvement(normal: Option<f64>, xtx: Option<f64>) -> Option<f64> {
    let normal = normal?;
    let xtx = xtx?;
    if !normal.is_finite() || !xtx.is_finite() || normal <= 0.0 { return None; }
    Some((normal - xtx) / normal * 100.0)
}

fn append_ab_results(
    log: TextCtrl,
    normal: &OfflineBenchCase,
    xtx: &OfflineBenchCase,
    text: &str,
    base: &Path,
    acoustic: &Path,
    sample: Option<&Path>,
    expressive: &ExpressiveRequest,
) {
    append_log(log, "--- Normal vs XTX 7900 A/B benchmark ---");
    append_log(log, &format!("Seed: {} (identical for both modes)", normal.seed));
    append_log(log, &format!("Text hash: fnv1a64:{:016x}", fnv1a64(text.as_bytes())));
    append_log(log, &format!("BaseLM: {}", short_file_name(base)));
    append_log(log, &format!("Acoustic: {}", short_file_name(acoustic)));
    append_log(log, &format!(
        "Reference: {}",
        sample.map(|p| p.display().to_string()).unwrap_or_else(|| "none (zero-shot)".to_string())
    ));
    append_log(log, &format!(
        "Inference settings: clone={}, CFG {:.2} (base {:.2}), temperature {:.2}, CFM steps {}",
        expressive.clone_mode, expressive.cfg_value, expressive.base_cfg_value, expressive.temperature, expressive.inference_timesteps
    ));
    for case in [normal, xtx] {
        let label = if case.mode == "xtx7900" { "XTX 7900" } else { "Normal" };
        let patches = case.patches.map(|p| p.to_string()).unwrap_or_else(|| "?".to_string());
        let audio = case.audio_seconds.map(|v| format!("{v:.2}s")).unwrap_or_else(|| "?".to_string());
        let rtf = case.rtf.map(|v| format!("{v:.3}")).unwrap_or_else(|| "?".to_string());
        let first = case.first_pcm_seconds.map(|v| format!("{v:.3}s")).unwrap_or_else(|| "?".to_string());
        let engine = case.engine_seconds.map(|v| format!("{v:.3}s")).unwrap_or_else(|| "?".to_string());
        append_log(log, &format!(
            "{label}: wall {:.3}s, engine {engine}, first PCM {first}, RTF {rtf}, patches {patches}, audio {audio}",
            case.wall_seconds
        ));
    }
    if let Some(v) = lower_is_better_improvement(Some(normal.wall_seconds), Some(xtx.wall_seconds)) {
        append_log(log, &format!("XTX wall-time improvement: {v:+.1}%"));
    }
    if let Some(v) = lower_is_better_improvement(normal.engine_seconds, xtx.engine_seconds) {
        append_log(log, &format!("XTX engine-time improvement: {v:+.1}%"));
    }
    if let Some(v) = lower_is_better_improvement(normal.first_pcm_seconds, xtx.first_pcm_seconds) {
        append_log(log, &format!("XTX first-PCM improvement: {v:+.1}%"));
    }
    if let Some(v) = lower_is_better_improvement(normal.rtf, xtx.rtf) {
        append_log(log, &format!("XTX RTF improvement: {v:+.1}%"));
    }
    if normal.patches != xtx.patches {
        append_log(log, &format!(
            "Note: stop prediction diverged despite the identical seed (Normal {:?} vs XTX {:?} patches); RTF remains normalized by generated audio length.",
            normal.patches, xtx.patches
        ));
    } else {
        append_log(log, "Comparability: same seed, same conditioning/settings, same generated patch count.");
    }
    append_log(log, "------------------------------------------");
}

fn append_gpu_profile_results(
    log: TextCtrl,
    rows: &[GpuProfileRow],
    text: &str,
    base: &Path,
    acoustic: &Path,
    sample: Option<&Path>,
    expressive: &ExpressiveRequest,
    seed: u64,
    speech: &SpeechWav,
    wall_seconds: f64,
) {
    append_log(log, "--- XTX 7900 offline GPU profile ---");
    append_log(log, &format!("Seed: {seed}"));
    append_log(log, &format!("Text hash: fnv1a64:{:016x}", fnv1a64(text.as_bytes())));
    append_log(log, &format!("BaseLM: {}", short_file_name(base)));
    append_log(log, &format!("Acoustic: {}", short_file_name(acoustic)));
    append_log(log, &format!(
        "Reference: {}",
        sample.map(|p| p.display().to_string()).unwrap_or_else(|| "none (zero-shot)".to_string())
    ));
    append_log(log, &format!(
        "Inference settings: clone={}, CFG {:.2} (base {:.2}), temperature {:.2}, CFM steps {}",
        expressive.clone_mode, expressive.cfg_value, expressive.base_cfg_value, expressive.temperature, expressive.inference_timesteps
    ));
    append_log(log, &format!("Profiled wall time: {wall_seconds:.3} s (offline; timestamp readback intentionally enabled)"));
    if let Some(rtf) = speech.rtf { append_log(log, &format!("Profiled RTF: {rtf:.3} (do not compare directly with stream-safe RTF)")); }
    let total_gpu_ms: f64 = rows.iter().map(|r| r.total_ms).sum();
    append_log(log, &format!("Summed measured kernel GPU time: {total_gpu_ms:.1} ms"));
    if rows.is_empty() {
        append_log(log, "No GPU timing rows were returned.");
    } else {
        append_log(log, "Hot kernels (descending total GPU time):");
        for row in rows.iter().take(16) {
            let share = if total_gpu_ms > 0.0 { row.total_ms / total_gpu_ms * 100.0 } else { 0.0 };
            append_log(log, &format!(
                "  {:<30} {:>9.2} ms  {:>6.1}%  calls {:>6}  avg {:>7.4} ms",
                row.name.as_str(), row.total_ms, share, row.calls, row.avg_ms
            ));
        }
    }
    append_log(log, "------------------------------------");
}

fn synthesize(
    text: String,
    state: SharedState,
    log: TextCtrl,
    input: TextCtrl,
    speak: Button,
    stop: Button,
    voice: Button,
    base_model: Button,
    acoustic_model: Button,
    load_models_button: Button,
    benchmark_ab_button: Button,
    profile_xtx_button: Button,
    word_spacing_control: SpinCtrl,
    word_spacing_ms: u32,
    gain_control: SpinCtrl,
    gain_percent: u32,
    live_controls: LivePlaybackControls,
    cancel: SynthesisCancel,
    stream_enabled: bool,
    engine_mode: String,
    sample_override: Option<PathBuf>,
    expressive: ExpressiveRequest,
    variations: u32,
) {
    let request_id = cancel.begin();
    speak.enable(false);
    stop.enable(true);
    voice.enable(false);
    base_model.enable(false);
    acoustic_model.enable(false);
    load_models_button.enable(false);
    benchmark_ab_button.enable(false);
    profile_xtx_button.enable(false);
    word_spacing_control.enable(false);
    gain_control.enable(false);
    input.enable(false);

    // The caller resolves preset -> neutral anchor -> legacy voice sample before
    // entering synthesis.  Do not perform another implicit fallback here: doing
    // so could bypass the neutral-anchor safety policy and silently enter a
    // different conditioning mode.  Zero-shot remains available only when no
    // reference/neutral anchor has ever been configured.
    let sample = sample_override;
    let expressive = match effective_expressive_for_sample(expressive, sample.as_deref()) {
        Ok(value) => value,
        Err(err) => {
            append_log(log, &format!("Error: {err}"));
            speak.enable(true);
            stop.enable(false);
            voice.enable(true);
            base_model.enable(true);
            acoustic_model.enable(true);
            load_models_button.enable(true);
            benchmark_ab_button.enable(true);
            profile_xtx_button.enable(true);
            word_spacing_control.enable(true);
            gain_control.enable(true);
            input.enable(true);
            return;
        }
    };

    append_log(log, &format!("You: {text}"));
    append_log(log, "VoxGen: synthesizing...");
    if sample.is_none() && expressive.clone_mode == "auto" {
        append_log(log, "No reference sample selected: using zero-shot generation.");
    }
    if let Some(control) = expressive.control.as_deref() {
        append_log(log, &format!("Style control (effective): {control}"));
    } else if expressive.clone_mode == "ultimate" {
        append_log(log, "Style control: inherited from Ultimate-cloning prompt audio.");
    } else {
        append_log(log, "Style control: automatic text prosody.");
    }
    append_log(log, &format!(
        "Generation: clone mode {}, CFG {:.2}, temperature {:.2}, CFM steps {}, variations {}.",
        expressive.clone_mode,
        expressive.cfg_value,
        expressive.temperature,
        expressive.inference_timesteps,
        variations,
    ));
    if (expressive.cfg_value - expressive.base_cfg_value).abs() > 0.0005 {
        append_log(log, &format!(
            "Managed style guidance: base CFG {:.2} -> effective {:.2} ({:+.2}).",
            expressive.base_cfg_value,
            expressive.cfg_value,
            expressive.cfg_value - expressive.base_cfg_value,
        ));
    }
    if word_spacing_ms > 0 {
        append_log(log, &format!("Pacing: extending detected short word gaps by +{word_spacing_ms} ms."));
    }
    let base_gain = gain_percent as f32 / 100.0;
    let effective_gain = expressive.effective_gain(base_gain);
    append_log(log, &format!(
        "Playback: stream {}, speed {}%, pitch {:+} semitones, gain {:.2}x{}.",
        if stream_enabled { "on" } else { "off" },
        live_controls.speed_percent(),
        live_controls.pitch_semitones(),
        effective_gain,
        if stream_enabled { " (speed/pitch adjustable while streaming)" } else { "" },
    ));
    if (expressive.managed_gain_multiplier - 1.0).abs() > 0.0005 {
        append_log(log, &format!(
            "Managed style level: base gain {:.2}x -> effective {:.2}x ({:+.0}%).",
            base_gain,
            effective_gain,
            (expressive.managed_gain_multiplier - 1.0) * 100.0,
        ));
    }

    let variations = variations.clamp(1, MAX_VARIATIONS);

    thread::spawn(move || {
        let result = (|| -> Result<SynthesisPlayback, String> {
            ensure_models_ready(&state, stream_enabled, gain_percent as f32 / 100.0, &engine_mode)?;
            if cancel.is_cancelled() {
                return Err("speech synthesis cancelled".to_string());
            }
            let base_seed = next_demo_seed();
            #[cfg(windows)]
            if stream_enabled {
                let mut summaries = Vec::new();
                for i in 0..variations {
                    if cancel.is_cancelled() {
                        return Err("speech synthesis cancelled".to_string());
                    }
                    let seed = if i == 0 { base_seed } else { base_seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) };
                    let streamed = speech_stream_windows(
                        &text,
                        sample.as_deref(),
                        word_spacing_ms,
                        gain_percent as f32 / 100.0,
                        seed,
                        &expressive,
                        &live_controls,
                        &cancel,
                        request_id,
                    )?;
                    summaries.push(VariationSummary {
                        index: i + 1,
                        seed: streamed.seed,
                        generated_patches: Some(streamed.generated_patches),
                        stopped_by_predictor: None,
                        rtf: Some(streamed.rtf),
                        generation_seconds: Some(streamed.generation_seconds),
                        first_chunk_seconds: streamed.first_chunk_seconds,
                        time_to_first_audio_seconds: streamed.time_to_first_audio_seconds,
                        initial_buffer_patches: Some(streamed.initial_buffer_patches),
                        initial_buffer_ms: Some(streamed.initial_buffer_ms),
                        average_patch_interval_ms: streamed.average_patch_interval_ms,
                        max_patch_interval_ms: streamed.max_patch_interval_ms,
                        late_patch_intervals: Some(streamed.late_patch_intervals),
                        patch_interval_count: Some(streamed.patch_interval_count),
                        patch_deadline_ms: Some(streamed.patch_deadline_ms),
                    });
                    if i + 1 < variations { thread::sleep(Duration::from_millis(250)); }
                }
                return Ok(SynthesisPlayback::Streamed { cloned: sample.is_some(), summaries });
            }

            let mut rendered_wavs = Vec::new();
            let mut summaries = Vec::new();
            for i in 0..variations {
                if cancel.is_cancelled() {
                    return Err("speech synthesis cancelled".to_string());
                }
                let seed = if i == 0 { base_seed } else { base_seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) };
                let request_started = std::time::Instant::now();
                let speech = speech_wav(&text, sample.as_deref(), gain_percent as f32 / 100.0, seed, &expressive, Some(request_id))?;
                let generation_seconds = request_started.elapsed().as_secs_f64();
                rendered_wavs.push(pcm16_wav_from_voxgen(&speech.wav, word_spacing_ms, &live_controls)?);
                summaries.push(VariationSummary {
                    index: i + 1,
                    seed,
                    generated_patches: speech.generated_patches,
                    stopped_by_predictor: speech.stopped_by_predictor,
                    rtf: speech.rtf,
                    generation_seconds: Some(generation_seconds),
                    first_chunk_seconds: speech.first_pcm_ms.map(|ms| ms / 1000.0),
                    time_to_first_audio_seconds: speech.first_pcm_ms.map(|ms| ms / 1000.0),
                    initial_buffer_patches: None,
                    initial_buffer_ms: None,
                    average_patch_interval_ms: None,
                    max_patch_interval_ms: None,
                    late_patch_intervals: None,
                    patch_interval_count: None,
                    patch_deadline_ms: None,
                });
            }
            let pcm16 = combine_pcm16_wavs(&rendered_wavs, 250)?;
            let path = playback_path();
            fs::write(&path, pcm16).map_err(|e| format!("write playback WAV {}: {e}", path.display()))?;
            Ok(SynthesisPlayback::File { path, cloned: sample.is_some(), summaries })
        })();

        wxdragon::call_after(Box::new(move || {
            if cancel.is_cancelled() {
                Sound::stop();
                append_log(log, "VoxGen: synthesis stopped.");
            } else {
            match result {
                #[cfg(windows)]
                Ok(SynthesisPlayback::Streamed { cloned, summaries }) => {
                    append_log(log, if cloned { "VoxGen: streamed cloned voice candidate(s)." } else { "VoxGen: streamed generated candidate(s)." });
                    for summary in &summaries {
                        let patches = summary.generated_patches.unwrap_or(0);
                        let duration = patches as f64 * 0.160;
                        let perf = summary.rtf.map(|v| format!(", RTF {v:.2}")).unwrap_or_default();
                        append_log(log, &format!("Variation {}/{}: {patches} acoustic patches (~{duration:.2}s){perf}, seed {}.", summary.index, variations, summary.seed));
                    }
                    append_benchmark_results(log, &engine_mode, true, &text, variations, &summaries);
                    input.set_focus();
                }
                Ok(SynthesisPlayback::File { path, cloned, summaries }) => {
                    Sound::stop();
                    if let Ok(mut guard) = state.lock() {
                        if let Some(old) = guard.playback_file.take() { let _ = fs::remove_file(old); }
                        guard.playback_file = Some(path.clone());
                    }
                    if Sound::play_file(path.to_string_lossy().as_ref(), SoundFlags::Async) {
                        append_log(log, if cloned { "VoxGen: playing cloned voice candidate(s)." } else { "VoxGen: playing generated candidate(s)." });
                        for summary in &summaries {
                            let patches = summary.generated_patches.unwrap_or(0);
                            let duration = patches as f64 * 0.160;
                            let stop_text = summary.stopped_by_predictor.map(|v| if v { "stop predictor" } else { "max-step limit" }).unwrap_or("unknown stop reason");
                            let perf = summary.rtf.map(|v| format!(", RTF {v:.2}")).unwrap_or_default();
                            append_log(log, &format!("Variation {}/{}: {patches} acoustic patches (~{duration:.2}s), {stop_text}{perf}, seed {}.", summary.index, variations, summary.seed));
                        }
                        append_benchmark_results(log, &engine_mode, false, &text, variations, &summaries);
                        input.set_focus();
                    } else {
                        append_log(log, "Playback error: wxSound could not play the generated WAV.");
                    }
                }
                Err(err) => append_log(log, &format!("Error: {err}")),
            }
            }
            input.enable(true);
            speak.enable(true);
            stop.enable(false);
            voice.enable(true);
            base_model.enable(true);
            acoustic_model.enable(true);
            load_models_button.enable(true);
            benchmark_ab_button.enable(true);
            profile_xtx_button.enable(true);
            word_spacing_control.enable(true);
            gain_control.enable(true);
        }));
    });
}

fn main() {
    #[cfg(windows)]
    SystemOptions::set_option_by_int("msw.no-manifest-check", 1);

    let settings_path = demo_settings_path();
    let (initial_settings, settings_warning) = match DemoSettings::load(&settings_path) {
        Ok(settings) => (settings, None),
        Err(err) => (DemoSettings::default(), Some(err)),
    };
    let initial_state = DemoState {
        voice_sample: existing_file(initial_settings.voice_sample.clone()),
        base_model: existing_file(initial_settings.base_model.clone()),
        acoustic_model: existing_file(initial_settings.acoustic_model.clone()),
        ..DemoState::default()
    };
    let state: SharedState = Arc::new(Mutex::new(initial_state));
    let state_for_app = Arc::clone(&state);
    let settings: SharedSettings = Arc::new(Mutex::new(initial_settings.clone()));
    let settings_for_app = Arc::clone(&settings);

    // Create settings.cfg on first launch. Do not relocate it elsewhere if the
    // executable directory is not writable; portable state belongs beside the demo.
    let initial_save_warning = initial_settings.save(&settings_path).err();

    let _ = wxdragon::main(move |_| {
        let frame = Frame::builder()
            .with_title("VoxGen Demo")
            .with_size(Size::new(1080, 780))
            .build();
        let panel = Panel::builder(&frame).build();

        let root = BoxSizer::builder(Orientation::Vertical).build();
        let model_buttons = BoxSizer::builder(Orientation::Horizontal).build();
        let diagnostics_row = BoxSizer::builder(Orientation::Horizontal).build();
        let expressive_controls_row = BoxSizer::builder(Orientation::Horizontal).build();
        let custom_controls_row = BoxSizer::builder(Orientation::Horizontal).build();
        let cloning_controls_row = BoxSizer::builder(Orientation::Horizontal).build();
        let generation_controls_row = BoxSizer::builder(Orientation::Horizontal).build();
        let playback_controls_row = BoxSizer::builder(Orientation::Horizontal).build();
        let action_buttons = BoxSizer::builder(Orientation::Horizontal).build();
        let live_playback_controls = LivePlaybackControls::new(
            initial_settings.speed_percent,
            initial_settings.pitch_semitones,
        );
        let synthesis_cancel = SynthesisCancel::default();

        // These remain the demo's two text boxes: activity/output above, TTS input below.
        let output = TextCtrl::builder(&panel)
            .with_style(TextCtrlStyle::MultiLine)
            .with_value("VoxGen Demo\nStarting engine...\n")
            .build();
        output.set_editable(false);

        let input = TextCtrl::builder(&panel)
            .with_style(TextCtrlStyle::MultiLine)
            .with_value("")
            .build();
        input.set_min_size(Size::new(-1, 100));

        let base_model_button = Button::builder(&panel)
            .with_label("Select BaseLM component...")
            .build();
        let acoustic_model_button = Button::builder(&panel)
            .with_label("Select Acoustic component...")
            .build();
        let load_models_button = Button::builder(&panel)
            .with_label("Load VoxCPM2")
            .build();
        load_models_button.set_tooltip(
            "Load the selected models. If Engine mode changed, restart VoxGen in that mode first.",
        );
        let benchmark_ab_button = Button::builder(&panel)
            .with_label("Benchmark Normal vs XTX")
            .build();
        benchmark_ab_button.set_tooltip(
            "Run the current text once in Normal and once in XTX 7900 with the identical seed and no playback, then compare the results.",
        );
        let profile_xtx_button = Button::builder(&panel)
            .with_label("Profile XTX")
            .build();
        profile_xtx_button.set_tooltip(
            "Run one offline XTX 7900 synthesis with Vulkan GPU timestamps enabled, print hot kernels, then restore the selected live mode.",
        );
        let engine_mode_label = StaticText::builder(&panel).with_label("Engine mode:").build();
        let engine_mode_labels = ENGINE_MODES.iter().map(|(_, label)| *label).collect::<Vec<_>>();
        let engine_mode_control = ComboBox::builder(&panel)
            .with_string_choices(&engine_mode_labels)
            .with_style(ComboBoxStyle::ReadOnly)
            .build();
        engine_mode_control.set_selection(table_index(&ENGINE_MODES, &initial_settings.engine_mode));
        engine_mode_control.set_min_size(Size::new(255, -1));
        engine_mode_control.set_tooltip(
            "Normal uses the portable Vulkan reference path. XTX 7900 launches VoxGen with --mode xtx7900.",
        );
        let initial_style_selection = Some(table_index(&STYLE_PRESETS, &initial_settings.style_preset));
        let initial_sample_button_label = format!(
            "Select {} sample...",
            table_label(&STYLE_PRESETS, initial_style_selection),
        );
        let voice_button = Button::builder(&panel)
            .with_label(&initial_sample_button_label)
            .build();
        let style_label = StaticText::builder(&panel).with_label("Style / emotion:").build();
        let style_labels = STYLE_PRESETS.iter().map(|(_, label)| *label).collect::<Vec<_>>();
        let style_control = ComboBox::builder(&panel)
            .with_string_choices(&style_labels)
            .with_style(ComboBoxStyle::ReadOnly)
            .build();
        style_control.set_selection(table_index(&STYLE_PRESETS, &initial_settings.style_preset));
        style_control.set_min_size(Size::new(170, -1));

        let intensity_label = StaticText::builder(&panel).with_label("Intensity:").build();
        let intensity_labels = INTENSITIES.iter().map(|(_, label)| *label).collect::<Vec<_>>();
        let intensity_control = ComboBox::builder(&panel)
            .with_string_choices(&intensity_labels)
            .with_style(ComboBoxStyle::ReadOnly)
            .build();
        intensity_control.set_selection(table_index(&INTENSITIES, &initial_settings.style_intensity));
        intensity_control.set_min_size(Size::new(100, -1));

        let custom_control_label = StaticText::builder(&panel).with_label("Custom instruction:").build();
        let custom_control = TextCtrl::builder(&panel)
            .with_value(&initial_settings.custom_control)
            .build();
        custom_control.set_min_size(Size::new(520, -1));

        let clone_mode_label = StaticText::builder(&panel).with_label("Clone mode:").build();
        let clone_labels = CLONE_MODES.iter().map(|(_, label)| *label).collect::<Vec<_>>();
        let clone_mode_control = ComboBox::builder(&panel)
            .with_string_choices(&clone_labels)
            .with_style(ComboBoxStyle::ReadOnly)
            .build();
        clone_mode_control.set_selection(table_index(&CLONE_MODES, &initial_settings.clone_mode));
        clone_mode_control.set_min_size(Size::new(170, -1));
        let ultimate_initial = initial_settings.clone_mode == "ultimate";
        intensity_control.enable(!ultimate_initial);
        custom_control.enable(!ultimate_initial);
        let prompt_text_label = StaticText::builder(&panel).with_label("Transcript of reference audio:").build();
        let prompt_text_control = TextCtrl::builder(&panel)
            .with_value(&initial_settings.prompt_text)
            .build();
        prompt_text_control.set_min_size(Size::new(350, -1));
        const REFERENCE_TRANSCRIPT_TOOLTIP: &str = "Enter exactly what is spoken in the selected reference WAV. This is not the text to synthesize.";
        prompt_text_label.set_tooltip(REFERENCE_TRANSCRIPT_TOOLTIP);
        prompt_text_control.set_tooltip(REFERENCE_TRANSCRIPT_TOOLTIP);
        prompt_text_control.enable(ultimate_initial);
        let emotion_ref_button = Button::builder(&panel).with_label("Set preset reference...").build();
        let clear_emotion_ref_button = Button::builder(&panel).with_label("Clear preset ref").build();

        let variations_label = StaticText::builder(&panel).with_label("Variations:").build();
        let variations_control = SpinCtrl::builder(&panel).with_range(1, MAX_VARIATIONS as i32).build();
        variations_control.set_value(initial_settings.variations as i32);
        variations_control.set_min_size(Size::new(60, -1));
        let cfg_label = StaticText::builder(&panel).with_label("CFG (%):").build();
        let cfg_control = SpinCtrl::builder(&panel).with_range(100, 300).build();
        cfg_control.set_value(initial_settings.cfg_percent as i32);
        cfg_control.set_min_size(Size::new(70, -1));
        let temperature_label = StaticText::builder(&panel).with_label("Temperature (%):").build();
        let temperature_control = SpinCtrl::builder(&panel).with_range(50, 150).build();
        temperature_control.set_value(initial_settings.temperature_percent as i32);
        temperature_control.set_min_size(Size::new(70, -1));
        let timesteps_label = StaticText::builder(&panel).with_label("CFM steps:").build();
        let timesteps_control = SpinCtrl::builder(&panel).with_range(4, 30).build();
        timesteps_control.set_value(initial_settings.inference_timesteps as i32);
        timesteps_control.set_min_size(Size::new(60, -1));
        let word_spacing_label = StaticText::builder(&panel)
            .with_label("Word spacing (ms):")
            .build();
        let word_spacing_control = SpinCtrl::builder(&panel)
            .with_range(0, 100)
            .build();
        word_spacing_control.set_value(initial_settings.word_spacing_ms as i32);
        word_spacing_control.set_min_size(Size::new(72, -1));

        let speed_label = StaticText::builder(&panel)
            .with_label("Speed (%):")
            .build();
        let speed_control = SpinCtrl::builder(&panel)
            .with_range(MIN_SPEED_PERCENT as i32, MAX_SPEED_PERCENT as i32)
            .build();
        speed_control.set_value(initial_settings.speed_percent as i32);
        speed_control.set_min_size(Size::new(72, -1));

        let pitch_label = StaticText::builder(&panel)
            .with_label("Pitch (semitones):")
            .build();
        let pitch_control = SpinCtrl::builder(&panel)
            .with_range(MIN_PITCH_SEMITONES, MAX_PITCH_SEMITONES)
            .build();
        pitch_control.set_value(initial_settings.pitch_semitones);
        pitch_control.set_min_size(Size::new(72, -1));
        let gain_label = StaticText::builder(&panel)
            .with_label("Gain (%):")
            .build();
        let gain_control = SpinCtrl::builder(&panel)
            .with_range(MIN_GAIN_PERCENT as i32, MAX_GAIN_PERCENT as i32)
            .build();
        gain_control.set_value(initial_settings.gain_percent as i32);
        gain_control.set_min_size(Size::new(72, -1));
        let live_hint = StaticText::builder(&panel)
            .with_label(if initial_settings.stream {
                "Live while speaking"
            } else {
                "Applied before playback"
            })
            .build();

        let speak_button = Button::builder(&panel).with_label("Speak").build();
        let stop_button = Button::builder(&panel).with_label("Stop").build();
        stop_button.enable(false);

        root.add(&output, 1, SizerFlag::Expand | SizerFlag::All, 10);
        model_buttons.add(&base_model_button, 0, SizerFlag::Right, 8);
        model_buttons.add(&acoustic_model_button, 0, SizerFlag::Right, 12);
        model_buttons.add(&engine_mode_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        model_buttons.add(&engine_mode_control, 0, SizerFlag::Right, 8);
        model_buttons.add(&load_models_button, 0, SizerFlag::Right, 8);
        model_buttons.add_stretch_spacer(1);
        root.add_sizer(
            &model_buttons,
            0,
            SizerFlag::Expand | SizerFlag::Left | SizerFlag::Right | SizerFlag::Bottom,
            6,
        );
        let diagnostics_label = StaticText::builder(&panel).with_label("Diagnostics:").build();
        diagnostics_row.add(&diagnostics_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        diagnostics_row.add(&benchmark_ab_button, 0, SizerFlag::Right, 8);
        diagnostics_row.add(&profile_xtx_button, 0, SizerFlag::Right, 8);
        diagnostics_row.add_stretch_spacer(1);
        root.add_sizer(
            &diagnostics_row,
            0,
            SizerFlag::Expand | SizerFlag::Left | SizerFlag::Right | SizerFlag::Bottom,
            10,
        );
        root.add(
            &input,
            0,
            SizerFlag::Expand | SizerFlag::Left | SizerFlag::Right | SizerFlag::Bottom,
            10,
        );
        expressive_controls_row.add(&style_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        expressive_controls_row.add(&style_control, 0, SizerFlag::Right, 12);
        expressive_controls_row.add(&intensity_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        expressive_controls_row.add(&intensity_control, 0, SizerFlag::Right, 12);
        expressive_controls_row.add_stretch_spacer(1);
        root.add_sizer(&expressive_controls_row, 0, SizerFlag::Expand | SizerFlag::Left | SizerFlag::Right | SizerFlag::Bottom, 10);

        custom_controls_row.add(&custom_control_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        custom_controls_row.add(&custom_control, 1, SizerFlag::Expand, 0);
        root.add_sizer(&custom_controls_row, 0, SizerFlag::Expand | SizerFlag::Left | SizerFlag::Right | SizerFlag::Bottom, 10);

        cloning_controls_row.add(&clone_mode_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        cloning_controls_row.add(&clone_mode_control, 0, SizerFlag::Right, 12);
        cloning_controls_row.add(&prompt_text_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        cloning_controls_row.add(&prompt_text_control, 1, SizerFlag::Right | SizerFlag::Expand, 12);
        cloning_controls_row.add(&emotion_ref_button, 0, SizerFlag::Right, 6);
        cloning_controls_row.add(&clear_emotion_ref_button, 0, SizerFlag::empty(), 0);
        root.add_sizer(&cloning_controls_row, 0, SizerFlag::Expand | SizerFlag::Left | SizerFlag::Right | SizerFlag::Bottom, 10);

        generation_controls_row.add(&variations_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        generation_controls_row.add(&variations_control, 0, SizerFlag::Right, 12);
        generation_controls_row.add(&cfg_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        generation_controls_row.add(&cfg_control, 0, SizerFlag::Right, 12);
        generation_controls_row.add(&temperature_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        generation_controls_row.add(&temperature_control, 0, SizerFlag::Right, 12);
        generation_controls_row.add(&timesteps_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        generation_controls_row.add(&timesteps_control, 0, SizerFlag::Right, 12);
        generation_controls_row.add_stretch_spacer(1);
        root.add_sizer(&generation_controls_row, 0, SizerFlag::Expand | SizerFlag::Left | SizerFlag::Right | SizerFlag::Bottom, 10);
        playback_controls_row.add(&speed_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        playback_controls_row.add(&speed_control, 0, SizerFlag::Right, 12);
        playback_controls_row.add(&pitch_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        playback_controls_row.add(&pitch_control, 0, SizerFlag::Right, 12);
        playback_controls_row.add(&gain_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        playback_controls_row.add(&gain_control, 0, SizerFlag::Right, 12);
        playback_controls_row.add(&live_hint, 0, SizerFlag::AlignCenterVertical, 0);
        playback_controls_row.add_stretch_spacer(1);
        root.add_sizer(
            &playback_controls_row,
            0,
            SizerFlag::Expand | SizerFlag::Left | SizerFlag::Right | SizerFlag::Bottom,
            10,
        );
        action_buttons.add(&voice_button, 0, SizerFlag::Right, 8);
        action_buttons.add(&word_spacing_label, 0, SizerFlag::Right | SizerFlag::AlignCenterVertical, 6);
        action_buttons.add(&word_spacing_control, 0, SizerFlag::Right, 8);
        action_buttons.add_stretch_spacer(1);
        action_buttons.add(&speak_button, 0, SizerFlag::Right, 8);
        action_buttons.add(&stop_button, 0, SizerFlag::empty(), 0);
        root.add_sizer(
            &action_buttons,
            0,
            SizerFlag::Expand | SizerFlag::Left | SizerFlag::Right | SizerFlag::Bottom,
            10,
        );

        panel.set_sizer(root, true);
        update_emotion_sample_button(&voice_button, &panel, style_control.get_selection());

        append_log(output, &format!("Settings: {}", settings_path.display()));
        if let Some(err) = settings_warning.as_ref() {
            append_log(output, &format!("Settings load warning: {err}; using defaults where needed."));
        }
        if let Some(err) = initial_save_warning.as_ref() {
            append_log(output, &format!("Settings save warning: {err}"));
        }
        if let Some(path) = initial_settings.voice_sample.as_ref().filter(|p| !p.is_file()) {
            append_log(output, &format!("Saved voice sample is missing and was ignored: {}", path.display()));
        }
        if let Some(path) = initial_settings.base_model.as_ref().filter(|p| !p.is_file()) {
            append_log(output, &format!("Saved BaseLM path is missing and was ignored: {}", path.display()));
        }
        if let Some(path) = initial_settings.acoustic_model.as_ref().filter(|p| !p.is_file()) {
            append_log(output, &format!("Saved Acoustic path is missing and was ignored: {}", path.display()));
        }
        for (preset, path) in &initial_settings.emotion_references {
            if !path.is_file() {
                append_log(output, &format!("Saved {preset} emotion reference is missing and will fall back to the default voice: {}", path.display()));
            }
        }

        // BaseLM selection: Q8_0 and F16 are both accepted; VoxGen detects the
        // format from the GGUF unless the API caller explicitly forces one.
        {
            let state = Arc::clone(&state_for_app);
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let speak_button = speak_button;
            base_model_button.on_click(move |_| {
                let dialog = FileDialog::builder(&frame).build();
                dialog.set_message("Select VoxCPM2 BaseLM GGUF (Q8_0 or F16)");
                dialog.set_wildcard("GGUF model (*.gguf)|*.gguf|All files (*.*)|*.*");
                let _ = dialog.show_modal();
                if let Some(path) = dialog.get_path() {
                    let path = PathBuf::from(path);
                    if !path.is_file() {
                        append_log(output, "BaseLM selection is not a file.");
                        return;
                    }
                    if let Ok(mut guard) = state.lock() {
                        guard.base_model = Some(path.clone());
                    }
                    if let Ok(mut cfg) = settings.lock() {
                        cfg.base_model = Some(path.clone());
                    }
                    if let Err(err) = save_shared_settings(&settings) {
                        append_log(output, &format!("Settings save warning: {err}"));
                    }
                    speak_button.enable(false);
                    append_log(output, &format!("Selected BaseLM component: {}", path.display()));
                    append_log(output, "Click Load VoxCPM2 to apply the new model selection.");
                }
            });
        }

        // Acoustic selection. VoxCPM2 uses the same Acoustic-F16 GGUF with
        // either Q8_0 or F16 BaseLM weights.
        {
            let state = Arc::clone(&state_for_app);
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let speak_button = speak_button;
            acoustic_model_button.on_click(move |_| {
                let dialog = FileDialog::builder(&frame).build();
                dialog.set_message("Select VoxCPM2 Acoustic F16 GGUF");
                dialog.set_wildcard("GGUF model (*.gguf)|*.gguf|All files (*.*)|*.*");
                let _ = dialog.show_modal();
                if let Some(path) = dialog.get_path() {
                    let path = PathBuf::from(path);
                    if !path.is_file() {
                        append_log(output, "Acoustic model selection is not a file.");
                        return;
                    }
                    if let Ok(mut guard) = state.lock() {
                        guard.acoustic_model = Some(path.clone());
                    }
                    if let Ok(mut cfg) = settings.lock() {
                        cfg.acoustic_model = Some(path.clone());
                    }
                    if let Err(err) = save_shared_settings(&settings) {
                        append_log(output, &format!("Settings save warning: {err}"));
                    }
                    speak_button.enable(false);
                    append_log(output, &format!("Selected Acoustic component: {}", path.display()));
                    append_log(output, "Click Load VoxCPM2 to apply the new model selection.");
                }
            });
        }

        // Unified model/mode action. If the selected execution mode differs from
        // the running VoxGen server, restart the engine first; otherwise keep the
        // current runtime and only (re)load the selected VoxCPM2 model bundle.
        {
            let state = Arc::clone(&state_for_app);
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let input = input;
            let base_button = base_model_button;
            let acoustic_button = acoustic_model_button;
            let load_button_copy = load_models_button;
            let benchmark_button_copy = benchmark_ab_button;
            let profile_button_copy = profile_xtx_button;
            let speak_button = speak_button;
            let mode_control = engine_mode_control;
            load_models_button.on_click(move |_| {
                let (base, acoustic) = match selected_models(&state) {
                    Ok(paths) => paths,
                    Err(err) => {
                        append_log(output, &format!("VoxCPM2 bundle selection: {err}"));
                        return;
                    }
                };
                let engine_mode = table_key(&ENGINE_MODES, mode_control.get_selection()).to_string();
                let mode_label = table_label(&ENGINE_MODES, mode_control.get_selection());
                if let Ok(mut cfg) = settings.lock() {
                    cfg.engine_mode = engine_mode.clone();
                }
                if let Err(err) = save_shared_settings(&settings) {
                    append_log(output, &format!("Settings save warning: {err}"));
                }

                let mode_mismatch = engine_check()
                    && (server_execution_mode().as_deref() != Some(engine_mode.as_str())
                        || (engine_mode == "xtx7900" && server_xtx_stream_safe() != Some(true)));
                if mode_mismatch {
                    append_log(output, &format!("Engine mode changed to {mode_label}; restarting VoxGen before loading VoxCPM2..."));
                }
                append_log(output, &format!("Loading BaseLM component: {}", base.display()));
                append_log(output, &format!("Loading Acoustic component: {}", acoustic.display()));
                input.enable(false);
                speak_button.enable(false);
                base_button.enable(false);
                acoustic_button.enable(false);
                load_button_copy.enable(false);
                benchmark_button_copy.enable(false);
                profile_button_copy.enable(false);
                mode_control.enable(false);

                let state = Arc::clone(&state);
                let settings = Arc::clone(&settings);
                thread::spawn(move || {
                    let (stream_enabled, default_gain) = settings
                        .lock()
                        .map(|cfg| (cfg.stream, cfg.gain_percent as f32 / 100.0))
                        .unwrap_or((true, DEFAULT_GAIN_PERCENT as f32 / 100.0));
                    let result = (if mode_mismatch {
                        stop_existing_voxgen_server(&state)
                    } else {
                        Ok(())
                    })
                    .and_then(|_| ensure_server(&state, stream_enabled, default_gain, &engine_mode))
                    .and_then(|_| load_models(&base, &acoustic));
                    wxdragon::call_after(Box::new(move || {
                        match result {
                            Ok(message) => {
                                append_log(output, &message);
                                append_log(output, &format!("Engine mode active: {mode_label}. VoxCPM2 bundle ready. Type text below and click Speak."));
                                speak_button.enable(true);
                            }
                            Err(err) => {
                                append_log(output, &format!("Model/mode load error: {err}"));
                                speak_button.enable(false);
                            }
                        }
                        input.enable(true);
                        base_button.enable(true);
                        acoustic_button.enable(true);
                        load_button_copy.enable(true);
                        benchmark_button_copy.enable(true);
                        profile_button_copy.enable(true);
                        mode_control.enable(true);
                    }));
                });
            });
        }

        {
            let state = Arc::clone(&state_for_app);
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let style_control_copy = style_control;
            voice_button.on_click(move |_| {
                let selection = style_control_copy.get_selection();
                let preset = table_key(&STYLE_PRESETS, selection);
                let preset_label = table_label(&STYLE_PRESETS, selection);
                let dialog = FileDialog::builder(&frame).build();
                dialog.set_message(&format!("Reference for {preset_label} emotion:"));
                dialog.set_wildcard("WAV audio (*.wav)|*.wav|All files (*.*)|*.*");
                let _ = dialog.show_modal();
                if let Some(path) = dialog.get_path() {
                    let path = PathBuf::from(path);
                    if !path.is_file() {
                        append_log(output, &format!("{preset_label} sample selection is not a file."));
                        return;
                    }
                    if let Err(err) = validate_voice_wav(&path) {
                        append_log(output, &format!("{preset_label} sample error: {err}"));
                        return;
                    }
                    if let Ok(mut cfg) = settings.lock() {
                        cfg.emotion_references.insert(preset.to_string(), path.clone());
                        // Neutral is the natural fallback for Auto or any preset that
                        // does not yet have its own dedicated emotional reference.
                        if preset == "neutral" {
                            cfg.voice_sample = Some(path.clone());
                        }
                    }
                    if preset == "neutral" {
                        if let Ok(mut guard) = state.lock() {
                            guard.voice_sample = Some(path.clone());
                        }
                    }
                    if let Err(err) = save_shared_settings(&settings) {
                        append_log(output, &format!("Settings save warning: {err}"));
                    }
                    append_log(output, &format!("{preset_label} sample: {}", path.display()));
                    if health_check() {
                        let warm_path = path.clone();
                        let warm_output = output;
                        thread::spawn(move || {
                            match warm_reference_audio(&warm_path) {
                                Ok(message) => wxdragon::call_after(Box::new(move || append_log(warm_output, &message))),
                                Err(err) => wxdragon::call_after(Box::new(move || append_log(warm_output, &format!("Reference warm-up warning: {err}")))),
                            }
                        });
                    }
                }
            });
        }

        {
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let mode_control_copy = engine_mode_control;
            let speak_button_copy = speak_button;
            engine_mode_control.on_selection_changed(move |_| {
                let mode = table_key(&ENGINE_MODES, mode_control_copy.get_selection()).to_string();
                let mode_label = table_label(&ENGINE_MODES, mode_control_copy.get_selection());
                if let Ok(mut cfg) = settings.lock() { cfg.engine_mode = mode.clone(); }
                let requires_restart = engine_check()
                    && (server_execution_mode().as_deref() != Some(mode.as_str())
                        || (mode == "xtx7900" && server_xtx_stream_safe() != Some(true)));
                if requires_restart {
                    speak_button_copy.enable(false);
                    append_log(output, &format!("Selected engine mode: {mode_label}. Click Load VoxCPM2 to switch mode and reload the models."));
                } else {
                    append_log(output, &format!("Selected engine mode: {mode_label}. The running engine already uses this mode."));
                }
                if let Err(err) = save_shared_settings(&settings) { append_log(output, &format!("Settings save warning: {err}")); }
            });
        }

        {
            let settings = Arc::clone(&settings_for_app);
            let state = Arc::clone(&state_for_app);
            let output = output;
            let style_control_copy = style_control;
            let voice_button_copy = voice_button;
            let panel_copy = panel;
            style_control.on_selection_changed(move |_| {
                let selection = style_control_copy.get_selection();
                let preset = table_key(&STYLE_PRESETS, selection);
                update_emotion_sample_button(&voice_button_copy, &panel_copy, selection);
                let settings_snapshot = if let Ok(mut cfg) = settings.lock() {
                    cfg.style_preset = preset.to_string();
                    Some(cfg.clone())
                } else {
                    append_log(output, "Settings lock failed; cannot resolve the voice anchor safely.");
                    None
                };
                if let Err(err) = save_shared_settings(&settings) {
                    append_log(output, &format!("Settings save warning: {err}"));
                }
                let runtime_voice_sample = state.lock().ok().and_then(|guard| guard.voice_sample.clone());
                let resolved_reference = settings_snapshot
                    .as_ref()
                    .map(|cfg| resolve_reference_sample(cfg, preset, runtime_voice_sample))
                    .transpose();
                match resolved_reference {
                    Ok(Some(resolved)) => {
                        if let Some(message) = resolved.describe(preset) { append_log(output, &message); }
                        if health_check() {
                            if let Some(warm_path) = resolved.path {
                                let warm_output = output;
                                thread::spawn(move || {
                                    match warm_reference_audio(&warm_path) {
                                        Ok(message) => wxdragon::call_after(Box::new(move || append_log(warm_output, &message))),
                                        Err(err) => wxdragon::call_after(Box::new(move || append_log(warm_output, &format!("Reference warm-up warning: {err}")))),
                                    }
                                });
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(err) => append_log(output, &err),
                }
            });
        }

        {
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let clone_mode_control_copy = clone_mode_control;
            let style_control_copy = style_control;
            let intensity_control_copy = intensity_control;
            let custom_control_copy = custom_control;
            let prompt_text_control_copy = prompt_text_control;
            clone_mode_control.on_selection_changed(move |_| {
                let mode = table_key(&CLONE_MODES, clone_mode_control_copy.get_selection());
                let ultimate = mode == "ultimate";
                intensity_control_copy.enable(!ultimate);
                custom_control_copy.enable(!ultimate);
                prompt_text_control_copy.enable(ultimate);
                if let Ok(mut cfg) = settings.lock() { cfg.clone_mode = mode.to_string(); }
                if let Err(err) = save_shared_settings(&settings) {
                    append_log(output, &format!("Settings save warning: {err}"));
                }
                if ultimate {
                    append_log(output, "Ultimate cloning uses the selected preset only as an emotional reference profile; intensity/custom textual control is disabled and delivery comes from the reference audio + exact transcript.");
                }
            });
        }

        {
            let settings = Arc::clone(&settings_for_app);
            let state = Arc::clone(&state_for_app);
            let output = output;
            let style_control_copy = style_control;
            emotion_ref_button.on_click(move |_| {
                let selection = style_control_copy.get_selection();
                let preset = table_key(&STYLE_PRESETS, selection);
                let preset_label = selection
                    .and_then(|i| STYLE_PRESETS.get(i as usize).map(|(_, label)| *label))
                    .unwrap_or(STYLE_PRESETS[0].1);
                let dialog = FileDialog::builder(&frame).build();
                dialog.set_message(&format!("Reference for {preset_label} emotion:"));
                dialog.set_wildcard("WAV audio (*.wav)|*.wav|All files (*.*)|*.*");
                let _ = dialog.show_modal();
                if let Some(path) = dialog.get_path() {
                    let path = PathBuf::from(path);
                    if let Err(err) = validate_voice_wav(&path) {
                        append_log(output, &format!("Emotion reference error: {err}"));
                        return;
                    }
                    if let Ok(mut cfg) = settings.lock() {
                        cfg.emotion_references.insert(preset.to_string(), path.clone());
                        if preset == "neutral" {
                            cfg.voice_sample = Some(path.clone());
                        }
                    }
                    if preset == "neutral" {
                        if let Ok(mut guard) = state.lock() {
                            guard.voice_sample = Some(path.clone());
                        }
                    }
                    if let Err(err) = save_shared_settings(&settings) { append_log(output, &format!("Settings save warning: {err}")); }
                    if preset == "neutral" {
                        append_log(output, &format!("Neutral voice anchor: {}", path.display()));
                    } else {
                        append_log(output, &format!("Emotion reference [{preset}]: {}", path.display()));
                    }
                    if health_check() {
                        let warm_path = path.clone();
                        let warm_output = output;
                        thread::spawn(move || {
                            match warm_reference_audio(&warm_path) {
                                Ok(message) => wxdragon::call_after(Box::new(move || append_log(warm_output, &message))),
                                Err(err) => wxdragon::call_after(Box::new(move || append_log(warm_output, &format!("Reference warm-up warning: {err}")))),
                            }
                        });
                    }
                }
            });
        }
        {
            let settings = Arc::clone(&settings_for_app);
            let state = Arc::clone(&state_for_app);
            let output = output;
            let style_control_copy = style_control;
            clear_emotion_ref_button.on_click(move |_| {
                let preset = table_key(&STYLE_PRESETS, style_control_copy.get_selection());
                let removed = if let Ok(mut cfg) = settings.lock() {
                    let removed = cfg.emotion_references.remove(preset);
                    if preset == "neutral" {
                        if removed.as_ref().is_some_and(|path| cfg.voice_sample.as_ref() == Some(path)) {
                            cfg.voice_sample = None;
                        }
                    }
                    removed
                } else {
                    None
                };
                if preset == "neutral" {
                    if let Some(removed_path) = removed.as_ref() {
                        if let Ok(mut guard) = state.lock() {
                            if guard.voice_sample.as_ref() == Some(removed_path) {
                                guard.voice_sample = None;
                            }
                        }
                    }
                }
                if let Err(err) = save_shared_settings(&settings) { append_log(output, &format!("Settings save warning: {err}")); }
                if removed.is_some() {
                    if preset == "neutral" {
                        append_log(output, "Cleared the neutral voice anchor. Zero-shot generation is allowed again only if no other voice sample is configured.");
                    } else {
                        append_log(output, &format!("Cleared emotion reference for preset [{preset}]."));
                    }
                } else {
                    append_log(output, &format!("No emotion reference was saved for preset [{preset}]."));
                }
            });
        }

        // Controlled same-seed A/B benchmark. This intentionally uses the non-streaming
        // endpoint and never plays the generated WAVs, so playback/DSP cannot contaminate
        // the engine comparison. The selected live mode is restored when the pair finishes.
        {
            let output = output;
            let stop_button_copy = stop_button;
            let cancel = synthesis_cancel.clone();
            stop_button.on_click(move |_| {
                if cancel.is_cancelled() {
                    return;
                }
                // Silence queued WinMM / wxSound audio locally first; the HTTP
                // cancellation request then stops GPU generation at the next safe
                // acoustic-patch boundary.
                cancel.cancel();
                Sound::stop();
                stop_button_copy.enable(false);
                append_log(output, "VoxGen: stopping synthesis...");
                let request_id = cancel.request_id();
                thread::spawn(move || {
                    let _ = cancel_active_server_speech(request_id);
                });
            });
        }

        {
            let state = Arc::clone(&state_for_app);
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let input = input;
            let speak_button_copy = speak_button;
            let voice_button_copy = voice_button;
            let base_button_copy = base_model_button;
            let acoustic_button_copy = acoustic_model_button;
            let load_button_copy = load_models_button;
            let benchmark_button_copy = benchmark_ab_button;
            let profile_button_copy = profile_xtx_button;
            let mode_control_copy = engine_mode_control;
            let style_control_copy = style_control;
            let intensity_control_copy = intensity_control;
            let custom_control_copy = custom_control;
            let clone_mode_control_copy = clone_mode_control;
            let prompt_text_control_copy = prompt_text_control;
            let cfg_control_copy = cfg_control;
            let temperature_control_copy = temperature_control;
            let timesteps_control_copy = timesteps_control;
            let gain_control_copy = gain_control;
            benchmark_ab_button.on_click(move |_| {
                let text = input.get_value().trim().to_owned();
                if text.is_empty() {
                    append_log(output, "Type some text in the lower box before benchmarking.");
                    input.set_focus();
                    return;
                }
                let (base, acoustic) = match selected_models(&state) {
                    Ok(paths) => paths,
                    Err(err) => {
                        append_log(output, &format!("A/B benchmark requires loaded model paths: {err}"));
                        return;
                    }
                };
                let preset = table_key(&STYLE_PRESETS, style_control_copy.get_selection()).to_string();
                let intensity = table_key(&INTENSITIES, intensity_control_copy.get_selection()).to_string();
                let clone_mode = table_key(&CLONE_MODES, clone_mode_control_copy.get_selection()).to_string();
                let custom = custom_control_copy.get_value();
                let prompt_text = prompt_text_control_copy.get_value();
                if clone_mode != "ultimate" && preset == "custom" && custom.trim().is_empty() {
                    append_log(output, "Custom style requires a delivery instruction.");
                    return;
                }
                if clone_mode == "ultimate" && prompt_text.trim().is_empty() {
                    append_log(output, "Ultimate cloning requires the exact transcript of the reference audio.");
                    return;
                }
                let base_cfg_value = cfg_control_copy.value().clamp(100, 300) as f32 / 100.0;
                let temperature = temperature_control_copy.value().clamp(50, 150) as f32 / 100.0;
                let inference_timesteps = timesteps_control_copy.value().clamp(4, 30) as u32;
                let mut expressive = build_demo_expressive_request(
                    &text, &preset, &intensity, &custom, clone_mode, prompt_text,
                    base_cfg_value, temperature, inference_timesteps,
                );
                let runtime_voice_sample = state.lock().ok().and_then(|guard| guard.voice_sample.clone());
                let (stream_enabled, settings_snapshot) = match settings.lock() {
                    Ok(cfg) => (cfg.stream, cfg.clone()),
                    Err(_) => { append_log(output, "A/B benchmark: settings lock failed; refusing unanchored generation."); return; }
                };
                let resolved_reference = match resolve_reference_sample(&settings_snapshot, &preset, runtime_voice_sample) {
                    Ok(value) => value,
                    Err(err) => { append_log(output, &format!("A/B benchmark: {err}")); return; }
                };
                if let Some(message) = resolved_reference.describe(&preset) { append_log(output, &message); }
                let sample = resolved_reference.path;
                expressive = match effective_expressive_for_sample(expressive, sample.as_deref()) {
                    Ok(v) => v,
                    Err(err) => { append_log(output, &format!("A/B benchmark: {err}")); return; }
                };
                let gain = gain_control_copy.value().clamp(MIN_GAIN_PERCENT as i32, MAX_GAIN_PERCENT as i32) as f32 / 100.0;
                let selected_mode = table_key(&ENGINE_MODES, mode_control_copy.get_selection()).to_string();
                let seed = next_demo_seed();

                Sound::stop();
                input.enable(false);
                speak_button_copy.enable(false);
                voice_button_copy.enable(false);
                base_button_copy.enable(false);
                acoustic_button_copy.enable(false);
                load_button_copy.enable(false);
                benchmark_button_copy.enable(false);
                profile_button_copy.enable(false);
                mode_control_copy.enable(false);
                append_log(output, &format!("Starting controlled Normal vs XTX 7900 benchmark with identical seed {seed}; playback is disabled..."));

                let state_bg = Arc::clone(&state);
                thread::spawn(move || {
                    let run = (|| -> Result<(OfflineBenchCase, OfflineBenchCase), String> {
                        let normal = run_offline_benchmark_case(
                            &state_bg, &base, &acoustic, gain, "normal",
                            &text, sample.as_deref(), &expressive, seed,
                        )?;
                        let xtx = run_offline_benchmark_case(
                            &state_bg, &base, &acoustic, gain, "xtx7900",
                            &text, sample.as_deref(), &expressive, seed,
                        )?;
                        Ok((normal, xtx))
                    })();
                    let restore = restore_selected_server(
                        &state_bg, &base, &acoustic, stream_enabled, gain, &selected_mode,
                    );
                    wxdragon::call_after(Box::new(move || {
                        match run {
                            Ok((normal, xtx)) => append_ab_results(
                                output, &normal, &xtx, &text, &base, &acoustic,
                                sample.as_deref(), &expressive,
                            ),
                            Err(err) => append_log(output, &format!("A/B benchmark error: {err}")),
                        }
                        match restore {
                            Ok(()) => {
                                append_log(output, &format!("Restored selected engine mode: {}.", if selected_mode == "xtx7900" { "XTX 7900" } else { "Normal" }));
                                speak_button_copy.enable(true);
                            }
                            Err(err) => {
                                append_log(output, &format!("Engine restore error after benchmark: {err}"));
                                speak_button_copy.enable(false);
                            }
                        }
                        input.enable(true);
                        voice_button_copy.enable(true);
                        base_button_copy.enable(true);
                        acoustic_button_copy.enable(true);
                        load_button_copy.enable(true);
                        benchmark_button_copy.enable(true);
                        profile_button_copy.enable(true);
                        mode_control_copy.enable(true);
                        input.set_focus();
                    }));
                });
            });
        }

        // Offline profiler. A dedicated XTX server is relaunched with streaming off and
        // GPU timestamps on; the resulting per-kernel timings are printed only after the
        // synthesis has completed. The stream-safe live server is restored afterward.
        {
            let state = Arc::clone(&state_for_app);
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let input = input;
            let speak_button_copy = speak_button;
            let voice_button_copy = voice_button;
            let base_button_copy = base_model_button;
            let acoustic_button_copy = acoustic_model_button;
            let load_button_copy = load_models_button;
            let benchmark_button_copy = benchmark_ab_button;
            let profile_button_copy = profile_xtx_button;
            let mode_control_copy = engine_mode_control;
            let style_control_copy = style_control;
            let intensity_control_copy = intensity_control;
            let custom_control_copy = custom_control;
            let clone_mode_control_copy = clone_mode_control;
            let prompt_text_control_copy = prompt_text_control;
            let cfg_control_copy = cfg_control;
            let temperature_control_copy = temperature_control;
            let timesteps_control_copy = timesteps_control;
            let gain_control_copy = gain_control;
            profile_xtx_button.on_click(move |_| {
                let text = input.get_value().trim().to_owned();
                if text.is_empty() {
                    append_log(output, "Type some text in the lower box before profiling.");
                    input.set_focus();
                    return;
                }
                let (base, acoustic) = match selected_models(&state) {
                    Ok(paths) => paths,
                    Err(err) => { append_log(output, &format!("XTX profile requires loaded model paths: {err}")); return; }
                };
                let preset = table_key(&STYLE_PRESETS, style_control_copy.get_selection()).to_string();
                let intensity = table_key(&INTENSITIES, intensity_control_copy.get_selection()).to_string();
                let clone_mode = table_key(&CLONE_MODES, clone_mode_control_copy.get_selection()).to_string();
                let custom = custom_control_copy.get_value();
                let prompt_text = prompt_text_control_copy.get_value();
                if clone_mode != "ultimate" && preset == "custom" && custom.trim().is_empty() {
                    append_log(output, "Custom style requires a delivery instruction.");
                    return;
                }
                if clone_mode == "ultimate" && prompt_text.trim().is_empty() {
                    append_log(output, "Ultimate cloning requires the exact transcript of the reference audio.");
                    return;
                }
                let base_cfg_value = cfg_control_copy.value().clamp(100, 300) as f32 / 100.0;
                let temperature = temperature_control_copy.value().clamp(50, 150) as f32 / 100.0;
                let inference_timesteps = timesteps_control_copy.value().clamp(4, 30) as u32;
                let mut expressive = build_demo_expressive_request(
                    &text, &preset, &intensity, &custom, clone_mode, prompt_text,
                    base_cfg_value, temperature, inference_timesteps,
                );
                let runtime_voice_sample = state.lock().ok().and_then(|guard| guard.voice_sample.clone());
                let (stream_enabled, settings_snapshot) = match settings.lock() {
                    Ok(cfg) => (cfg.stream, cfg.clone()),
                    Err(_) => { append_log(output, "XTX profile: settings lock failed; refusing unanchored generation."); return; }
                };
                let resolved_reference = match resolve_reference_sample(&settings_snapshot, &preset, runtime_voice_sample) {
                    Ok(value) => value,
                    Err(err) => { append_log(output, &format!("XTX profile: {err}")); return; }
                };
                if let Some(message) = resolved_reference.describe(&preset) { append_log(output, &message); }
                let sample = resolved_reference.path;
                expressive = match effective_expressive_for_sample(expressive, sample.as_deref()) {
                    Ok(v) => v,
                    Err(err) => { append_log(output, &format!("XTX profile: {err}")); return; }
                };
                let gain = gain_control_copy.value().clamp(MIN_GAIN_PERCENT as i32, MAX_GAIN_PERCENT as i32) as f32 / 100.0;
                let selected_mode = table_key(&ENGINE_MODES, mode_control_copy.get_selection()).to_string();
                let seed = next_demo_seed();

                Sound::stop();
                input.enable(false);
                speak_button_copy.enable(false);
                voice_button_copy.enable(false);
                base_button_copy.enable(false);
                acoustic_button_copy.enable(false);
                load_button_copy.enable(false);
                benchmark_button_copy.enable(false);
                profile_button_copy.enable(false);
                mode_control_copy.enable(false);
                append_log(output, "Starting offline XTX GPU profile; live streaming is temporarily disabled and timestamp readback is enabled...");

                let state_bg = Arc::clone(&state);
                thread::spawn(move || {
                    let run = (|| -> Result<(SpeechWav, Vec<GpuProfileRow>, f64), String> {
                        stop_existing_voxgen_server(&state_bg)?;
                        ensure_offline_profile_server(&state_bg, gain)?;
                        load_models(&base, &acoustic)?;
                        reset_server_gpu_profile()?;
                        let started = std::time::Instant::now();
                        let speech = speech_wav(&text, sample.as_deref(), gain, seed, &expressive, None)?;
                        let wall_seconds = started.elapsed().as_secs_f64();
                        let rows = fetch_server_gpu_profile()?;
                        Ok((speech, rows, wall_seconds))
                    })();
                    let restore = restore_selected_server(
                        &state_bg, &base, &acoustic, stream_enabled, gain, &selected_mode,
                    );
                    wxdragon::call_after(Box::new(move || {
                        match run {
                            Ok((speech, rows, wall_seconds)) => append_gpu_profile_results(
                                output, &rows, &text, &base, &acoustic, sample.as_deref(),
                                &expressive, seed, &speech, wall_seconds,
                            ),
                            Err(err) => append_log(output, &format!("XTX offline profile error: {err}")),
                        }
                        match restore {
                            Ok(()) => {
                                append_log(output, &format!("Restored selected engine mode: {}.", if selected_mode == "xtx7900" { "XTX 7900" } else { "Normal" }));
                                speak_button_copy.enable(true);
                            }
                            Err(err) => {
                                append_log(output, &format!("Engine restore error after profiling: {err}"));
                                speak_button_copy.enable(false);
                            }
                        }
                        input.enable(true);
                        voice_button_copy.enable(true);
                        base_button_copy.enable(true);
                        acoustic_button_copy.enable(true);
                        load_button_copy.enable(true);
                        benchmark_button_copy.enable(true);
                        profile_button_copy.enable(true);
                        mode_control_copy.enable(true);
                        input.set_focus();
                    }));
                });
            });
        }

        // Speed and pitch are intentionally left enabled during synthesis.
        // Their atomics are sampled by the 128-sample DSP loop, so edits affect
        // audio which has not yet been queued to the output device.
        {
            let controls = live_playback_controls.clone();
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let speed_control_copy = speed_control;
            speed_control.on_value_changed(move |_| {
                controls.set_speed_percent(speed_control_copy.value());
                if let Ok(mut cfg) = settings.lock() {
                    cfg.speed_percent = controls.speed_percent();
                }
                if let Err(err) = save_shared_settings(&settings) {
                    append_log(output, &format!("Settings save warning: {err}"));
                }
            });
        }
        {
            let controls = live_playback_controls.clone();
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let pitch_control_copy = pitch_control;
            pitch_control.on_value_changed(move |_| {
                controls.set_pitch_semitones(pitch_control_copy.value());
                if let Ok(mut cfg) = settings.lock() {
                    cfg.pitch_semitones = controls.pitch_semitones();
                }
                if let Err(err) = save_shared_settings(&settings) {
                    append_log(output, &format!("Settings save warning: {err}"));
                }
            });
        }

        {
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let word_spacing_control_copy = word_spacing_control;
            word_spacing_control.on_value_changed(move |_| {
                if let Ok(mut cfg) = settings.lock() {
                    cfg.word_spacing_ms = word_spacing_control_copy.value().clamp(0, 100) as u32;
                }
                if let Err(err) = save_shared_settings(&settings) {
                    append_log(output, &format!("Settings save warning: {err}"));
                }
            });
        }

        {
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let gain_control_copy = gain_control;
            gain_control.on_value_changed(move |_| {
                if let Ok(mut cfg) = settings.lock() {
                    cfg.gain_percent = gain_control_copy
                        .value()
                        .clamp(MIN_GAIN_PERCENT as i32, MAX_GAIN_PERCENT as i32) as u32;
                }
                if let Err(err) = save_shared_settings(&settings) {
                    append_log(output, &format!("Settings save warning: {err}"));
                }
            });
        }

        {
            let state = Arc::clone(&state_for_app);
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let input = input;
            let speak_button_copy = speak_button;
            let stop_button_copy = stop_button;
            let voice_button_copy = voice_button;
            let base_button_copy = base_model_button;
            let acoustic_button_copy = acoustic_model_button;
            let load_button_copy = load_models_button;
            let benchmark_button_copy = benchmark_ab_button;
            let profile_button_copy = profile_xtx_button;
            let word_spacing_control_copy = word_spacing_control;
            let gain_control_copy = gain_control;
            let style_control_copy = style_control;
            let intensity_control_copy = intensity_control;
            let custom_control_copy = custom_control;
            let clone_mode_control_copy = clone_mode_control;
            let prompt_text_control_copy = prompt_text_control;
            let variations_control_copy = variations_control;
            let cfg_control_copy = cfg_control;
            let temperature_control_copy = temperature_control;
            let timesteps_control_copy = timesteps_control;
            let engine_mode_control_copy = engine_mode_control;
            // Give the Speak callback its own shared-control handle. The callback is
            // `move`, so capturing `live_playback_controls` directly would consume
            // the outer handle and reproduce the v0.7.13 E0382 ownership failure.
            let speak_live_playback_controls = live_playback_controls.clone();
            let speak_cancel = synthesis_cancel.clone();
            speak_button.on_click(move |_| {
                let text = input.get_value().trim().to_owned();
                if text.is_empty() {
                    append_log(output, "Type some text in the lower box first.");
                    input.set_focus();
                    return;
                }
                let preset = table_key(&STYLE_PRESETS, style_control_copy.get_selection()).to_string();
                let intensity = table_key(&INTENSITIES, intensity_control_copy.get_selection()).to_string();
                let clone_mode = table_key(&CLONE_MODES, clone_mode_control_copy.get_selection()).to_string();
                let custom = custom_control_copy.get_value();
                let prompt_text = prompt_text_control_copy.get_value();
                if clone_mode != "ultimate" && preset == "custom" && custom.trim().is_empty() {
                    append_log(output, "Custom style requires a delivery instruction.");
                    custom_control_copy.set_focus();
                    return;
                }
                if clone_mode == "ultimate" && prompt_text.trim().is_empty() {
                    append_log(output, "Ultimate cloning requires the exact transcript of the reference audio.");
                    prompt_text_control_copy.set_focus();
                    return;
                }
                let word_spacing_ms = word_spacing_control_copy.value().clamp(0, 100) as u32;
                let gain_percent = gain_control_copy.value().clamp(MIN_GAIN_PERCENT as i32, MAX_GAIN_PERCENT as i32) as u32;
                let variations = variations_control_copy.value().clamp(1, MAX_VARIATIONS as i32) as u32;
                let cfg_percent = cfg_control_copy.value().clamp(100, 300) as u32;
                let temperature_percent = temperature_control_copy.value().clamp(50, 150) as u32;
                let inference_timesteps = timesteps_control_copy.value().clamp(4, 30) as u32;
                let engine_mode = table_key(&ENGINE_MODES, engine_mode_control_copy.get_selection()).to_string();
                let runtime_voice_sample = state.lock().ok().and_then(|guard| guard.voice_sample.clone());
                let (stream_enabled, settings_snapshot) = match settings.lock() {
                    Ok(mut cfg) => {
                        cfg.style_preset = preset.clone();
                        cfg.style_intensity = intensity.clone();
                        cfg.custom_control = custom.clone();
                        cfg.clone_mode = clone_mode.clone();
                        cfg.prompt_text = prompt_text.clone();
                        cfg.variations = variations;
                        cfg.cfg_percent = cfg_percent;
                        cfg.temperature_percent = temperature_percent;
                        cfg.inference_timesteps = inference_timesteps;
                        cfg.word_spacing_ms = word_spacing_ms;
                        cfg.gain_percent = gain_percent;
                        cfg.engine_mode = engine_mode.clone();
                        (cfg.stream, cfg.clone())
                    }
                    Err(_) => {
                        append_log(output, "Settings lock failed; refusing unanchored voice generation.");
                        return;
                    }
                };
                if let Err(err) = save_shared_settings(&settings) {
                    append_log(output, &format!("Settings save warning: {err}"));
                }
                let resolved_reference = match resolve_reference_sample(&settings_snapshot, &preset, runtime_voice_sample) {
                    Ok(value) => value,
                    Err(err) => { append_log(output, &err); return; }
                };
                if let Some(message) = resolved_reference.describe(&preset) { append_log(output, &message); }
                let sample_override = resolved_reference.path;
                let expressive = build_demo_expressive_request(
                    &text,
                    &preset,
                    &intensity,
                    &custom,
                    clone_mode,
                    prompt_text,
                    cfg_percent as f32 / 100.0,
                    temperature_percent as f32 / 100.0,
                    inference_timesteps,
                );
                synthesize(
                    text,
                    Arc::clone(&state),
                    output,
                    input,
                    speak_button_copy,
                    stop_button_copy,
                    voice_button_copy,
                    base_button_copy,
                    acoustic_button_copy,
                    load_button_copy,
                    benchmark_button_copy,
                    profile_button_copy,
                    word_spacing_control_copy,
                    word_spacing_ms,
                    gain_control_copy,
                    gain_percent,
                    speak_live_playback_controls.clone(),
                    speak_cancel.clone(),
                    stream_enabled,
                    engine_mode,
                    sample_override,
                    expressive,
                    variations,
                );
            });
        }

        {
            let state = Arc::clone(&state_for_app);
            let settings = Arc::clone(&settings_for_app);
            let controls = live_playback_controls.clone();
            let word_spacing_control_copy = word_spacing_control;
            let gain_control_copy = gain_control;
            let style_control_copy = style_control;
            let intensity_control_copy = intensity_control;
            let custom_control_copy = custom_control;
            let clone_mode_control_copy = clone_mode_control;
            let prompt_text_control_copy = prompt_text_control;
            let variations_control_copy = variations_control;
            let cfg_control_copy = cfg_control;
            let temperature_control_copy = temperature_control;
            let timesteps_control_copy = timesteps_control;
            let engine_mode_control_copy = engine_mode_control;
            frame.on_close(move |event| {
                Sound::stop();
                if let Ok(mut guard) = state.lock() {
                    if let Ok(mut cfg) = settings.lock() {
                        cfg.base_model = guard.base_model.clone();
                        cfg.acoustic_model = guard.acoustic_model.clone();
                        cfg.voice_sample = guard.voice_sample.clone();
                        cfg.word_spacing_ms = word_spacing_control_copy.value().clamp(0, 100) as u32;
                        cfg.speed_percent = controls.speed_percent();
                        cfg.pitch_semitones = controls.pitch_semitones();
                        cfg.gain_percent = gain_control_copy.value().clamp(MIN_GAIN_PERCENT as i32, MAX_GAIN_PERCENT as i32) as u32;
                        cfg.style_preset = table_key(&STYLE_PRESETS, style_control_copy.get_selection()).to_string();
                        cfg.style_intensity = table_key(&INTENSITIES, intensity_control_copy.get_selection()).to_string();
                        cfg.custom_control = custom_control_copy.get_value();
                        cfg.clone_mode = table_key(&CLONE_MODES, clone_mode_control_copy.get_selection()).to_string();
                        cfg.prompt_text = prompt_text_control_copy.get_value();
                        cfg.variations = variations_control_copy.value().clamp(1, MAX_VARIATIONS as i32) as u32;
                        cfg.cfg_percent = cfg_control_copy.value().clamp(100, 300) as u32;
                        cfg.temperature_percent = temperature_control_copy.value().clamp(50, 150) as u32;
                        cfg.inference_timesteps = timesteps_control_copy.value().clamp(4, 30) as u32;
                        cfg.engine_mode = table_key(&ENGINE_MODES, engine_mode_control_copy.get_selection()).to_string();
                    }
                    let _ = save_shared_settings(&settings);
                    if let Some(path) = guard.playback_file.take() { let _ = fs::remove_file(path); }
                    if guard.owns_server {
                        if let Some(child) = guard.child.as_mut() { let _ = child.kill(); let _ = child.wait(); }
                        guard.child = None;
                    }
                }
                event.skip(true);
            });
        }

        // Start only the server process first. Model selection/loading is then
        // performed through the same public API exposed to any other client.
        {
            let state = Arc::clone(&state_for_app);
            let settings = Arc::clone(&settings_for_app);
            let output = output;
            let speak_button = speak_button;
            let base_button = base_model_button;
            let acoustic_button = acoustic_model_button;
            let load_button = load_models_button;
            let benchmark_button = benchmark_ab_button;
            let profile_button = profile_xtx_button;
            speak_button.enable(false);
            base_button.enable(false);
            acoustic_button.enable(false);
            load_button.enable(false);
            benchmark_button.enable(false);
            profile_button.enable(false);
            thread::spawn(move || {
                let (stream_enabled, default_gain, engine_mode) = settings
                    .lock()
                    .map(|cfg| (cfg.stream, cfg.gain_percent as f32 / 100.0, cfg.engine_mode.clone()))
                    .unwrap_or((true, DEFAULT_GAIN_PERCENT as f32 / 100.0, "normal".to_string()));
                let result = initialize_engine_and_models(&state, stream_enabled, default_gain, &engine_mode);
                let mut warm_message: Option<String> = None;
                if result.is_ok() {
                    let (voice_sample, base_model, acoustic_model) = state.lock()
                        .map(|guard| (guard.voice_sample.clone(), guard.base_model.clone(), guard.acoustic_model.clone()))
                        .unwrap_or((None, None, None));
                    let active_reference = if let Ok(mut cfg) = settings.lock() {
                        cfg.base_model = base_model;
                        cfg.acoustic_model = acoustic_model;
                        let preset = cfg.style_preset.clone();
                        resolve_reference_sample(&cfg, &preset, voice_sample.clone())
                    } else {
                        Err("Settings lock failed; cannot prewarm the voice anchor safely.".to_string())
                    };
                    match active_reference {
                        Ok(resolved) => {
                            if let Some(description) = resolved.describe(
                                &settings.lock().ok().map(|cfg| cfg.style_preset.clone()).unwrap_or_else(|| "auto".to_string())
                            ) {
                                warm_message = Some(description);
                            }
                            if health_check() {
                                if let Some(sample) = resolved.path.as_deref() {
                                    let warmed = match warm_reference_audio(sample) {
                                        Ok(message) => message,
                                        Err(err) => format!("Reference warm-up warning: {err}"),
                                    };
                                    warm_message = Some(match warm_message {
                                        Some(prefix) => format!("{prefix}\n{warmed}"),
                                        None => warmed,
                                    });
                                }
                            }
                        }
                        Err(err) => warm_message = Some(err),
                    }
                    let _ = save_shared_settings(&settings);
                }
                wxdragon::call_after(Box::new(move || {
                    match result {
                        Ok((message, ready)) => {
                            append_log(output, &message);
                            if let Some(warm) = warm_message.as_deref() { append_log(output, warm); }
                            if ready {
                                append_log(output, "Ready. Type text below and click Speak.");
                                speak_button.enable(true);
                            } else {
                                append_log(output, "Select both VoxCPM2 component files and click Load VoxCPM2.");
                            }
                        }
                        Err(err) => {
                            append_log(output, &format!("Engine/model startup error: {err}"));
                            append_log(output, "Select model paths after starting VoxGen v0.7.40 manually, or fix the error and reopen the demo.");
                        }
                    }
                    base_button.enable(true);
                    acoustic_button.enable(true);
                    load_button.enable(true);
                    benchmark_button.enable(true);
                    profile_button.enable(true);
                }));
            });
        }

        frame.show(true);
        frame.centre();
        input.set_focus();
    });
}
