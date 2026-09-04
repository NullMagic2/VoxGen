use crate::{
    gguf::BaseFormat,
    local::CfmOptions,
    runtime::{Runtime, TtsOptions},
    vulkan::{ExecutionMode, XtxTuning},
};
use crate::prosody_control::{
    build_style_control, build_transition_control_with_speed,
    managed_profile_semantics, managed_style_tuning, managed_transition_cfg_delta,
    MoodSpeedTransition, MAX_NATURAL_MOOD_SPEED_DELTA_PERCENT,
    MAX_MOOD_SPEED_PERCENT, MIN_MEANINGFUL_MOOD_SPEED_DELTA_PERCENT,
    MIN_MOOD_SPEED_PERCENT, DEFAULT_MANAGED_INTENSITY, DEFAULT_MANAGED_PROFILE,
    MANAGED_INTENSITIES, MANAGED_PROFILES,
};
use voxgen::playback_dsp::{
    OutputPeakGuard, PlaybackControls, StreamingPlaybackDsp, DEFAULT_PITCH_SEMITONES,
    DEFAULT_SPEED_PERCENT, MAX_PITCH_SEMITONES, MAX_SPEED_PERCENT, MIN_PITCH_SEMITONES,
    MIN_SPEED_PERCENT, OUTPUT_PEAK_CEILING,
};
use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    fs,
    hash::{Hash, Hasher},
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, RwLock,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

static TEMP_ID: AtomicU64 = AtomicU64::new(1);
static DEFAULT_SEED_NONCE: AtomicU64 = AtomicU64::new(1);

const DEFAULT_CONTINUITY_BOUNDARY: &str = "continuous";
const DEFAULT_MANAGED_PACE_PERCENT: f32 = 100.0;
const DEFAULT_TTS_MIN_STEPS: u32 = 2;
const DEFAULT_TTS_MAX_STEPS: u32 = 200;
const DEFAULT_STREAMING_PREFIX_LEN: usize = 6;
const DEFAULT_INFERENCE_TIMESTEPS: u32 = 10;
const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_SWAY_SAMPLING_COEF: f32 = 1.0;
const DEFAULT_CFG_ZERO_STAR: bool = true;
const OUTPUT_SAMPLE_RATE_HZ: u32 = 48_000;
const OUTPUT_CHANNELS: u16 = 1;
const OUTPUT_BITS_PER_SAMPLE: u16 = 32;
const DEFAULT_TRAILING_PAUSE: &str = "none";
const TRAILING_PAUSE_VALUES: &[&str] = &["none", "short", "normal", "long"];
const TRAILING_PAUSE_SHORT_MS: u32 = 150;
const TRAILING_PAUSE_NORMAL_MS: u32 = 350;
const TRAILING_PAUSE_LONG_MS: u32 = 650;

fn default_speech_seed() -> u64 {
    // Seed ownership belongs to VoxGen when the caller omits `seed`.  Mix a
    // monotonic nonce with wall-clock entropy and process identity so separate
    // requests do not silently reuse the historical fixed seed 42.  Explicit
    // caller seeds remain fully deterministic.
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let nonce = DEFAULT_SEED_NONCE.fetch_add(1, Ordering::Relaxed);
    let mut x = time ^ nonce.rotate_left(17) ^ (std::process::id() as u64).rotate_left(33);
    // SplitMix64 finalizer.
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    if x == 0 { 1 } else { x }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeechRequest {
    #[serde(alias = "text")]
    input: String,
    #[serde(default)]
    prompt_text: Option<String>,
    #[serde(default)]
    control: Option<String>,
    #[serde(default)]
    style: Option<String>,
    #[serde(default)]
    intensity: Option<String>,
    #[serde(default)]
    pace_percent: Option<f32>,
    #[serde(default)]
    continuity_id: Option<String>,
    #[serde(default)]
    boundary: Option<String>,
    #[serde(default)]
    pause_after: Option<String>,
    #[serde(default)]
    clone_mode: Option<String>,
    #[serde(default)]
    reference_audio: Option<String>,
    #[serde(default)]
    prompt_audio: Option<String>,
    #[serde(default)]
    reference_audio_path: Option<PathBuf>,
    #[serde(default)]
    prompt_audio_path: Option<PathBuf>,
    #[serde(default)]
    response_format: Option<String>,
    #[serde(default)]
    gain: Option<f32>,
    #[serde(default)]
    speed_percent: Option<f32>,
    #[serde(default)]
    pitch_semitones: Option<f32>,
    #[serde(default)]
    inference_timesteps: Option<u32>,
    #[serde(default)]
    cfg_value: Option<f32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    sway_sampling_coef: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    min_steps: Option<u32>,
    #[serde(default)]
    max_steps: Option<u32>,
    #[serde(default)]
    streaming_prefix_len: Option<usize>,
    #[serde(default)]
    cfg_zero_star: Option<bool>,
    #[serde(default)]
    request_id: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeechSequenceRequest {
    #[serde(default)]
    request_id: Option<u64>,
    segments: Vec<SpeechRequest>,
}

const MAX_SPEECH_SEQUENCE_SEGMENTS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContinuityBoundary {
    Continuous,
    HardCut,
}

impl ContinuityBoundary {
    fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or(DEFAULT_CONTINUITY_BOUNDARY).trim().to_ascii_lowercase().as_str() {
            "continuous" => Ok(Self::Continuous),
            "hard_cut" | "hard-cut" => Ok(Self::HardCut),
            other => bail!("boundary must be continuous or hard_cut, got {other:?}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Continuous => "continuous",
            Self::HardCut => "hard_cut",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrailingPause {
    None,
    Short,
    Normal,
    Long,
}

impl TrailingPause {
    fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or(DEFAULT_TRAILING_PAUSE).trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "short" => Ok(Self::Short),
            "normal" => Ok(Self::Normal),
            "long" => Ok(Self::Long),
            other => bail!("pause_after must be one of {}, got {other:?}", TRAILING_PAUSE_VALUES.join(", ")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Short => "short",
            Self::Normal => "normal",
            Self::Long => "long",
        }
    }

    fn milliseconds(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Short => TRAILING_PAUSE_SHORT_MS,
            Self::Normal => TRAILING_PAUSE_NORMAL_MS,
            Self::Long => TRAILING_PAUSE_LONG_MS,
        }
    }
}

#[derive(Debug, Clone)]
struct ManagedDestination {
    style: String,
    intensity: String,
    requested_pace_percent: f32,
    continuity_id: Option<String>,
    boundary: ContinuityBoundary,
}

#[derive(Debug, Clone)]
struct ContinuityState {
    style: String,
    intensity: String,
    pace_percent: f32,
    speaker_key: String,
    updated_at: Instant,
}

#[derive(Debug, Default)]
struct ContinuityStore {
    sessions: HashMap<String, ContinuityState>,
    id_generations: HashMap<String, u64>,
    global_generation: u64,
}

#[derive(Debug, Clone)]
struct ContinuityPlan {
    destination: ManagedDestination,
    previous: Option<ContinuityState>,
    effective_pace_percent: f32,
    effective_control: String,
    cfg_delta: f32,
    speaker_key: String,
    global_generation: u64,
    id_generation: u64,
}

const CONTINUITY_TTL: Duration = Duration::from_secs(30 * 60);
const MAX_CONTINUITY_SESSIONS: usize = 256;

#[derive(Debug, Deserialize, Default)]
struct SpeechCancelRequest {
    #[serde(default)]
    request_id: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContinuityResetRequest {
    continuity_id: String,
}

/// Server-side model selection. Paths are paths on the machine running VoxGen,
/// not paths on the HTTP client machine.
#[derive(Debug, Deserialize)]
struct ConditioningWarmRequest {
    reference_audio_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct ModelLoadRequest {
    #[serde(alias = "base_lm_path", alias = "voxcpm2_base_lm")]
    base_lm: PathBuf,
    #[serde(default, alias = "acoustic_path", alias = "voxcpm2_acoustic")]
    acoustic: Option<PathBuf>,
    #[serde(default)]
    base_format: Option<String>,
    #[serde(default)]
    gpu: Option<usize>,
    #[serde(default)]
    max_context: Option<u32>,
}

/// Model lifetime is deliberately separate from speech-request lifetime.
/// `inference_gate` serializes model swaps with inference so a reload never
/// overlaps old and new model VRAM allocations.
struct ServerState {
    runtime: RwLock<Option<Arc<Runtime>>>,
    inference_gate: Mutex<()>,
    loading: AtomicBool,
    cancel_speech: AtomicBool,
    active_speech_request: AtomicU64,
    cancel_speech_request: AtomicU64,
    continuity: Mutex<ContinuityStore>,
    streaming_enabled: bool,
    default_gain: f32,
    default_base_format: Option<BaseFormat>,
    default_gpu: Option<usize>,
    default_mode: ExecutionMode,
    default_xtx_tuning: XtxTuning,
    default_max_context: u32,
}

impl ServerState {
    fn new(
        runtime: Option<Arc<Runtime>>,
        default_base_format: Option<BaseFormat>,
        default_gpu: Option<usize>,
        default_mode: ExecutionMode,
        default_xtx_tuning: XtxTuning,
        default_max_context: u32,
        streaming_enabled: bool,
        default_gain: f32,
    ) -> Self {
        Self {
            runtime: RwLock::new(runtime),
            inference_gate: Mutex::new(()),
            loading: AtomicBool::new(false),
            cancel_speech: AtomicBool::new(false),
            active_speech_request: AtomicU64::new(0),
            cancel_speech_request: AtomicU64::new(0),
            continuity: Mutex::new(ContinuityStore::default()),
            streaming_enabled,
            default_gain,
            default_base_format,
            default_gpu,
            default_mode,
            default_xtx_tuning,
            default_max_context,
        }
    }

    fn clear_continuity(&self) -> Result<()> {
        let mut store = self
            .continuity
            .lock()
            .map_err(|_| anyhow::anyhow!("VoxGen continuity state lock is poisoned"))?;
        store.global_generation = store.global_generation.wrapping_add(1);
        store.sessions.clear();
        Ok(())
    }

    fn reset_continuity_id(&self, continuity_id: &str) -> Result<bool> {
        let mut store = self
            .continuity
            .lock()
            .map_err(|_| anyhow::anyhow!("VoxGen continuity state lock is poisoned"))?;
        let removed = store.sessions.remove(continuity_id).is_some();
        let generation = store.id_generations.entry(continuity_id.to_owned()).or_insert(0);
        *generation = generation.wrapping_add(1);
        Ok(removed)
    }

    fn runtime_snapshot(&self) -> Result<Option<Arc<Runtime>>> {
        Ok(self
            .runtime
            .read()
            .map_err(|_| anyhow::anyhow!("VoxGen model state lock is poisoned"))?
            .clone())
    }
}

struct ActiveSpeechRequest {
    state: Arc<ServerState>,
    request_id: u64,
}

impl Drop for ActiveSpeechRequest {
    fn drop(&mut self) {
        if self.request_id != 0 {
            let _ = self.state.active_speech_request.compare_exchange(
                self.request_id,
                0,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
            let _ = self.state.cancel_speech_request.compare_exchange(
                self.request_id,
                0,
                Ordering::AcqRel,
                Ordering::Acquire,
            );
        }
    }
}

fn send_with_headers(mut s: TcpStream, status: &str, content_type: &str, body: &[u8], extra: &[(&str, String)]) -> Result<()> {
    write!(
        s,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\n",
        body.len()
    )?;
    for (name, value) in extra {
        if name.bytes().any(|b| b == b'\r' || b == b'\n') || value.bytes().any(|b| b == b'\r' || b == b'\n') {
            bail!("invalid HTTP response header");
        }
        write!(s, "{name}: {value}\r\n")?;
    }
    s.write_all(b"\r\n")?;
    s.write_all(body)?;
    Ok(())
}

fn send(s: TcpStream, status: &str, content_type: &str, body: &[u8]) -> Result<()> {
    send_with_headers(s, status, content_type, body, &[])
}

fn send_json(s: TcpStream, status: &str, value: Value) -> Result<()> {
    let body = serde_json::to_vec(&value)?;
    send(s, status, "application/json; charset=utf-8", &body)
}

fn write_chunk(s: &mut TcpStream, body: &[u8]) -> Result<()> {
    write!(s, "{:X}\r\n", body.len())?;
    s.write_all(body)?;
    s.write_all(b"\r\n")?;
    Ok(())
}

fn read_request(s: &mut TcpStream) -> Result<(String, String, Vec<u8>)> {
    let mut all = Vec::with_capacity(8192);
    let mut tmp = [0u8; 8192];
    let header_end;
    loop {
        let n = s.read(&mut tmp).context("read HTTP request")?;
        if n == 0 {
            bail!("connection closed before complete HTTP headers");
        }
        all.extend_from_slice(&tmp[..n]);
        if let Some(i) = find_bytes(&all, b"\r\n\r\n") {
            header_end = i + 4;
            break;
        }
        if all.len() > 1024 * 1024 {
            bail!("HTTP headers exceed 1 MiB");
        }
    }
    let hdr = std::str::from_utf8(&all[..header_end]).context("HTTP headers are not UTF-8")?;
    let first = hdr.lines().next().unwrap_or("");
    let mut p = first.split_whitespace();
    let method = p.next().unwrap_or("").to_owned();
    let path = p.next().unwrap_or("").to_owned();
    let mut content_len = 0usize;
    for line in hdr.lines().skip(1) {
        if let Some((k, v)) = line.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_len = v.trim().parse().context("invalid Content-Length")?;
            }
        }
    }
    if content_len > 64 * 1024 * 1024 {
        bail!("HTTP request body exceeds 64 MiB");
    }
    while all.len() - header_end < content_len {
        let n = s.read(&mut tmp)?;
        if n == 0 {
            bail!("connection closed before Content-Length body completed");
        }
        all.extend_from_slice(&tmp[..n]);
    }
    Ok((
        method,
        path,
        all[header_end..header_end + content_len].to_vec(),
    ))
}

fn find_bytes(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).position(|x| x == n)
}

struct Temps(Vec<PathBuf>);
impl Drop for Temps {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = fs::remove_file(p);
        }
    }
}

fn audio_path(
    encoded: &Option<String>,
    path: &Option<PathBuf>,
    label: &str,
    temps: &mut Temps,
) -> Result<Option<PathBuf>> {
    if encoded.is_some() && path.is_some() {
        bail!("provide either {label}_audio base64 or {label}_audio_path, not both");
    }
    if let Some(p) = path {
        return Ok(Some(p.clone()));
    }
    let Some(raw) = encoded else {
        return Ok(None);
    };
    let raw = raw.trim();
    let raw = raw
        .split_once(",base64,")
        .map(|(_, b)| b)
        .unwrap_or(raw);
    let bytes = BASE64
        .decode(raw)
        .with_context(|| format!("invalid base64 in {label}_audio"))?;
    let p = std::env::temp_dir().join(format!(
        "voxgen-{}-{}-{}.wav",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed),
        label
    ));
    fs::write(&p, bytes)?;
    temps.0.push(p.clone());
    Ok(Some(p))
}

fn normalize_style(value: &str) -> Result<String> {
    let normalized = match value.trim().to_ascii_lowercase().as_str() {
        "neutral" => "neutral",
        "warm" => "warm",
        "cheerful" => "cheerful",
        "excited" => "excited",
        "sad" => "sad",
        "concerned" => "concerned",
        "angry" => "angry",
        "gentle" => "gentle",
        "serious" => "serious",
        "whisper" => "whisper",
        other => bail!("unsupported style {other:?}; use neutral, warm, cheerful, excited, sad, concerned, angry, gentle, serious, or whisper"),
    };
    Ok(normalized.to_owned())
}

fn normalize_intensity(value: Option<&str>) -> Result<String> {
    let normalized = match value.unwrap_or("normal").trim().to_ascii_lowercase().as_str() {
        "subtle" => "subtle",
        "normal" => "normal",
        "strong" => "strong",
        other => bail!("unsupported intensity {other:?}; use subtle, normal, or strong"),
    };
    Ok(normalized.to_owned())
}

fn parse_managed_destination(req: &SpeechRequest) -> Result<Option<ManagedDestination>> {
    let has_control = req.control.as_deref().is_some_and(|x| !x.trim().is_empty());
    if req.style.is_some() && has_control {
        bail!("style cannot be combined with control");
    }
    if req.style.is_none() {
        if req.intensity.is_some() || req.pace_percent.is_some() || req.continuity_id.is_some() || req.boundary.is_some() {
            bail!("intensity, pace_percent, continuity_id, and boundary require style");
        }
        return Ok(None);
    }

    let style = normalize_style(req.style.as_deref().unwrap_or_default())?;
    let intensity = normalize_intensity(req.intensity.as_deref())?;
    let pace_percent = req.pace_percent.unwrap_or(100.0);
    if !pace_percent.is_finite() || !(MIN_MOOD_SPEED_PERCENT..=MAX_MOOD_SPEED_PERCENT).contains(&pace_percent) {
        bail!("pace_percent must be finite and between {MIN_MOOD_SPEED_PERCENT:.0} and {MAX_MOOD_SPEED_PERCENT:.0}");
    }
    let continuity_id = req.continuity_id.as_deref().map(str::trim).filter(|x| !x.is_empty()).map(str::to_owned);
    if req.continuity_id.is_some() && continuity_id.is_none() {
        bail!("continuity_id cannot be empty");
    }
    if continuity_id.as_deref().is_some_and(|id| id.len() > 128) {
        bail!("continuity_id must be at most 128 bytes");
    }
    if req.boundary.is_some() && continuity_id.is_none() {
        bail!("boundary requires continuity_id");
    }
    let boundary = ContinuityBoundary::parse(req.boundary.as_deref())?;
    if continuity_id.is_some() && (req.speed_percent.unwrap_or(100.0) - 100.0).abs() > 1.0e-4 {
        bail!("speed_percent must remain 100 while continuity_id is active; use pace_percent so VoxGen owns pace continuity");
    }
    if (pace_percent - 100.0).abs() > 1.0e-4 && (req.speed_percent.unwrap_or(100.0) - 100.0).abs() > 1.0e-4 {
        bail!("speed_percent and managed pace_percent cannot both change speaking rate; leave speed_percent at 100");
    }
    Ok(Some(ManagedDestination { style, intensity, requested_pace_percent: pace_percent, continuity_id, boundary }))
}

fn validate_speech_request(req: &SpeechRequest) -> Result<()> {
    if req.input.trim().is_empty() {
        bail!("speech input is empty");
    }
    let has_reference = req.reference_audio.as_deref().is_some_and(|x| !x.trim().is_empty())
        || req.reference_audio_path.is_some();
    let has_prompt = req.prompt_audio.as_deref().is_some_and(|x| !x.trim().is_empty())
        || req.prompt_audio_path.is_some();
    let has_prompt_text = req.prompt_text.as_deref().is_some_and(|x| !x.trim().is_empty());
    let clone_mode = req.clone_mode.as_deref().unwrap_or("auto").trim().to_ascii_lowercase();
    match clone_mode.as_str() {
        "auto" => {}
        "reference" | "controllable" => {
            if !has_reference { bail!("clone_mode=reference requires reference audio"); }
            if has_prompt || has_prompt_text { bail!("clone_mode=reference cannot use prompt audio/text"); }
        }
        "ultimate" => {
            if !has_prompt_text { bail!("clone_mode=ultimate requires prompt_text with the exact reference transcript"); }
            if !has_prompt && !has_reference { bail!("clone_mode=ultimate requires prompt or reference audio"); }
        }
        _ => bail!("clone_mode must be auto, reference/controllable, or ultimate"),
    }
    let managed = parse_managed_destination(req)?;
    let _ = TrailingPause::parse(req.pause_after.as_deref())?;
    if (req.control.as_deref().is_some_and(|x| !x.trim().is_empty()) || managed.is_some())
        && (has_prompt || has_prompt_text || clone_mode == "ultimate")
    {
        bail!("control/style cannot be combined with ultimate/prompt continuation cloning");
    }
    match req.response_format.as_deref().unwrap_or("wav") {
        "wav" | "pcm" | "f32" => {}
        x => bail!("unsupported response_format {x:?}; use wav or pcm"),
    }
    Ok(())
}

fn parse_speech_request(body: &[u8]) -> Result<SpeechRequest> {
    let req: SpeechRequest = serde_json::from_slice(body).context("parse speech request JSON")?;
    validate_speech_request(&req)?;
    Ok(req)
}

fn parse_speech_sequence_request(body: &[u8]) -> Result<SpeechSequenceRequest> {
    let req: SpeechSequenceRequest = serde_json::from_slice(body).context("parse speech sequence request JSON")?;
    if req.segments.is_empty() {
        bail!("speech sequence must contain at least one segment");
    }
    if req.segments.len() > MAX_SPEECH_SEQUENCE_SEGMENTS {
        bail!("speech sequence may contain at most {MAX_SPEECH_SEQUENCE_SEGMENTS} segments");
    }
    for (index, segment) in req.segments.iter().enumerate() {
        validate_speech_request(segment)
            .with_context(|| format!("invalid speech sequence segment {}", index + 1))?;
        if segment.request_id.is_some() {
            bail!("speech sequence segment {} contains request_id; use the sequence-level request_id instead", index + 1);
        }
    }
    Ok(req)
}

fn speaker_conditioning_key(req: &SpeechRequest) -> String {
    let mut hasher = DefaultHasher::new();
    if let Some(raw) = req.reference_audio.as_deref().filter(|x| !x.trim().is_empty()) {
        "inline-reference".hash(&mut hasher);
        raw.hash(&mut hasher);
        return format!("inline:{:016x}", hasher.finish());
    }
    if let Some(path) = req.reference_audio_path.as_ref() {
        "path-reference".hash(&mut hasher);
        let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.clone());
        canonical.to_string_lossy().hash(&mut hasher);
        if let Ok(metadata) = fs::metadata(&canonical) {
            metadata.len().hash(&mut hasher);
            if let Ok(modified) = metadata.modified() { modified.hash(&mut hasher); }
        }
        return format!("path:{:016x}", hasher.finish());
    }
    "none".to_owned()
}

fn append_pace_hold(control: &mut String, pace_percent: f32) {
    if (pace_percent - 100.0).abs() >= 0.001 {
        control.push_str(&format!(" Speaking pace: {:.0}%.", pace_percent));
    }
}

fn destination_control(destination: &ManagedDestination, pace_percent: f32, text: &str) -> Result<(String, f32)> {
    let mut effective = build_style_control(&destination.style, &destination.intensity, "", text)
        .ok_or_else(|| anyhow::anyhow!("unsupported style/intensity combination"))?;
    append_pace_hold(&mut effective, pace_percent);
    let tuning = managed_style_tuning(&destination.style, &destination.intensity)
        .ok_or_else(|| anyhow::anyhow!("unsupported style/intensity combination"))?;
    Ok((effective, tuning.cfg_delta))
}

fn continuity_plan(state: &ServerState, req: &SpeechRequest, destination: ManagedDestination, speaker_key: String) -> Result<ContinuityPlan> {
    let Some(continuity_id) = destination.continuity_id.as_deref() else {
        let effective_pace_percent = destination.requested_pace_percent;
        let (effective_control, cfg_delta) = destination_control(&destination, effective_pace_percent, &req.input)?;
        return Ok(ContinuityPlan {
            destination,
            previous: None,
            effective_pace_percent,
            effective_control,
            cfg_delta,
            speaker_key,
            global_generation: 0,
            id_generation: 0,
        });
    };

    let mut store = state
        .continuity
        .lock()
        .map_err(|_| anyhow::anyhow!("VoxGen continuity state lock is poisoned"))?;
    let now = Instant::now();
    store.sessions.retain(|_, session| now.duration_since(session.updated_at) <= CONTINUITY_TTL);
    let global_generation = store.global_generation;
    let id_generation = *store.id_generations.get(continuity_id).unwrap_or(&0);
    let previous = store.sessions.get(continuity_id)
        .filter(|prior| prior.speaker_key == speaker_key)
        .cloned();
    drop(store);

    let effective_pace_percent = if let Some(prior) = previous.as_ref() {
        let delta = destination.requested_pace_percent - prior.pace_percent;
        let magnitude = delta.abs();
        if magnitude < MIN_MEANINGFUL_MOOD_SPEED_DELTA_PERCENT {
            prior.pace_percent
        } else if magnitude > MAX_NATURAL_MOOD_SPEED_DELTA_PERCENT {
            prior.pace_percent + delta.signum() * MAX_NATURAL_MOOD_SPEED_DELTA_PERCENT
        } else {
            destination.requested_pace_percent
        }
    } else {
        destination.requested_pace_percent
    };

    let (effective_control, cfg_delta) = if let Some(prior) = previous.as_ref() {
        let style_changed = prior.style != destination.style;
        let intensity_changed = prior.intensity != destination.intensity;
        let pace_changed = (prior.pace_percent - effective_pace_percent).abs() >= MIN_MEANINGFUL_MOOD_SPEED_DELTA_PERCENT;
        let keep_nondefault_pace = effective_pace_percent != 100.0;
        let should_transition = intensity_changed || pace_changed || (style_changed && destination.boundary == ContinuityBoundary::Continuous);

        if should_transition {
            let (from_style, from_intensity) = if style_changed && destination.boundary == ContinuityBoundary::HardCut {
                (destination.style.as_str(), prior.intensity.as_str())
            } else {
                (prior.style.as_str(), prior.intensity.as_str())
            };
            let speed = if pace_changed || keep_nondefault_pace {
                Some(MoodSpeedTransition::new(prior.pace_percent, effective_pace_percent).map_err(anyhow::Error::msg)?)
            } else {
                None
            };
            let control = build_transition_control_with_speed(
                from_style,
                from_intensity,
                &destination.style,
                &destination.intensity,
                speed,
                &req.input,
            ).map_err(anyhow::Error::msg)?;
            let delta = managed_transition_cfg_delta(
                from_style,
                from_intensity,
                &destination.style,
                &destination.intensity,
            ).map_err(anyhow::Error::msg)?;
            (control, delta)
        } else {
            destination_control(&destination, effective_pace_percent, &req.input)?
        }
    } else {
        destination_control(&destination, effective_pace_percent, &req.input)?
    };

    Ok(ContinuityPlan {
        destination,
        previous,
        effective_pace_percent,
        effective_control,
        cfg_delta,
        speaker_key,
        global_generation,
        id_generation,
    })
}

fn commit_continuity(state: &ServerState, plan: &ContinuityPlan) -> Result<bool> {
    let Some(continuity_id) = plan.destination.continuity_id.as_deref() else { return Ok(false); };
    let mut store = state
        .continuity
        .lock()
        .map_err(|_| anyhow::anyhow!("VoxGen continuity state lock is poisoned"))?;
    if store.global_generation != plan.global_generation
        || *store.id_generations.get(continuity_id).unwrap_or(&0) != plan.id_generation
    {
        return Ok(false);
    }
    let now = Instant::now();
    store.sessions.retain(|_, session| now.duration_since(session.updated_at) <= CONTINUITY_TTL);
    if !store.sessions.contains_key(continuity_id) && store.sessions.len() >= MAX_CONTINUITY_SESSIONS {
        if let Some(oldest) = store.sessions.iter().min_by_key(|(_, session)| session.updated_at).map(|(id, _)| id.clone()) {
            store.sessions.remove(&oldest);
        }
    }
    store.sessions.insert(continuity_id.to_owned(), ContinuityState {
        style: plan.destination.style.clone(),
        intensity: plan.destination.intensity.clone(),
        pace_percent: plan.effective_pace_percent,
        speaker_key: plan.speaker_key.clone(),
        updated_at: now,
    });
    Ok(true)
}

fn continuity_headers(plan: Option<&ContinuityPlan>) -> Vec<(&'static str, String)> {
    let Some(plan) = plan else { return Vec::new(); };
    let previous = plan.previous.as_ref();
    vec![
        ("X-VoxGen-Previous-Style", previous.map(|x| x.style.clone()).unwrap_or_else(|| "none".to_owned())),
        ("X-VoxGen-Effective-Style", plan.destination.style.clone()),
        ("X-VoxGen-Previous-Intensity", previous.map(|x| x.intensity.clone()).unwrap_or_else(|| "none".to_owned())),
        ("X-VoxGen-Effective-Intensity", plan.destination.intensity.clone()),
        ("X-VoxGen-Previous-Pace-Percent", previous.map(|x| format!("{:.3}", x.pace_percent)).unwrap_or_else(|| "none".to_owned())),
        ("X-VoxGen-Effective-Pace-Percent", format!("{:.3}", plan.effective_pace_percent)),
        ("X-VoxGen-Requested-Pace-Percent", format!("{:.3}", plan.destination.requested_pace_percent)),
        ("X-VoxGen-Boundary", plan.destination.boundary.as_str().to_owned()),
    ]
}

fn options(runtime: &Runtime, r: &SpeechRequest, managed_cfg_delta: Option<f32>) -> Result<TtsOptions> {
    let cfg = runtime.default_cfm_cfg_rate();
    let cfg_value = match r.cfg_value {
        Some(explicit) => explicit,
        None => match managed_cfg_delta {
            Some(delta) => (cfg + delta).clamp(1.0, 3.0),
            None => cfg,
        },
    };
    Ok(TtsOptions {
        min_steps: r.min_steps.unwrap_or(DEFAULT_TTS_MIN_STEPS),
        max_steps: r.max_steps.unwrap_or(DEFAULT_TTS_MAX_STEPS),
        streaming_prefix_len: r.streaming_prefix_len.unwrap_or(DEFAULT_STREAMING_PREFIX_LEN),
        cfm: CfmOptions {
            n_timesteps: r.inference_timesteps.unwrap_or(DEFAULT_INFERENCE_TIMESTEPS),
            cfg_value,
            temperature: r.temperature.unwrap_or(DEFAULT_TEMPERATURE),
            sway_sampling_coef: r.sway_sampling_coef.unwrap_or(DEFAULT_SWAY_SAMPLING_COEF),
            seed: r.seed.unwrap_or_else(default_speech_seed),
            use_cfg_zero_star: r.cfg_zero_star.unwrap_or(DEFAULT_CFG_ZERO_STAR),
        },
    })
}

fn wav_header_f32(sample_rate: u32, data_bytes: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(44);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&data_bytes.saturating_add(36).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&3u16.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&sample_rate.to_le_bytes());
    v.extend_from_slice(&(sample_rate * 4).to_le_bytes());
    v.extend_from_slice(&4u16.to_le_bytes());
    v.extend_from_slice(&32u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_bytes.to_le_bytes());
    v
}

fn pcm_bytes(x: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(x.len() * 4);
    for &v in x {
        let safe = if v.is_finite() { v } else { 0.0 };
        b.extend_from_slice(&safe.clamp(-OUTPUT_PEAK_CEILING, OUTPUT_PEAK_CEILING).to_le_bytes());
    }
    b
}

fn wav_bytes(x: &[f32]) -> Vec<u8> {
    let pcm = pcm_bytes(x);
    let mut b = wav_header_f32(OUTPUT_SAMPLE_RATE_HZ, pcm.len().min(u32::MAX as usize) as u32);
    b.extend_from_slice(&pcm);
    b
}

fn parse_base_format(value: Option<&str>, fallback: Option<BaseFormat>) -> Result<Option<BaseFormat>> {
    match value.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(fallback),
        Some("auto") => Ok(None),
        Some("q8_0") | Some("q8-0") | Some("Q8_0") => Ok(Some(BaseFormat::Q8_0)),
        Some("f16") | Some("F16") => Ok(Some(BaseFormat::F16)),
        Some(other) => bail!("unsupported base_format {other:?}; use auto, q8_0, or f16"),
    }
}

fn model_json(state: &ServerState) -> Result<Value> {
    let loading = state.loading.load(Ordering::Acquire);
    // Hold the read guard while serializing status instead of cloning the Arc.
    // A model reload needs the write lock before it can remove/drop the old
    // runtime, so even concurrent status requests cannot accidentally keep the
    // previous multi-GiB runtime alive while the replacement is allocated.
    let slot = state
        .runtime
        .read()
        .map_err(|_| anyhow::anyhow!("VoxGen model state lock is poisoned"))?;
    if let Some(runtime) = slot.as_ref() {
        let st = runtime.status();
        Ok(json!({
            "loaded": true,
            "loading": loading,
            "speech_inference_ready": st.speech_inference_ready,
            "base_lm": st.base_lm.path.clone(),
            "acoustic": st.acoustic.map(|a| a.path.clone()),
            "base_format": st.base_format.as_str(),
            "gpu": {"index": st.gpu.index, "name": st.gpu.name.clone()},
            "mode": st.execution_mode,
            "max_context": st.baselm.config.active_context_length,
            "runtime": st
        }))
    } else {
        Ok(json!({
            "loaded": false,
            "loading": loading,
            "speech_inference_ready": false,
            "base_lm": null,
            "acoustic": null,
            "base_format": null,
            "mode": state.default_mode.as_str()
        }))
    }
}

fn health_json(state: &ServerState) -> Result<Value> {
    let current = model_json(state)?;
    let profile_semantics = MANAGED_PROFILES
        .iter()
        .filter_map(|profile| managed_profile_semantics(profile).map(|semantics| {
            ((*profile).to_owned(), Value::String(semantics.to_owned()))
        }))
        .collect::<serde_json::Map<String, Value>>();
    Ok(json!({
        "ok": true,
        "engine": "VoxGen",
        "version": env!("CARGO_PKG_VERSION"),
        "pid": std::process::id(),
        "iteration": 7,
        "model_loaded": current.get("loaded").and_then(Value::as_bool).unwrap_or(false),
        "model_loading": current.get("loading").and_then(Value::as_bool).unwrap_or(false),
        "speech_inference_ready": current.get("speech_inference_ready").and_then(Value::as_bool).unwrap_or(false),
        "streaming_enabled": state.streaming_enabled,
        "default_gain": state.default_gain,
        "native_playback_dsp": true,
        "native_managed_prosody": true,
        "managed_prosody": {
            "version": 12,
            "profiles": MANAGED_PROFILES,
            "default_profile": DEFAULT_MANAGED_PROFILE,
            "intensities": MANAGED_INTENSITIES,
            "default_intensity": DEFAULT_MANAGED_INTENSITY,
            "profile_semantics": profile_semantics,
            "subtle_positive_cue_floor": true,
            "managed_cfg_guidance": {"warm_delta": 0.20, "gentle_subtle_delta": 0.10, "gentle_normal_strong_delta": 0.15, "subtle_cheerful_delta": 0.10, "concerned_delta": 0.10, "whisper_delta": 0.10, "sad_delta": 0.10, "serious_subtle_delta": 0.05, "serious_normal_strong_delta": 0.10, "excited_delta": 0.0, "angry_delta": 0.0, "explicit_cfg_preserved": true},
            "short_utterance_guard": true,
            "custom_controls_preserved": true,
            "automatic_continuity": {
                "enabled": true,
                "request_model": "destination-only",
                "explicit_transition_api": false,
                "boundaries": ["continuous", "hard_cut"],
                "default_boundary": DEFAULT_CONTINUITY_BOUNDARY,
                "boundary_semantics": {
                    "continuous": "preserve-style-intensity-and-pace-continuity-from-the-last-successful-state",
                    "hard_cut": "style-may-cut-intensity-and-pace-still-smooth"
                },
                "single_pass_synthesis": true,
                "waveform_crossfade": false,
                "state_ttl_seconds": CONTINUITY_TTL.as_secs(),
                "max_active_sessions": MAX_CONTINUITY_SESSIONS,
                "state_commit": "successful-synthesis-only",
                "speaker_conditioning_scoped": true,
                "pace": {
                    "min_percent": MIN_MOOD_SPEED_PERCENT,
                    "max_percent": MAX_MOOD_SPEED_PERCENT,
                    "default_percent": DEFAULT_MANAGED_PACE_PERCENT,
                    "suppress_delta_below_percent_points": MIN_MEANINGFUL_MOOD_SPEED_DELTA_PERCENT,
                    "max_advance_per_phrase_percent_points": MAX_NATURAL_MOOD_SPEED_DELTA_PERCENT,
                    "realization": "single-pass-prosody-conditioning-not-midstream-wsola",
                    "continuity_playback_speed_requirement": DEFAULT_SPEED_PERCENT
                },
                "reset_endpoint": "/v1/audio/continuity/reset"
            }
        },
        "playback_dsp": {
            "algorithm": "sinc+speech-wsola-ncc-confidence+peak-guard",
            "overlap_crossfade": "raised-cosine-amplitude-complementary",
            "wsola_low_confidence_fallback": "predicted-analysis-position",
            "internal_clipping": false,
            "output_peak_guard": {"enabled": true, "sample_peak_ceiling": OUTPUT_PEAK_CEILING, "stream_release_ms": 250.0, "uniform_offline_attenuation": true},
            "speed_percent": {"min": MIN_SPEED_PERCENT, "max": MAX_SPEED_PERCENT, "default": DEFAULT_SPEED_PERCENT},
            "pitch_semitones": {"min": MIN_PITCH_SEMITONES, "max": MAX_PITCH_SEMITONES, "default": DEFAULT_PITCH_SEMITONES}
        },
        "trailing_pause": {
            "field": "pause_after",
            "values": TRAILING_PAUSE_VALUES,
            "default": DEFAULT_TRAILING_PAUSE,
            "semantics": {
                "none": "no additional engine-appended silence after the synthesized phrase",
                "short": "brief phrase-separation pause",
                "normal": "ordinary sentence-level phrase-separation pause",
                "long": "deliberate paragraph-or-turn-level phrase-separation pause"
            },
            "realization": "engine-owned-trailing-silence-after-successful-synthesis"
        },
        "sequence_synthesis": {
            "enabled": true,
            "stream_endpoint": "/v1/audio/speech/sequence/stream",
            "request_model": "ordered-semantic-plan-segments",
            "max_segments": MAX_SPEECH_SEQUENCE_SEGMENTS,
            "transport": "single-http-stream",
            "segment_semantics": "ordered-semantic-plan-entries; compatible-steady-state-neighbors-may-share-one-acoustic-generation-run",
            "execution_compiler": "adjacent-compatible-steady-state-coalescing",
            "physical_run_semantics": "one-model-generation-per-compiled-acoustic-run",
            "interior_pause_realization": "none-short-normal-use-engine-control-guided-in-utterance-boundaries; long-forces-a-physical-run-boundary; terminal-pause-remains-engine-appended-silence",
            "delivery": "progressive-pcm-continuous-sequence-writer",
            "continuity_commit": "after-successful-acoustic-run; coalesced-semantic-entries-commit-atomically-to-the-run-final-state",
            "cancellation": "sequence-level-request-id"
        },
        "speech_request": {
            "defaults": {
                "response_format": "wav",
                "clone_mode": "auto",
                "gain": state.default_gain,
                "inference_timesteps": DEFAULT_INFERENCE_TIMESTEPS,
                "temperature": DEFAULT_TEMPERATURE,
                "sway_sampling_coef": DEFAULT_SWAY_SAMPLING_COEF,
                "min_steps": DEFAULT_TTS_MIN_STEPS,
                "max_steps": DEFAULT_TTS_MAX_STEPS,
                "streaming_prefix_len": DEFAULT_STREAMING_PREFIX_LEN,
                "cfg_zero_star": DEFAULT_CFG_ZERO_STAR,
                "pause_after": DEFAULT_TRAILING_PAUSE,
                "seed_policy": "random-per-request-when-omitted"
            },
            "constraints": {
                "gain": {"min": 0.0, "finite": true},
                "inference_timesteps": {"min": 1},
                "temperature": {"min": 0.0, "finite": true},
                "cfg_value": {"finite": true},
                "sway_sampling_coef": {"finite": true},
                "min_steps": {"min": 0},
                "max_steps": {"min": 1},
                "streaming_prefix_len": {"min": 1}
            }
        },
        "output_audio": {
            "streaming": {
                "container": "wav",
                "sample_format": "float32_le",
                "bits_per_sample": OUTPUT_BITS_PER_SAMPLE,
                "channels": OUTPUT_CHANNELS,
                "sample_rate_hz": OUTPUT_SAMPLE_RATE_HZ
            },
            "non_streaming_formats": ["wav", "pcm", "f32"]
        },
        "execution": {
            "supported_modes": [
                {"id": "normal", "label": "Normal", "description": "portable Vulkan compute path"},
                {"id": "xtx7900", "label": "XTX 7900", "description": "RX 7900 XTX stream-safe tuned Vulkan path"}
            ],
            "default_mode": state.default_mode.as_str(),
            "active_mode": current.get("mode").cloned().unwrap_or_else(|| json!(state.default_mode.as_str()))
        },
        "mode": current.get("mode").cloned().unwrap_or_else(|| json!(state.default_mode.as_str())),
        "gpu_profile": state.default_xtx_tuning.gpu_profile,
        "benchmark_profile": state.default_mode == ExecutionMode::Xtx7900
            && state.default_xtx_tuning.gpu_profile
            && !state.streaming_enabled,
        "xtx_coopmat": state.default_xtx_tuning.cooperative_matrix,
        "model": current
    }))
}

fn load_model(state: &ServerState, body: &[u8]) -> Result<Value> {
    let request: ModelLoadRequest = serde_json::from_slice(body).context("parse model load JSON")?;
    if !request.base_lm.is_file() {
        bail!("BaseLM GGUF does not exist: {}", request.base_lm.display());
    }
    if let Some(path) = request.acoustic.as_ref() {
        if !path.is_file() {
            bail!("acoustic GGUF does not exist: {}", path.display());
        }
    }

    let requested = parse_base_format(
        request.base_format.as_deref(),
        state.default_base_format,
    )?;
    let gpu = request.gpu.or(state.default_gpu);
    let max_context = request.max_context.unwrap_or(state.default_max_context);
    if max_context == 0 {
        bail!("max_context must be >= 1");
    }

    // Never allocate the replacement runtime while the previous one is still
    // active: an F16 BaseLM swap can otherwise transiently require nearly 2x VRAM.
    let _inference = state
        .inference_gate
        .lock()
        .map_err(|_| anyhow::anyhow!("VoxGen inference/model-load lock is poisoned"))?;
    state.loading.store(true, Ordering::Release);

    let old = {
        let mut slot = state
            .runtime
            .write()
            .map_err(|_| anyhow::anyhow!("VoxGen model state lock is poisoned"))?;
        slot.take()
    };
    drop(old);
    state.clear_continuity()?;

    let loaded = Runtime::load(
        &request.base_lm,
        request.acoustic.as_deref(),
        requested,
        gpu,
        state.default_mode,
        state.default_xtx_tuning,
        max_context,
    );

    match loaded {
        Ok(runtime) => {
            let runtime = Arc::new(runtime);
            {
                let mut slot = state
                    .runtime
                    .write()
                    .map_err(|_| anyhow::anyhow!("VoxGen model state lock is poisoned"))?;
                *slot = Some(runtime);
            }
            state.loading.store(false, Ordering::Release);
            model_json(state)
        }
        Err(err) => {
            state.loading.store(false, Ordering::Release);
            Err(err)
        }
    }
}

fn unload_model(state: &ServerState) -> Result<Value> {
    let _inference = state
        .inference_gate
        .lock()
        .map_err(|_| anyhow::anyhow!("VoxGen inference/model-load lock is poisoned"))?;
    let old = {
        let mut slot = state
            .runtime
            .write()
            .map_err(|_| anyhow::anyhow!("VoxGen model state lock is poisoned"))?;
        slot.take()
    };
    drop(old);
    state.clear_continuity()?;
    Ok(json!({"ok": true, "loaded": false, "speech_inference_ready": false}))
}

fn runtime_or_503(s: TcpStream, state: &ServerState) -> Result<Option<(TcpStream, Arc<Runtime>)>> {
    let runtime = state.runtime_snapshot()?;
    match runtime {
        Some(runtime) => Ok(Some((s, runtime))),
        None => {
            send_json(
                s,
                "503 Service Unavailable",
                json!({
                    "error": "no VoxGen model is loaded",
                    "hint": "POST /v1/models/load with server-side base_lm and acoustic GGUF paths"
                }),
            )?;
            Ok(None)
        }
    }
}

fn resolve_speech_audio_paths(
    req: &SpeechRequest,
    effective_control: Option<&str>,
    temps: &mut Temps,
) -> Result<(Option<PathBuf>, Option<PathBuf>)> {
    let ref_path = audio_path(
        &req.reference_audio,
        &req.reference_audio_path,
        "reference",
        temps,
    )?;
    let mut prompt_path = audio_path(
        &req.prompt_audio,
        &req.prompt_audio_path,
        "prompt",
        temps,
    )?;
    let mut ref_path = ref_path;
    let clone_mode = req.clone_mode.as_deref().unwrap_or("auto").trim().to_ascii_lowercase();
    match clone_mode.as_str() {
        "auto" => {}
        "reference" | "controllable" => {
            if ref_path.is_none() { bail!("clone_mode=reference requires reference audio"); }
            if prompt_path.is_some() || req.prompt_text.as_deref().is_some_and(|x|!x.trim().is_empty()) {
                bail!("clone_mode=reference cannot use prompt audio/text");
            }
        }
        "ultimate" => {
            if req.prompt_text.as_deref().map(str::trim).unwrap_or("").is_empty() {
                bail!("clone_mode=ultimate requires prompt_text with the exact reference transcript");
            }
            let source = prompt_path.clone().or_else(|| ref_path.clone())
                .ok_or_else(|| anyhow::anyhow!("clone_mode=ultimate requires prompt or reference audio"))?;
            if prompt_path.is_none() { prompt_path = Some(source.clone()); }
            if ref_path.is_none() { ref_path = Some(source); }
        }
        _ => bail!("clone_mode must be auto, reference/controllable, or ultimate"),
    }
    if effective_control.is_some_and(|x| !x.trim().is_empty()) && prompt_path.is_some() {
        bail!("control/style cannot be combined with ultimate/prompt continuation cloning");
    }
    Ok((ref_path, prompt_path))
}

fn optional_f32_eq(a: Option<f32>, b: Option<f32>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => (x - y).abs() <= 1.0e-4,
        _ => false,
    }
}

fn same_managed_destination(a: &ManagedDestination, b: &ManagedDestination) -> bool {
    a.style == b.style
        && a.intensity == b.intensity
        && (a.requested_pace_percent - b.requested_pace_percent).abs() <= 1.0e-4
        && a.continuity_id == b.continuity_id
}

fn continuity_plan_is_steady(plan: &ContinuityPlan) -> bool {
    let Some(previous) = plan.previous.as_ref() else { return true; };
    let style_changed = previous.style != plan.destination.style;
    let intensity_changed = previous.intensity != plan.destination.intensity;
    let pace_changed = (previous.pace_percent - plan.effective_pace_percent).abs()
        >= MIN_MEANINGFUL_MOOD_SPEED_DELTA_PERCENT;
    !(intensity_changed
        || pace_changed
        || (style_changed && plan.destination.boundary == ContinuityBoundary::Continuous))
}

fn same_sequence_run_settings(a: &SpeechRequest, b: &SpeechRequest) -> bool {
    // The semantic destination, boundary, pause intent, and input text are handled
    // separately by the sequence compiler. Everything below changes conditioning,
    // decoding, playback DSP, or reproducibility and therefore must match before two
    // semantic plan entries can share one physical model generation.
    a.prompt_text == b.prompt_text
        && a.control == b.control
        && a.clone_mode == b.clone_mode
        && a.reference_audio == b.reference_audio
        && a.prompt_audio == b.prompt_audio
        && a.reference_audio_path == b.reference_audio_path
        && a.prompt_audio_path == b.prompt_audio_path
        && a.response_format == b.response_format
        && optional_f32_eq(a.gain, b.gain)
        && optional_f32_eq(a.speed_percent, b.speed_percent)
        && optional_f32_eq(a.pitch_semitones, b.pitch_semitones)
        && a.inference_timesteps == b.inference_timesteps
        && optional_f32_eq(a.cfg_value, b.cfg_value)
        && optional_f32_eq(a.temperature, b.temperature)
        && optional_f32_eq(a.sway_sampling_coef, b.sway_sampling_coef)
        // Low-level segment-local generation-window overrides are deliberately
        // not absorbed into a larger run. DD leaves them omitted, so the normal
        // planner path still coalesces freely while explicit expert requests keep
        // their original per-segment semantics.
        && a.min_steps.is_none()
        && b.min_steps.is_none()
        && a.max_steps.is_none()
        && b.max_steps.is_none()
        && a.streaming_prefix_len.is_none()
        && b.streaming_prefix_len.is_none()
        && a.cfg_zero_star == b.cfg_zero_star
        // An explicit segment seed is a request for segment-local reproducibility.
        // Never absorb it into a larger generation with different RNG semantics.
        && a.seed.is_none()
        && b.seed.is_none()
}

fn internal_pause_separator(pause: TrailingPause) -> &'static str {
    match pause {
        TrailingPause::None => " ",
        TrailingPause::Short => "\n",
        TrailingPause::Normal => "\n\n",
        // Long pauses are deliberately kept as physical run boundaries so VoxGen
        // can preserve the advertised engine-appended long silence exactly.
        TrailingPause::Long => "\n\n\n",
    }
}

fn append_internal_pause_guidance(control: &mut String, pauses: &[TrailingPause]) {
    if pauses.is_empty() || pauses.iter().all(|pause| *pause == TrailingPause::None) {
        return;
    }
    let labels = pauses.iter().map(|pause| pause.as_str()).collect::<Vec<_>>().join(", ");
    control.push_str(&format!(" Interior pauses in order: [{labels}]. Preserve them naturally without speaking the labels."));
}

fn merge_sequence_run(
    members: &[SpeechRequest],
    active_generation_context: u32,
) -> Result<(SpeechRequest, Vec<TrailingPause>)> {
    if members.is_empty() {
        bail!("cannot merge an empty speech sequence run");
    }
    let mut merged = members[0].clone();
    let mut input = members[0].input.trim().to_owned();
    let mut internal_pauses = Vec::with_capacity(members.len().saturating_sub(1));
    for index in 0..members.len().saturating_sub(1) {
        let pause = TrailingPause::parse(members[index].pause_after.as_deref())?;
        if pause == TrailingPause::Long {
            bail!("long semantic pause cannot be absorbed into a coalesced acoustic run");
        }
        internal_pauses.push(pause);
        input.push_str(internal_pause_separator(pause));
        input.push_str(members[index + 1].input.trim());
    }
    merged.input = input;
    merged.pause_after = members.last().and_then(|member| member.pause_after.clone());

    if members.len() > 1 {
        // A merged utterance needs a generation ceiling proportional to the ceilings
        // the same semantic units would have had separately. Clamp it to the model's
        // active context instead of leaving the historical 200-step per-request cap,
        // which would truncate longer compiled narration runs.
        let requested = members.iter().fold(0u32, |total, member| {
            total.saturating_add(member.max_steps.unwrap_or(DEFAULT_TTS_MAX_STEPS))
        });
        let ceiling = active_generation_context.max(DEFAULT_TTS_MAX_STEPS);
        merged.max_steps = Some(requested.min(ceiling));
    }
    Ok((merged, internal_pauses))
}

fn progressive_streaming_segment(
    runtime: &Runtime,
    state: &ServerState,
    req: &SpeechRequest,
    effective_control: Option<&str>,
    ref_path: Option<&std::path::Path>,
    prompt_path: Option<&std::path::Path>,
    opt: &TtsOptions,
    gain: f32,
    playback_controls: PlaybackControls,
    trailing_pause: TrailingPause,
    pcm_tx: &std::sync::mpsc::SyncSender<Vec<u8>>,
) -> Result<usize> {
    let mut emitted_bytes = 0usize;
    let mut playback_dsp = StreamingPlaybackDsp::new(OUTPUT_SAMPLE_RATE_HZ, playback_controls)
        .map_err(anyhow::Error::msg)?;
    let mut peak_guard = OutputPeakGuard::new(OUTPUT_SAMPLE_RATE_HZ).map_err(anyhow::Error::msg)?;
    let synth = runtime.synthesize_cancelable(
        &req.input,
        effective_control,
        req.prompt_text.as_deref(),
        ref_path,
        prompt_path,
        opt,
        Some(&state.cancel_speech),
        Some(|chunk: &[f32], _sr: u32| -> Result<()> {
            let processed = playback_dsp.push(chunk).map_err(anyhow::Error::msg)?;
            if processed.is_empty() { return Ok(()); }
            let protected = peak_guard.process(&processed, gain).map_err(anyhow::Error::msg)?;
            let bytes = pcm_bytes(&protected);
            emitted_bytes = emitted_bytes.saturating_add(bytes.len());
            pcm_tx
                .send(bytes)
                .map_err(|_| anyhow::anyhow!("sequence streaming audio writer disconnected"))
        }),
    );

    if synth.is_ok() && !state.cancel_speech.load(Ordering::Acquire) {
        let tail = playback_dsp.finish().map_err(anyhow::Error::msg)?;
        if !tail.is_empty() {
            let protected = peak_guard.process(&tail, gain).map_err(anyhow::Error::msg)?;
            let bytes = pcm_bytes(&protected);
            emitted_bytes = emitted_bytes.saturating_add(bytes.len());
            pcm_tx
                .send(bytes)
                .map_err(|_| anyhow::anyhow!("sequence streaming audio writer disconnected"))?;
        }
        let mut remaining = ((OUTPUT_SAMPLE_RATE_HZ as u64 * trailing_pause.milliseconds() as u64) / 1000) as usize;
        let block_frames = (OUTPUT_SAMPLE_RATE_HZ as usize / 4).max(1);
        while remaining > 0 && !state.cancel_speech.load(Ordering::Acquire) {
            let frames = remaining.min(block_frames);
            let bytes = vec![0u8; frames * std::mem::size_of::<f32>()];
            emitted_bytes = emitted_bytes.saturating_add(bytes.len());
            pcm_tx
                .send(bytes)
                .map_err(|_| anyhow::anyhow!("sequence streaming audio writer disconnected"))?;
            remaining -= frames;
        }
    }
    if state.cancel_speech.load(Ordering::Acquire) {
        let _ = synth;
        bail!("speech synthesis cancelled");
    }
    synth?;
    Ok(emitted_bytes)
}

fn speech_sequence_stream(mut s: TcpStream, state: Arc<ServerState>, req: SpeechSequenceRequest) -> Result<()> {
    // The request contains the complete semantic plan. Adjacent steady destinations
    // are coalesced before synthesis, and one writer owns the response for the entire
    // sequence so style/run boundaries never create extra socket threads or joins.
    let _inference = state
        .inference_gate
        .lock()
        .map_err(|_| anyhow::anyhow!("VoxGen inference/model-load lock is poisoned"))?;
    let Some((_stream, runtime)) = runtime_or_503(s.try_clone()?, &state)? else {
        return Ok(());
    };

    let request_id = req.request_id.unwrap_or(0);
    state.cancel_speech.store(false, Ordering::Release);
    if request_id != 0 {
        state.active_speech_request.store(request_id, Ordering::Release);
        if state.cancel_speech_request.load(Ordering::Acquire) == request_id {
            state.cancel_speech.store(true, Ordering::Release);
        }
    }
    let _active_request = ActiveSpeechRequest { state: state.clone(), request_id };
    let segments = req.segments;
    let total_segments = segments.len();

    write!(
        s,
        "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nCache-Control: no-store\r\nX-VoxGen-Sample-Rate: {OUTPUT_SAMPLE_RATE_HZ}\r\nX-VoxGen-Native-Playback-DSP: 1\r\nX-VoxGen-Sequence: 1\r\nX-VoxGen-Sequence-Segments: {total_segments}\r\nX-VoxGen-Sequence-Compiler: adjacent-compatible-steady-state-coalescing\r\nX-VoxGen-Sequence-Delivery: progressive-pcm-continuous-writer\r\n\r\n",
    )?;
    write_chunk(&mut s, &wav_header_f32(OUTPUT_SAMPLE_RATE_HZ, u32::MAX))?;

    let (pcm_tx, pcm_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(4);
    let mut writer_stream = s.try_clone().context("clone sequence streaming socket")?;
    let writer = thread::spawn(move || -> Result<()> {
        while let Ok(bytes) = pcm_rx.recv() {
            write_chunk(&mut writer_stream, &bytes)?;
        }
        Ok(())
    });

    let active_generation_context = runtime.active_context_length();
    let mut semantic_index = 0usize;
    let mut acoustic_run = 0usize;
    let sequence_result = (|| -> Result<()> {
        while semantic_index < total_segments {
            if state.cancel_speech.load(Ordering::Acquire) {
                bail!("speech synthesis cancelled");
            }

            let first = &segments[semantic_index];
            let first_speaker_key = speaker_conditioning_key(first);
            let first_destination = parse_managed_destination(first)?;
            let first_plan = first_destination
                .clone()
                .map(|destination| continuity_plan(&state, first, destination, first_speaker_key.clone()))
                .transpose()?;

            let mut run_end = semantic_index + 1;
            if first.seed.is_none()
                && first_destination.is_some()
                && first_plan.as_ref().is_some_and(continuity_plan_is_steady)
            {
                let base_destination = first_destination.as_ref().expect("checked above");
                while run_end < total_segments {
                    let previous_member = &segments[run_end - 1];
                    if TrailingPause::parse(previous_member.pause_after.as_deref())? == TrailingPause::Long {
                        break;
                    }
                    let next = &segments[run_end];
                    let Some(next_destination) = parse_managed_destination(next)? else { break; };
                    if next_destination.boundary != ContinuityBoundary::Continuous
                        || !same_managed_destination(base_destination, &next_destination)
                        || !same_sequence_run_settings(first, next)
                        || speaker_conditioning_key(next) != first_speaker_key
                    {
                        break;
                    }
                    run_end += 1;
                }
            }

            let member_count = run_end - semantic_index;
            let (run_request, internal_pauses) = if member_count > 1 {
                merge_sequence_run(&segments[semantic_index..run_end], active_generation_context)?
            } else {
                (segments[semantic_index].clone(), Vec::new())
            };

            acoustic_run += 1;
            let started = Instant::now();
            let speaker_key = speaker_conditioning_key(&run_request);
            let managed_destination = parse_managed_destination(&run_request)?;
            let mut continuity = managed_destination
                .map(|destination| continuity_plan(&state, &run_request, destination, speaker_key))
                .transpose()?;
            if let Some(plan) = continuity.as_mut() {
                append_internal_pause_guidance(&mut plan.effective_control, &internal_pauses);
            }
            let effective_control = continuity
                .as_ref()
                .map(|plan| plan.effective_control.as_str())
                .or_else(|| run_request.control.as_deref());
            let mut temps = Temps(Vec::new());
            let (ref_path, prompt_path) = resolve_speech_audio_paths(&run_request, effective_control, &mut temps)?;
            let opt = options(
                &runtime,
                &run_request,
                continuity.as_ref().map(|plan| plan.cfg_delta),
            )?;
            let gain = run_request.gain.unwrap_or(state.default_gain);
            if !gain.is_finite() || gain < 0.0 {
                bail!("speech gain must be a finite value >= 0.0");
            }
            let playback_controls = PlaybackControls::new(
                run_request.speed_percent.unwrap_or(DEFAULT_SPEED_PERCENT),
                run_request.pitch_semitones.unwrap_or(DEFAULT_PITCH_SEMITONES),
            ).map_err(anyhow::Error::msg)?;
            let trailing_pause = TrailingPause::parse(run_request.pause_after.as_deref())?;

            let emitted_bytes = progressive_streaming_segment(
                &runtime,
                &state,
                &run_request,
                effective_control,
                ref_path.as_deref(),
                prompt_path.as_deref(),
                &opt,
                gain,
                playback_controls,
                trailing_pause,
                &pcm_tx,
            )?;
            if let Some(plan) = continuity.as_ref() {
                let _ = commit_continuity(&state, plan)?;
            }

            let range = if member_count == 1 {
                format!("{}", semantic_index + 1)
            } else {
                format!("{}-{}", semantic_index + 1, run_end)
            };
            eprintln!(
                "[VoxGen sequence] acoustic run {acoustic_run} compiled semantic segment(s) {range} ({member_count} unit{}) · delivered {:.3}s audio after {:.3}s synthesis",
                if member_count == 1 { "" } else { "s" },
                emitted_bytes as f64 / (OUTPUT_SAMPLE_RATE_HZ as f64 * std::mem::size_of::<f32>() as f64),
                started.elapsed().as_secs_f64(),
            );
            semantic_index = run_end;
        }
        Ok(())
    })();

    drop(pcm_tx);
    let writer_result = writer
        .join()
        .map_err(|_| anyhow::anyhow!("sequence streaming audio writer thread panicked"))?;

    if state.cancel_speech.load(Ordering::Acquire) {
        let _ = sequence_result;
        let _ = writer_result;
        s.write_all(b"0\r\n\r\n")?;
        return Ok(());
    }
    sequence_result?;
    writer_result?;
    eprintln!(
        "[VoxGen sequence] compiled {total_segments} semantic segment(s) into {acoustic_run} acoustic generation run(s)"
    );
    s.write_all(b"0\r\n\r\n")?;
    Ok(())
}

fn speech(mut s: TcpStream, state: Arc<ServerState>, req: SpeechRequest, streaming: bool) -> Result<()> {
    // Serialize synthesis against model reload/unload. Continuity reset intentionally
    // does not take this gate; generation epochs prevent an in-flight request from
    // recreating a session that was explicitly reset while synthesis was running.
    let _inference = state
        .inference_gate
        .lock()
        .map_err(|_| anyhow::anyhow!("VoxGen inference/model-load lock is poisoned"))?;
    let Some((_stream, runtime)) = runtime_or_503(s.try_clone()?, &state)? else {
        return Ok(());
    };

    let speaker_key = speaker_conditioning_key(&req);
    let managed_destination = parse_managed_destination(&req)?;
    let continuity = managed_destination
        .map(|destination| continuity_plan(&state, &req, destination, speaker_key))
        .transpose()?;
    let effective_control = continuity
        .as_ref()
        .map(|plan| plan.effective_control.as_str())
        .or_else(|| req.control.as_deref());

    let mut temps = Temps(Vec::new());
    let (ref_path, prompt_path) = resolve_speech_audio_paths(&req, effective_control, &mut temps)?;

    let opt = options(
        &runtime,
        &req,
        continuity.as_ref().map(|plan| plan.cfg_delta),
    )?;
    let gain = req.gain.unwrap_or(state.default_gain);
    if !gain.is_finite() || gain < 0.0 {
        bail!("speech gain must be a finite value >= 0.0");
    }
    let playback_controls = PlaybackControls::new(
        req.speed_percent.unwrap_or(100.0),
        req.pitch_semitones.unwrap_or(0.0),
    ).map_err(anyhow::Error::msg)?;
    let trailing_pause = TrailingPause::parse(req.pause_after.as_deref())?;
    let trailing_pause_ms = trailing_pause.milliseconds();

    // Establish a request-scoped cancellation identity only after request parsing
    // and conditioning validation. A targeted Stop can arrive slightly before the
    // speech POST itself; preserving its request_id here prevents that race from
    // accidentally clearing the cancellation and starting synthesis anyway.
    let request_id = req.request_id.unwrap_or(0);
    state.cancel_speech.store(false, Ordering::Release);
    if request_id != 0 {
        state.active_speech_request.store(request_id, Ordering::Release);
        if state.cancel_speech_request.load(Ordering::Acquire) == request_id {
            state.cancel_speech.store(true, Ordering::Release);
        }
    }
    let _active_request = ActiveSpeechRequest { state: state.clone(), request_id };

    if streaming {
        write!(
            s,
            "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nCache-Control: no-store\r\nX-VoxGen-Sample-Rate: {OUTPUT_SAMPLE_RATE_HZ}\r\nX-VoxGen-Native-Playback-DSP: 1\r\nX-VoxGen-Peak-Guard: 0.980\r\nX-VoxGen-Speed-Percent: {:.3}\r\nX-VoxGen-Pitch-Semitones: {:+.3}\r\nX-VoxGen-CFG: {:.3}\r\nX-VoxGen-Pause-After: {}\r\n",
            playback_controls.speed_percent,
            playback_controls.pitch_semitones,
            opt.cfm.cfg_value,
            trailing_pause.as_str()
        )?;
        for (name, value) in continuity_headers(continuity.as_ref()) {
            write!(s, "{name}: {value}\r\n")?;
        }
        s.write_all(b"\r\n")?;
        // Streaming length is unknown until stop prediction fires. 0xffffffff is the conventional streaming sentinel.
        write_chunk(&mut s, &wav_header_f32(OUTPUT_SAMPLE_RATE_HZ, u32::MAX))?;
        let (pcm_tx, pcm_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(4);
        let mut writer_stream = s.try_clone().context("clone streaming socket")?;
        let writer = thread::spawn(move || -> Result<()> {
            while let Ok(bytes) = pcm_rx.recv() {
                write_chunk(&mut writer_stream, &bytes)?;
            }
            Ok(())
        });
        let mut playback_dsp = StreamingPlaybackDsp::new(OUTPUT_SAMPLE_RATE_HZ, playback_controls)
            .map_err(anyhow::Error::msg)?;
        let mut peak_guard = OutputPeakGuard::new(OUTPUT_SAMPLE_RATE_HZ).map_err(anyhow::Error::msg)?;
        let synth = runtime.synthesize_cancelable(
            &req.input,
            effective_control,
            req.prompt_text.as_deref(),
            ref_path.as_deref(),
            prompt_path.as_deref(),
            &opt,
            Some(&state.cancel_speech),
            Some(|chunk: &[f32], _sr: u32| -> Result<()> {
                let processed = playback_dsp.push(chunk).map_err(anyhow::Error::msg)?;
                if processed.is_empty() {
                    return Ok(());
                }
                let protected = peak_guard.process(&processed, gain).map_err(anyhow::Error::msg)?;
                pcm_tx
                    .send(pcm_bytes(&protected))
                    .map_err(|_| anyhow::anyhow!("streaming audio writer disconnected"))
            }),
        );
        if synth.is_ok() && !state.cancel_speech.load(Ordering::Acquire) {
            let tail = playback_dsp.finish().map_err(anyhow::Error::msg)?;
            if !tail.is_empty() {
                let protected = peak_guard.process(&tail, gain).map_err(anyhow::Error::msg)?;
                pcm_tx
                    .send(pcm_bytes(&protected))
                    .map_err(|_| anyhow::anyhow!("streaming audio writer disconnected"))?;
            }
            let mut remaining = ((OUTPUT_SAMPLE_RATE_HZ as u64 * trailing_pause_ms as u64) / 1000) as usize;
            let block_frames = (OUTPUT_SAMPLE_RATE_HZ as usize / 4).max(1);
            while remaining > 0 && !state.cancel_speech.load(Ordering::Acquire) {
                let frames = remaining.min(block_frames);
                pcm_tx
                    .send(vec![0u8; frames * std::mem::size_of::<f32>()])
                    .map_err(|_| anyhow::anyhow!("streaming audio writer disconnected"))?;
                remaining -= frames;
            }
        }
        drop(pcm_tx);
        let writer_result = writer
            .join()
            .map_err(|_| anyhow::anyhow!("streaming audio writer thread panicked"))?;
        // A cancelled stream is a normal control-flow event: do not commit continuity state.
        if state.cancel_speech.load(Ordering::Acquire) {
            let _ = synth;
            let _ = writer_result;
            s.write_all(b"0\r\n\r\n")?;
            s.flush()?;
            return Ok(());
        }
        synth?;
        writer_result?;
        if let Some(plan) = continuity.as_ref() {
            let _ = commit_continuity(&state, plan)?;
        }
        s.write_all(b"0\r\n\r\n")?;
        s.flush()?;
        Ok(())
    } else {
        let result = match runtime.synthesize_cancelable::<fn(&[f32], u32) -> Result<()>>(
            &req.input,
            effective_control,
            req.prompt_text.as_deref(),
            ref_path.as_deref(),
            prompt_path.as_deref(),
            &opt,
            Some(&state.cancel_speech),
            None,
        ) {
            Ok(result) => result,
            Err(_err) if state.cancel_speech.load(Ordering::Acquire) => {
                return send_json(s, "409 Conflict", json!({"ok": false, "cancelled": true}));
            }
            Err(err) => return Err(err),
        };
        let rendered_samples = StreamingPlaybackDsp::process_all(OUTPUT_SAMPLE_RATE_HZ, playback_controls, &result.samples)
            .map_err(anyhow::Error::msg)?;
        let mut protected_samples = OutputPeakGuard::process_all(OUTPUT_SAMPLE_RATE_HZ, &rendered_samples, gain)
            .map_err(anyhow::Error::msg)?;
        let trailing_frames = ((OUTPUT_SAMPLE_RATE_HZ as u64 * trailing_pause_ms as u64) / 1000) as usize;
        protected_samples.extend(std::iter::repeat(0.0f32).take(trailing_frames));
        let rendered_audio_seconds = protected_samples.len() as f64 / OUTPUT_SAMPLE_RATE_HZ as f64;
        if let Some(plan) = continuity.as_ref() {
            let _ = commit_continuity(&state, plan)?;
        }
        let mut headers = vec![
            ("X-VoxGen-Generated-Patches", result.generated_patches.to_string()),
            ("X-VoxGen-Stopped-By-Predictor", result.stopped_by_predictor.to_string()),
            ("X-VoxGen-Audio-Seconds", format!("{:.6}", rendered_audio_seconds)),
            ("X-VoxGen-Speed-Percent", format!("{:.3}", playback_controls.speed_percent)),
            ("X-VoxGen-Pitch-Semitones", format!("{:+.3}", playback_controls.pitch_semitones)),
            ("X-VoxGen-Peak-Guard", format!("{:.3}", OUTPUT_PEAK_CEILING)),
            ("X-VoxGen-CFG", format!("{:.3}", opt.cfm.cfg_value)),
            ("X-VoxGen-Pause-After", trailing_pause.as_str().to_owned()),
            ("X-VoxGen-Elapsed-Ms", format!("{:.3}", result.elapsed_ms)),
            ("X-VoxGen-First-PCM-Ms", format!("{:.3}", result.first_pcm_ms.unwrap_or_default())),
            ("X-VoxGen-RTF", format!("{:.6}", result.rtf)),
        ];
        headers.extend(continuity_headers(continuity.as_ref()));
        match req.response_format.as_deref().unwrap_or("wav") {
            "wav" => send_with_headers(s, "200 OK", "audio/wav", &wav_bytes(&protected_samples), &headers),
            "pcm" | "f32" => send_with_headers(
                s,
                "200 OK",
                "application/octet-stream",
                &pcm_bytes(&protected_samples),
                &headers,
            ),
            _ => unreachable!("response_format validated before synthesis"),
        }
    }
}

fn handle(mut s: TcpStream, state: Arc<ServerState>) -> Result<()> {
    let peer_addr = s.peer_addr().ok();
    let (method, path, body) = read_request(&mut s)?;
    match (method.as_str(), path.as_str()) {
        ("POST", "/v1/server/shutdown") => {
            let loopback = peer_addr
                .map(|addr| addr.ip().is_loopback())
                .unwrap_or(false);
            if !loopback {
                return send_json(
                    s,
                    "403 Forbidden",
                    json!({"error": "server shutdown is only available from loopback"}),
                );
            }
            let pid = std::process::id();
            send_json(s, "200 OK", json!({"ok": true, "pid": pid, "shutting_down": true}))?;
            // Let the response reach the demo before terminating the process.
            // std::process::exit from this detached request thread cleanly stops
            // the listener as well as all other server threads.
            thread::spawn(|| {
                thread::sleep(Duration::from_millis(150));
                std::process::exit(0);
            });
            Ok(())
        }
        ("POST", "/v1/audio/continuity/reset") => {
            // Do not take inference_gate: a reset must invalidate an in-flight
            // synthesis immediately. commit_continuity checks this ID generation.
            let reset: ContinuityResetRequest = match serde_json::from_slice(&body) {
                Ok(value) => value,
                Err(err) => return send_json(s, "400 Bad Request", json!({"ok": false, "error": format!("invalid continuity reset request: {err}")})),
            };
            let continuity_id = reset.continuity_id.trim();
            if continuity_id.is_empty() || continuity_id.len() > 128 {
                return send_json(s, "400 Bad Request", json!({"ok": false, "error": "continuity_id must contain 1..=128 bytes"}));
            }
            let removed = state.reset_continuity_id(continuity_id)?;
            send_json(s, "200 OK", json!({"ok": true, "continuity_id": continuity_id, "reset": true, "had_state": removed}))
        }
        ("POST", "/v1/audio/speech/cancel") => {
            // Deliberately do not take inference_gate here: the endpoint must stay
            // responsive while a long speech request owns that gate. Runtime checks
            // this flag at completed GPU-operation / acoustic-patch boundaries.
            let cancel: SpeechCancelRequest = if body.is_empty() {
                SpeechCancelRequest::default()
            } else {
                match serde_json::from_slice(&body) {
                    Ok(value) => value,
                    Err(err) => return send_json(s, "400 Bad Request", json!({"ok": false, "error": format!("invalid speech cancel request: {err}")})),
                }
            };
            let active = state.active_speech_request.load(Ordering::Acquire);
            let matched = if let Some(request_id) = cancel.request_id.filter(|id| *id != 0) {
                state.cancel_speech_request.store(request_id, Ordering::Release);
                if active == request_id {
                    state.cancel_speech.store(true, Ordering::Release);
                    true
                } else {
                    false
                }
            } else {
                // Compatibility path for API clients that do not supply request_id:
                // cancel whichever serialized speech request is active right now.
                state.cancel_speech.store(true, Ordering::Release);
                active != 0
            };
            send_json(s, "200 OK", json!({"ok": true, "cancelling": true, "matched_active_request": matched}))
        }
        ("GET", "/health") | ("GET", "/v1/health") => {
            send_json(s, "200 OK", health_json(&state)?)
        }
        ("GET", "/v1/models/current") => send_json(s, "200 OK", model_json(&state)?),
        ("GET", "/v1/models") | ("GET", "/v1/audio/speech/models") => {
            let current = model_json(&state)?;
            send_json(
                s,
                "200 OK",
                json!({
                    "object": "list",
                    "data": if current.get("loaded").and_then(Value::as_bool).unwrap_or(false) {
                        vec![json!({
                            "id": "voxcpm2",
                            "object": "model",
                            "engine": "VoxGen",
                            "base_lm": current.get("base_lm"),
                            "acoustic": current.get("acoustic"),
                            "base_format": current.get("base_format"),
                            "speech_inference_ready": current.get("speech_inference_ready")
                        })]
                    } else { Vec::<Value>::new() },
                    "current": current
                }),
            )
        }
        ("POST", "/v1/models/load") | ("POST", "/v1/voxgen/models/load") => {
            match load_model(&state, &body) {
                Ok(current) => send_json(s, "200 OK", json!({"ok": true, "model": current})),
                Err(err) => send_json(
                    s,
                    "400 Bad Request",
                    json!({"ok": false, "error": format!("{err:#}")}),
                ),
            }
        }
        ("POST", "/v1/models/unload") | ("POST", "/v1/voxgen/models/unload") => {
            match unload_model(&state) {
                Ok(value) => send_json(s, "200 OK", value),
                Err(err) => send_json(
                    s,
                    "500 Internal Server Error",
                    json!({"ok": false, "error": format!("{err:#}")}),
                ),
            }
        }
        ("GET", "/v1/voxgen/diagnostics") | ("GET", "/v1/voxcpm2/diagnostics") => {
            let _inference = state.inference_gate.lock().map_err(|_| anyhow::anyhow!("VoxGen inference/model-load lock is poisoned"))?;
            let Some((_stream, runtime)) = runtime_or_503(s.try_clone()?, &state)? else {
                return Ok(());
            };
            send_json(s, "200 OK", serde_json::to_value(runtime.status())?)
        }
        ("GET", "/v1/profile/gpu") => {
            let _inference = state.inference_gate.lock().map_err(|_| anyhow::anyhow!("VoxGen inference/model-load lock is poisoned"))?;
            let Some((_stream, runtime)) = runtime_or_503(s.try_clone()?, &state)? else {
                return Ok(());
            };
            let snapshot = runtime.gpu.gpu_profile_snapshot();
            send_json(s, "200 OK", json!({
                "ok": true,
                "mode": runtime.gpu.mode.as_str(),
                "profile": snapshot
            }))
        }
        ("POST", "/v1/profile/gpu/reset") => {
            let _inference = state.inference_gate.lock().map_err(|_| anyhow::anyhow!("VoxGen inference/model-load lock is poisoned"))?;
            let Some((_stream, runtime)) = runtime_or_503(s.try_clone()?, &state)? else {
                return Ok(());
            };
            runtime.reset_gpu_profile();
            send_json(s, "200 OK", json!({
                "ok": true,
                "enabled": runtime.gpu.gpu_profiling_enabled()
            }))
        }
        ("POST", "/v1/voxgen/baselm/reset") => {
            let _inference = state.inference_gate.lock().map_err(|_| anyhow::anyhow!("VoxGen inference/model-load lock is poisoned"))?;
            let Some((_stream, runtime)) = runtime_or_503(s.try_clone()?, &state)? else {
                return Ok(());
            };
            runtime.reset_baselm();
            send_json(s, "200 OK", json!({"ok": true, "position": 0}))
        }
        ("POST", "/v1/voxgen/residual/reset") => {
            let _inference = state.inference_gate.lock().map_err(|_| anyhow::anyhow!("VoxGen inference/model-load lock is poisoned"))?;
            let Some((_stream, runtime)) = runtime_or_503(s.try_clone()?, &state)? else {
                return Ok(());
            };
            runtime.reset_residual();
            send_json(s, "200 OK", json!({"ok": true, "position": 0}))
        }
        ("POST", "/v1/voxgen/pipeline/reset") => {
            let _inference = state.inference_gate.lock().map_err(|_| anyhow::anyhow!("VoxGen inference/model-load lock is poisoned"))?;
            let Some((_stream, runtime)) = runtime_or_503(s.try_clone()?, &state)? else {
                return Ok(());
            };
            runtime.reset_pipeline();
            state.clear_continuity()?;
            send_json(
                s,
                "200 OK",
                json!({"ok": true, "baselm_position": 0, "residual_position": 0}),
            )
        }
        ("POST", "/v1/audio/conditioning/warm") => {
            let request: ConditioningWarmRequest = match serde_json::from_slice(&body) {
                Ok(value) => value,
                Err(err) => return send_json(s, "400 Bad Request", json!({"ok": false, "error": format!("invalid conditioning warm request: {err}")})),
            };
            let _inference = state.inference_gate.lock().map_err(|_| anyhow::anyhow!("VoxGen inference/model-load lock is poisoned"))?;
            let Some((_stream, runtime)) = runtime_or_503(s.try_clone()?, &state)? else {
                return Ok(());
            };
            match runtime.warm_reference_wav(&request.reference_audio_path) {
                Ok((stats, patches)) => send_json(s, "200 OK", json!({
                    "ok": true,
                    "reference_audio_path": request.reference_audio_path,
                    "latent_patches": patches,
                    "encode_ms": stats.encode_ms
                })),
                Err(err) => send_json(s, "400 Bad Request", json!({"ok": false, "error": format!("{err:#}")})),
            }
        }
        ("POST", "/v1/audio/speech/sequence/stream") => {
            if !state.streaming_enabled {
                send_json(
                    s,
                    "409 Conflict",
                    json!({
                        "error": "speech streaming is disabled",
                        "hint": "restart VoxGen with --stream on"
                    }),
                )
            } else {
                match parse_speech_sequence_request(&body) {
                    Ok(req) => speech_sequence_stream(s, state, req),
                    Err(err) => send_json(s, "400 Bad Request", json!({"ok": false, "error": format!("{err:#}")})),
                }
            }
        }
        ("POST", "/v1/audio/speech") => {
            match parse_speech_request(&body) {
                Ok(req) => speech(s, state, req, false),
                Err(err) => send_json(s, "400 Bad Request", json!({"ok": false, "error": format!("{err:#}")})),
            }
        }
        ("POST", "/v1/audio/speech/stream") => {
            if !state.streaming_enabled {
                send_json(
                    s,
                    "409 Conflict",
                    json!({
                        "error": "speech streaming is disabled",
                        "hint": "restart VoxGen with --stream on"
                    }),
                )
            } else {
                match parse_speech_request(&body) {
                    Ok(req) => speech(s, state, req, true),
                    Err(err) => send_json(s, "400 Bad Request", json!({"ok": false, "error": format!("{err:#}")})),
                }
            }
        }
        _ => send_json(s, "404 Not Found", json!({"error": "not found", "path": path})),
    }
}

pub fn serve(
    host: &str,
    port: u16,
    runtime: Option<Arc<Runtime>>,
    default_base_format: Option<BaseFormat>,
    default_gpu: Option<usize>,
    default_mode: ExecutionMode,
    default_xtx_tuning: XtxTuning,
    default_max_context: u32,
    streaming_enabled: bool,
    default_gain: f32,
) -> Result<()> {
    let listener = TcpListener::bind((host, port)).with_context(|| format!("bind {host}:{port}"))?;
    let state = Arc::new(ServerState::new(
        runtime,
        default_base_format,
        default_gpu,
        default_mode,
        default_xtx_tuning,
        default_max_context,
        streaming_enabled,
        default_gain,
    ));
    eprintln!("[VoxGen] listening on http://{host}:{port}");
    eprintln!("[VoxGen] server execution mode: {}", default_mode.as_str());
    if default_mode == ExecutionMode::Xtx7900 {
        eprintln!(
            "[VoxGen] XTX stream tuning: shared residual-rms/swiglu · GPU timestamps {} · cooperative matrix {}",
            if default_xtx_tuning.gpu_profile { "on" } else { "off" },
            if default_xtx_tuning.cooperative_matrix { "on" } else { "off" },
        );
    }
    eprintln!(
        "[VoxGen] speech streaming: {}",
        if streaming_enabled { "on" } else { "off" }
    );
    eprintln!("[VoxGen] default speech gain: {:.3}x", default_gain);
    if state.runtime_snapshot()?.is_none() {
        eprintln!("[VoxGen] server started without a model; POST /v1/models/load to select GGUF paths");
    }
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let state = state.clone();
                thread::spawn(move || {
                    if let Err(e) = handle(stream, state) {
                        eprintln!("[VoxGen] request error: {e:#}");
                    }
                });
            }
            Err(e) => eprintln!("[VoxGen] accept error: {e}"),
        }
    }
    Ok(())
}
