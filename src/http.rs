use crate::{
    gguf::BaseFormat,
    local::CfmOptions,
    runtime::{Runtime, TtsOptions},
    vulkan::{ExecutionMode, XtxTuning},
};
use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, RwLock,
    },
    thread,
    time::Duration,
};

static TEMP_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Deserialize)]
struct SpeechRequest {
    #[serde(alias = "text")]
    input: String,
    #[serde(default)]
    prompt_text: Option<String>,
    #[serde(default)]
    control: Option<String>,
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

#[derive(Debug, Deserialize, Default)]
struct SpeechCancelRequest {
    #[serde(default)]
    request_id: Option<u64>,
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
            streaming_enabled,
            default_gain,
            default_base_format,
            default_gpu,
            default_mode,
            default_xtx_tuning,
            default_max_context,
        }
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
    s.flush()?;
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

fn parse_speech_request(body: &[u8]) -> Result<SpeechRequest> {
    let req: SpeechRequest = serde_json::from_slice(body).context("parse speech request JSON")?;
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
    if req.control.as_deref().is_some_and(|x| !x.trim().is_empty())
        && (has_prompt || has_prompt_text || clone_mode == "ultimate")
    {
        bail!("control cannot be combined with ultimate/prompt continuation cloning");
    }
    Ok(req)
}

fn options(runtime: &Runtime, r: &SpeechRequest) -> TtsOptions {
    let cfg = runtime
        .status()
        .local
        .as_ref()
        .map(|x| x.config.cfm_cfg_rate)
        .unwrap_or(2.0);
    TtsOptions {
        min_steps: r.min_steps.unwrap_or(2),
        max_steps: r.max_steps.unwrap_or(200),
        streaming_prefix_len: r.streaming_prefix_len.unwrap_or(4),
        cfm: CfmOptions {
            n_timesteps: r.inference_timesteps.unwrap_or(10),
            cfg_value: r.cfg_value.unwrap_or(cfg),
            temperature: r.temperature.unwrap_or(1.0),
            sway_sampling_coef: r.sway_sampling_coef.unwrap_or(1.0),
            seed: r.seed.unwrap_or(42),
            use_cfg_zero_star: r.cfg_zero_star.unwrap_or(true),
        },
    }
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

fn pcm_bytes(x: &[f32], gain: f32) -> Vec<u8> {
    let mut b = Vec::with_capacity(x.len() * 4);
    for &v in x {
        b.extend_from_slice(&(v * gain).clamp(-1.0, 1.0).to_le_bytes());
    }
    b
}

fn wav_bytes(x: &[f32], gain: f32) -> Vec<u8> {
    let pcm = pcm_bytes(x, gain);
    let mut b = wav_header_f32(48000, pcm.len().min(u32::MAX as usize) as u32);
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

fn speech(mut s: TcpStream, state: Arc<ServerState>, req: SpeechRequest, streaming: bool) -> Result<()> {
    // Serialize synthesis against model reload/unload. Runtime's internal locks
    // still govern its component pipeline; this gate is specifically lifecycle safety.
    let _inference = state
        .inference_gate
        .lock()
        .map_err(|_| anyhow::anyhow!("VoxGen inference/model-load lock is poisoned"))?;
    let Some((_stream, runtime)) = runtime_or_503(s.try_clone()?, &state)? else {
        return Ok(());
    };

    let mut temps = Temps(Vec::new());
    let ref_path = audio_path(
        &req.reference_audio,
        &req.reference_audio_path,
        "reference",
        &mut temps,
    )?;
    let mut prompt_path = audio_path(
        &req.prompt_audio,
        &req.prompt_audio_path,
        "prompt",
        &mut temps,
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
    if req.control.as_deref().is_some_and(|x|!x.trim().is_empty()) && prompt_path.is_some() {
        bail!("control cannot be combined with ultimate/prompt continuation cloning");
    }
    let opt = options(&runtime, &req);
    let gain = req.gain.unwrap_or(state.default_gain);
    if !gain.is_finite() || gain < 0.0 {
        bail!("speech gain must be a finite value >= 0.0");
    }

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
            "HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nCache-Control: no-store\r\nX-VoxGen-Sample-Rate: 48000\r\n\r\n"
        )?;
        // Streaming length is unknown until stop prediction fires. 0xffffffff is the conventional streaming sentinel.
        write_chunk(&mut s, &wav_header_f32(48000, u32::MAX))?;
        // Keep socket I/O off the inference thread. A small bounded queue allows the GPU
        // pipeline to run ahead of ordinary localhost/network write latency while preserving
        // backpressure if a client genuinely cannot consume audio fast enough.
        let (pcm_tx, pcm_rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(4);
        let mut writer_stream = s.try_clone().context("clone streaming socket")?;
        let writer = thread::spawn(move || -> Result<()> {
            while let Ok(bytes) = pcm_rx.recv() {
                write_chunk(&mut writer_stream, &bytes)?;
            }
            Ok(())
        });
        let synth = runtime.synthesize_cancelable(
            &req.input,
            req.control.as_deref(),
            req.prompt_text.as_deref(),
            ref_path.as_deref(),
            prompt_path.as_deref(),
            &opt,
            Some(&state.cancel_speech),
            Some(|chunk: &[f32], _sr: u32| -> Result<()> {
                pcm_tx
                    .send(pcm_bytes(chunk, gain))
                    .map_err(|_| anyhow::anyhow!("streaming audio writer disconnected"))
            }),
        );
        drop(pcm_tx);
        let writer_result = writer
            .join()
            .map_err(|_| anyhow::anyhow!("streaming audio writer thread panicked"))?;
        if state.cancel_speech.load(Ordering::Acquire) {
            // A cancelled stream is a normal control-flow event. The writer has
            // already drained any bytes produced before the safe cancellation
            // boundary; terminate chunked transfer cleanly instead of surfacing a
            // noisy server error for the Stop button.
            let _ = synth;
            let _ = writer_result;
            s.write_all(b"0\r\n\r\n")?;
            s.flush()?;
            return Ok(());
        }
        synth?;
        writer_result?;
        s.write_all(b"0\r\n\r\n")?;
        s.flush()?;
        Ok(())
    } else {
        let result = match runtime.synthesize_cancelable::<fn(&[f32], u32) -> Result<()>>(
            &req.input,
            req.control.as_deref(),
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
        let headers = [
            ("X-VoxGen-Generated-Patches", result.generated_patches.to_string()),
            ("X-VoxGen-Stopped-By-Predictor", result.stopped_by_predictor.to_string()),
            ("X-VoxGen-Audio-Seconds", format!("{:.6}", result.audio_seconds)),
            ("X-VoxGen-Elapsed-Ms", format!("{:.3}", result.elapsed_ms)),
            ("X-VoxGen-First-PCM-Ms", format!("{:.3}", result.first_pcm_ms.unwrap_or_default())),
            ("X-VoxGen-RTF", format!("{:.6}", result.rtf)),
        ];
        match req.response_format.as_deref().unwrap_or("wav") {
            "wav" => send_with_headers(s, "200 OK", "audio/wav", &wav_bytes(&result.samples, gain), &headers),
            "pcm" | "f32" => send_with_headers(
                s,
                "200 OK",
                "application/octet-stream",
                &pcm_bytes(&result.samples, gain),
                &headers,
            ),
            x => bail!("unsupported response_format {x:?}; use wav or pcm"),
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
