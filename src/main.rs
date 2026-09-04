#![recursion_limit = "512"]

mod acoustic;
mod audiovae;
mod baselm;
mod conditioning;
mod gguf;
mod http;
mod local;
mod profiler;
mod prosody_control;
mod runtime;
mod tokenizer;
mod vulkan;

use acoustic::AcousticConfig;
use audiovae::{AudioPadSide, AudioVaeConfig};
use anyhow::{bail, Context, Result};
use baselm::BaseLmConfig;
use clap::{Parser, ValueEnum};
use gguf::BaseFormat;
use local::{CfmOptions, LocalConfig};
use runtime::{Runtime, TtsOptions};
use vulkan::{ExecutionMode, XtxTuning};
use voxgen::playback_dsp::{OutputPeakGuard, PlaybackControls, StreamingPlaybackDsp};
use crate::prosody_control::{build_style_control, managed_style_tuning};
use serde_json::json;
use std::{fs, path::{Path, PathBuf}, sync::Arc, time::Instant};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BaseFormatArg {
    Auto,
    #[value(name = "q8_0", alias = "q8-0")]
    Q8_0,
    F16,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum AudioPadSideArg { Left, Right }
impl From<AudioPadSideArg> for AudioPadSide {
    fn from(v:AudioPadSideArg)->Self{match v{AudioPadSideArg::Left=>AudioPadSide::Left,AudioPadSideArg::Right=>AudioPadSide::Right}}
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum StreamArg {
    On,
    Off,
}

impl StreamArg {
    fn enabled(self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum CloneModeArg {
    /// Infer conditioning mode from the supplied reference/prompt arguments.
    Auto,
    /// Reference-only cloning. Native `--control` style instructions are allowed.
    #[value(alias = "controllable")]
    Reference,
    /// Prompt-audio + exact transcript cloning. If only one WAV is supplied,
    /// VoxGen uses it as both prompt and reference for maximum speaker similarity.
    Ultimate,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum ModeArg {
    Normal,
    #[value(name = "xtx7900")]
    Xtx7900,
}
impl From<ModeArg> for ExecutionMode {
    fn from(v: ModeArg) -> Self { match v { ModeArg::Normal => ExecutionMode::Normal, ModeArg::Xtx7900 => ExecutionMode::Xtx7900 } }
}

#[derive(Parser, Debug)]
#[command(
    name = "voxgen",
    version,
    about = "VoxGen: specialized VoxCPM2 Vulkan inference engine"
)]
struct Args {
    /// VoxCPM2 MiniCPM4 BaseLM GGUF (Q8_0 or F16).
    #[arg(long = "base-lm", visible_alias = "voxcpm2-base-lm")]
    base_lm: Option<PathBuf>,

    /// VoxCPM2 Acoustic F16 GGUF. Required for FSQ / ResidualLM commands.
    #[arg(long = "acoustic", visible_alias = "voxcpm2-acoustic")]
    acoustic: Option<PathBuf>,

    /// Detect BaseLM type from GGUF by default; force a format to catch model swaps.
    #[arg(long, value_enum, default_value = "auto")]
    base_format: BaseFormatArg,

    /// Persistent BaseLM + ResidualLM KV-cache positions. VoxCPM2 generation max is 8192.
    #[arg(long, default_value_t = 8192)]
    max_context: u32,

    /// Start the HTTP API even when no model is supplied. Models can then be loaded with POST /v1/models/load.
    #[arg(long)]
    server: bool,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8091)]
    port: u16,

    /// Strict compatibility option. VoxGen supports GPU-only execution, therefore only -1 is valid.
    #[arg(long = "n-gpu-layers", default_value_t = -1)]
    n_gpu_layers: i32,

    /// Vulkan device index; omitted means prefer a discrete AMD GPU with the largest local heap.
    #[arg(long)]
    gpu: Option<usize>,

    /// GPU execution profile. normal preserves generic Vulkan kernels; xtx7900 enables RX 7900 XTX-tuned subgroup kernels.
    #[arg(long, value_enum, default_value = "normal")]
    mode: ModeArg,

    /// Collect per-kernel Vulkan timestamp timings in XTX mode. Off by default because synchronous query readback can disrupt real-time streaming.
    #[arg(long = "gpu-profile", value_enum, default_value = "off")]
    gpu_profile: StreamArg,

    /// Offline profiling convenience mode. Requires --mode xtx7900 and --stream off,
    /// enables GPU timestamps, and is intended for benchmark runs rather than playback.
    #[arg(long = "benchmark-profile")]
    benchmark_profile: bool,

    /// Enable the experimental cooperative-matrix LocEnc/LocDiT path in XTX mode. Off by default until validated on the active Radeon driver.
    #[arg(long = "xtx-coopmat", value_enum, default_value = "off")]
    xtx_coopmat: StreamArg,

    #[arg(long)]
    list_devices: bool,

    /// Validate BaseLM architecture/tensors without allocating VRAM.
    #[arg(long)]
    inspect_base_lm: bool,

    /// Validate VoxCPM2 Acoustic tensors (FSQ, ResidualLM, LocEnc, LocDiT) without allocating VRAM.
    #[arg(long)]
    inspect_acoustic: bool,

    /// Load enabled components and print runtime/GPU/memory state, then exit.
    #[arg(long)]
    diagnostics_json: bool,

    /// Run one autoregressive MiniCPM4 BaseLM step from this token ID.
    #[arg(long)]
    baselm_token: Option<u32>,

    /// Sequential correctness-first BaseLM prefill. Example: --baselm-prefill 1,2,3,4
    #[arg(long)]
    baselm_prefill: Option<String>,

    /// Raw little-endian f32 file containing exactly 2048 floats for a BaseLM embedding step.
    #[arg(long)]
    baselm_embedding_f32: Option<PathBuf>,

    /// Benchmark repeated autoregressive BaseLM steps with this token ID.
    #[arg(long)]
    baselm_bench_token: Option<u32>,

    /// Run only FSQ on a raw 2048-float BaseLM hidden vector.
    #[arg(long)]
    fsq_input_f32: Option<PathBuf>,

    /// Raw 2048-float BaseLM hidden vector for a standalone FSQ+fusion+ResidualLM step.
    #[arg(long)]
    residual_base_hidden_f32: Option<PathBuf>,

    /// Raw 2048-float current acoustic embedding paired with --residual-base-hidden-f32.
    #[arg(long)]
    residual_current_embedding_f32: Option<PathBuf>,

    /// Exact implemented generation-loop handoff: current embedding -> BaseLM -> FSQ -> fusion -> ResidualLM.
    #[arg(long)]
    base_residual_embedding_f32: Option<PathBuf>,

    /// Text-prefix prefill through both language-model caches. Example: --base-residual-text-prefill 1,2,3
    #[arg(long)]
    base_residual_text_prefill: Option<String>,

    /// Raw little-endian f32 file containing one VoxCPM2 latent patch: 4x64 = 256 floats.
    #[arg(long)]
    locenc_patch_f32: Option<PathBuf>,

    /// Raw 4x64 noisy-target latent patch for one LocDiT estimator call.
    #[arg(long)]
    locdit_x_f32: Option<PathBuf>,

    /// Raw 4x64 condition patch for one LocDiT estimator call.
    #[arg(long)]
    locdit_cond_f32: Option<PathBuf>,

    /// Raw 2048-float mu file (two 1024-D LocDiT mu tokens) for the estimator smoke path.
    #[arg(long)]
    locdit_mu_f32: Option<PathBuf>,

    /// CFM time value for the raw LocDiT estimator.
    #[arg(long, default_value_t = 0.5)]
    locdit_t: f32,

    /// Delta-time embedding supplied to LocDiT V2. UnifiedCFM mean_mode=false normally passes zero.
    #[arg(long, default_value_t = 0.0)]
    locdit_dt: f32,

    /// Final text-token sequence used to build the VoxCPM2 conditioning prefix.
    /// For continuation, tokenize prompt_text+target_text as one string and pass those token IDs here.
    #[arg(long)]
    conditioning_text_tokens: Option<String>,

    /// Raw latent reference features: N * 256 little-endian f32 values. Alternatively use --reference-wav.
    #[arg(long)]
    reference_latents_f32: Option<PathBuf>,

    /// Raw latent prompt/continuation features: N * 256 little-endian f32 values.
    #[arg(long)]
    prompt_latents_f32: Option<PathBuf>,

    /// Run UnifiedCFM from an explicit 2048-float LocDiT mu file. Pair with --cfm-cond-f32.
    #[arg(long)]
    cfm_mu_f32: Option<PathBuf>,

    /// One 4x64 (=256-float) condition patch for UnifiedCFM.
    #[arg(long)]
    cfm_cond_f32: Option<PathBuf>,

    /// Optional exact starting x for CFM, 256 floats. If omitted VoxGen generates Gaussian noise on Vulkan.
    #[arg(long)]
    cfm_initial_x_f32: Option<PathBuf>,

    /// After latent reference/prompt prefill, run UnifiedCFM to generate the next 4x64 latent patch.
    #[arg(long)]
    cfm_after_conditioning: bool,

    #[arg(long, visible_alias = "inference-timesteps", default_value_t = 10)]
    cfm_steps: u32,

    /// CFG value. If omitted, use voxcpm.cfm.cfg_rate from the acoustic GGUF (normally 2.0).
    #[arg(long, visible_alias = "cfg-value")]
    cfm_cfg: Option<f32>,

    #[arg(long, visible_alias = "temperature", default_value_t = 1.0)]
    cfm_temperature: f32,

    #[arg(long, default_value_t = 1.0)]
    cfm_sway: f32,

    #[arg(long, visible_alias = "seed", default_value_t = 42)]
    cfm_seed: u64,

    /// Disable CFG-Zero* optimal negative scaling and use conventional CFG scaling.
    #[arg(long)]
    cfm_no_zero_star: bool,

    /// Benchmark explicit-mu CFM. Includes final 256-float GPU readback per solve.
    #[arg(long)]
    cfm_bench: bool,

    /// Optional path to write the generated 256-float latent patch as little-endian f32.
    #[arg(long)]
    cfm_output_f32: Option<PathBuf>,

    /// Encode a WAV file with AudioVAE V2 and return frame-major 64-D latents.
    #[arg(long)]
    vae_encode_wav: Option<PathBuf>,

    /// Encode raw mono 16-kHz little-endian f32 PCM with AudioVAE V2.
    #[arg(long)]
    vae_encode_pcm16k_f32: Option<PathBuf>,

    /// Audio alignment used before AudioVAE encoding. Reference voice uses right; continuation prompt uses left.
    #[arg(long, value_enum, default_value = "right")]
    vae_pad_side: AudioPadSideArg,

    /// Optional raw little-endian f32 output for --vae-encode-wav (frame-major [T,64]).
    #[arg(long)]
    vae_output_latents_f32: Option<PathBuf>,

    /// Decode raw frame-major AudioVAE latents (N*64 f32 values) to native 48-kHz PCM.
    #[arg(long)]
    vae_decode_latents_f32: Option<PathBuf>,

    /// Write AudioVAE decoded samples as mono 32-bit-float WAV.
    #[arg(long)]
    vae_output_wav: Option<PathBuf>,

    /// Write AudioVAE decoded native 48-kHz mono samples as raw little-endian f32 PCM.
    #[arg(long)]
    vae_output_pcm_f32: Option<PathBuf>,

    /// Encode this WAV and immediately decode its latent representation (AudioVAE roundtrip smoke path).
    #[arg(long)]
    vae_roundtrip_wav: Option<PathBuf>,

    /// Reference speaker WAV. Mutually exclusive with --reference-latents-f32; encoded with right patch padding.
    #[arg(long)]
    reference_wav: Option<PathBuf>,

    /// Continuation/prompt WAV. Mutually exclusive with --prompt-latents-f32; encoded with left patch padding.
    #[arg(long)]
    prompt_wav: Option<PathBuf>,

    /// End-to-end VoxGen text-to-speech input. Activates the step-7 autoregressive pipeline.
    #[arg(long)]
    text: Option<String>,

    /// Exact transcript of --prompt-wav for continuation voice cloning.
    #[arg(long)]
    prompt_text: Option<String>,

    /// VoxCPM2 native natural-language style/emotion instruction. Internally
    /// tokenized as `(control)text`. Compatible with voice design/reference-only
    /// cloning, but intentionally incompatible with prompt/ultimate cloning.
    #[arg(long)]
    control: Option<String>,

    /// VoxGen-managed destination style. This is destination-only; prior state
    /// is never supplied by CLI clients. For persistent cross-phrase continuity,
    /// use the HTTP API with continuity_id.
    #[arg(long)]
    style: Option<String>,

    /// Destination intensity for --style: subtle, normal, or strong.
    #[arg(long, default_value = "normal")]
    intensity: String,

    /// Managed speaking-rate target for --style, as percent of ordinary pace.
    /// This is prosody conditioning, not WSOLA playback speed.
    #[arg(long, default_value_t = 100.0)]
    pace_percent: f32,

    /// Voice-cloning conditioning strategy.
    #[arg(long, value_enum, default_value = "auto")]
    clone_mode: CloneModeArg,

    /// Generate multiple candidate performances with distinct deterministic seeds.
    /// When --output-wav is supplied, candidates receive _v01, _v02, ... suffixes.
    #[arg(long, default_value_t = 1)]
    variations: u32,

    /// Write complete/stream-assembled native 48-kHz float WAV.
    #[arg(long)]
    output_wav: Option<PathBuf>,

    /// Linear output gain applied to emitted speech audio. 1.0 is neutral.
    #[arg(long, default_value_t = 1.0)]
    gain: f32,

    /// Native playback tempo control. 100 keeps the generated duration unchanged.
    #[arg(long = "speed", default_value_t = 100.0)]
    speed_percent: f32,

    /// Native playback pitch shift in semitones, independent of speed.
    #[arg(long = "pitch", default_value_t = 0.0)]
    pitch_semitones: f32,

    /// Enable or disable rolling AudioVAE/HTTP speech streaming. Default: off.
    /// `--stream` by itself remains accepted as an alias for `--stream on`.
    #[arg(
        long,
        value_enum,
        default_value = "off",
        num_args = 0..=1,
        default_missing_value = "on"
    )]
    stream: StreamArg,

    /// Minimum generated acoustic steps before accepting the stop predictor.
    #[arg(long, default_value_t = 2)]
    min_steps: u32,

    /// Hard maximum generated acoustic steps (160 ms per step).
    #[arg(long, default_value_t = 200)]
    max_steps: u32,

    /// Number of recent latent patches decoded for each streaming chunk. VoxCPM2 default is 3.
    #[arg(long, default_value_t = 4)]
    streaming_prefix_len: usize,

    /// Tokenize text with the GGUF-native VoxCPM2 tokenizer and exit.
    #[arg(long)]
    tokenize: Option<String>,

    /// Benchmark FSQ+fusion+ResidualLM using the two residual input files.
    #[arg(long)]
    residual_bench: bool,

    #[arg(long, default_value_t = 2)]
    bench_warmup: u32,
    #[arg(long, default_value_t = 20)]
    bench_iters: u32,

    /// Number of largest vocabulary logits returned by BaseLM smoke/prefill commands.
    #[arg(long, default_value_t = 8)]
    top_k: usize,
}

fn variation_output_path(base:&Path,index:u32,total:u32)->PathBuf{
    if total<=1{return base.to_path_buf();}
    let parent=base.parent().unwrap_or_else(||Path::new(""));
    let stem=base.file_stem().and_then(|x|x.to_str()).unwrap_or("output");
    let ext=base.extension().and_then(|x|x.to_str()).unwrap_or("wav");
    parent.join(format!("{stem}_v{:02}.{ext}",index+1))
}

fn main() -> Result<()> {
    let args = Args::parse();
    let stream_enabled = args.stream.enabled();
    if args.benchmark_profile && stream_enabled {
        bail!("--benchmark-profile is offline-only; use --stream off");
    }
    if args.benchmark_profile && !matches!(args.mode, ModeArg::Xtx7900) {
        bail!("--benchmark-profile requires --mode xtx7900");
    }
    if !args.gain.is_finite() || args.gain < 0.0 {
        bail!("--gain must be a finite value >= 0.0");
    }
    let playback_controls = PlaybackControls::new(args.speed_percent, args.pitch_semitones)
        .map_err(anyhow::Error::msg)?;
    if args.variations == 0 || args.variations > 8 {
        bail!("--variations must be between 1 and 8");
    }
    if !args.cfm_temperature.is_finite() || args.cfm_temperature <= 0.0 {
        bail!("--cfm-temperature/--temperature must be finite and > 0.0");
    }

    if args.list_devices {
        let devices = vulkan::enumerate_devices()?;
        println!("Found {} Vulkan devices", devices.len());
        for d in devices {
            println!(
                "Vulkan{}: {} vendor=0x{:04x} device=0x{:04x} type={} api={} float16={} storage16={} subgroup={} subgroup_arithmetic={} local_mem={:.2}GiB max_storage={:.2}GiB",
                d.index,
                d.name,
                d.vendor_id,
                d.device_id,
                d.device_type,
                d.api_version,
                d.shader_float16,
                d.storage_buffer_16bit,
                d.subgroup_size,
                d.subgroup_arithmetic,
                d.local_heap_bytes as f64 / 1073741824.0,
                d.max_storage_buffer_range as f64 / 1073741824.0,
            );
        }
        return Ok(());
    }

    if args.n_gpu_layers != -1 {
        bail!("VoxGen requires --n-gpu-layers -1. Partial GPU offload/CPU fallback is intentionally unsupported.");
    }
    let requested = match args.base_format {
        BaseFormatArg::Auto => None,
        BaseFormatArg::Q8_0 => Some(BaseFormat::Q8_0),
        BaseFormatArg::F16 => Some(BaseFormat::F16),
    };
    let execution_mode: ExecutionMode = args.mode.into();
    let xtx_tuning = XtxTuning {
        gpu_profile: args.gpu_profile.enabled() || args.benchmark_profile,
        cooperative_matrix: args.xtx_coopmat.enabled(),
    };

    // Lifecycle-server mode: keep Vulkan/model initialization out of startup and
    // let a local client select the exact GGUF paths via POST /v1/models/load.
    // This is what the wxDragon demo uses so model selection is explicit.
    if args.server && args.base_lm.is_none() {
        if args.acoustic.is_some() {
            bail!("--acoustic cannot be used without --base-lm at startup; start with --server and POST both paths to /v1/models/load instead");
        }
        eprintln!("[VoxGen] starting model-lifecycle server without loaded weights");
        return http::serve(
            &args.host,
            args.port,
            None,
            requested,
            args.gpu,
            execution_mode,
            xtx_tuning,
            args.max_context,
            stream_enabled,
            args.gain,
        );
    }

    let base = args.base_lm.context("--base-lm is required unless --list-devices is used or --server starts an empty model-lifecycle server")?;

    if args.inspect_base_lm || args.inspect_acoustic {
        let base_summary = gguf::load_summary(&base)?;
        let format = base_summary.validate_baselm(requested)?;
        let base_config = BaseLmConfig::from_gguf(&base_summary, args.max_context)?;
        if args.inspect_acoustic {
            let acoustic_path = args.acoustic.as_ref().context("--inspect-acoustic requires --acoustic")?;
            let acoustic_summary = gguf::load_summary(acoustic_path)?;
            acoustic_summary.validate_acoustic_f16()?;
            let acoustic_config = AcousticConfig::from_gguf(&acoustic_summary, &base_config, args.max_context)?;
            let local_config = LocalConfig::from_gguf(&acoustic_summary, &base_config)?;
            let audio_vae = AudioVaeConfig::from_gguf(&acoustic_summary)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "engine": "VoxGen",
                    "iteration": 7,
                    "base_format": format,
                    "base_lm": base_config,
                    "acoustic_gguf": acoustic_summary,
                    "residual_fsq": acoustic_config,
                    "local": local_config,
                    "audio_vae": audio_vae,
                    "latent_conditioning": true,
                    "cfm_solver": true,
                    "wav_conditioning": true,
                    "native_pcm_decode": true,
                    "gpu_allocated": false,
                    "no_cpu_fallback": true
                }))?
            );
        } else {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "engine": "VoxGen",
                    "iteration": 7,
                    "base_format": format,
                    "gguf": base_summary,
                    "baselm": base_config,
                    "gpu_allocated": false,
                    "no_cpu_fallback": true
                }))?
            );
        }
        return Ok(());
    }

    let runtime = Arc::new(Runtime::load(
        &base,
        args.acoustic.as_deref(),
        requested,
        args.gpu,
        execution_mode,
        xtx_tuning,
        args.max_context,
    )?);
    let status = runtime.status();
    eprintln!("[VoxGen] GPU: {} (Vulkan{})", status.gpu.name, status.gpu.index);
    eprintln!("[VoxGen] Mode: {}", status.execution_mode);
    if status.execution_mode == "xtx7900" {
        eprintln!("[VoxGen] RX 7900 XTX stream-safe kernels enabled · live32/profile16 cross-engine prefill batching · x4 linear lanes · forced wave32 · cooperative matrix {} · GPU timestamps {}",
            if runtime.gpu.xtx_coopmat_enabled() { "experimental on" } else { "off (subgroup fallback)" },
            if status.gpu_profile.enabled { "on (benchmarking)" } else { "off" });
    } else { eprintln!("[VoxGen] Generic Vulkan kernels enabled"); }
    eprintln!(
        "[VoxGen] BaseLM: {} ({}) · MiniCPM4 {}L/{}D · context {}",
        status.base_lm.path.display(),
        status.base_format.as_str(),
        status.baselm.config.block_count,
        status.baselm.config.embedding_length,
        status.baselm.config.active_context_length,
    );
    if let Some(a) = status.residual_fsq.as_ref() {
        eprintln!(
            "[VoxGen] ResidualLM: {}L/{}D · no_rope={} · context {} · FSQ {}D scale {}",
            a.config.residual_block_count,
            a.config.embedding_length,
            a.config.no_rope,
            a.config.active_context_length,
            a.config.fsq_latent_dim,
            a.config.fsq_scale,
        );
    }
    if let Some(l) = status.local.as_ref() {
        eprintln!(
            "[VoxGen] LocEnc: {}L/{}D/{} tokens · LocDiT: {}L/{}D/{} tokens · UnifiedCFM Euler/CFG-Zero* ready · default cfg {}",
            l.config.locenc_layers, l.config.locenc_hidden, l.config.locenc_tokens,
            l.config.locdit_layers, l.config.locdit_hidden, l.config.locdit_tokens, l.config.cfm_cfg_rate,
        );
    }
    if let Some(v) = status.audio_vae.as_ref() {
        eprintln!(
            "[VoxGen] AudioVAE V2: {} Hz -> latent/{} -> {} Hz · enc hop {} · dec hop {} · patch {} ms",
            v.config.sample_rate, v.config.latent_dim, v.config.out_sample_rate, v.config.encoder_hop, v.config.decoder_hop,
            (1000u32 * v.config.input_samples_per_patch / v.config.sample_rate),
        );
    }
    eprintln!(
        "[VoxGen] GPU allocations: {:.2} GiB total (BaseLM {:.2} GiB, Acoustic/Residual {:.2} GiB, LocEnc/LocDiT {:.3} GiB, AudioVAE dynamic {:.3} GiB)",
        status.memory.total_allocated_bytes as f64 / 1073741824.0,
        status.memory.baselm_allocated_bytes as f64 / 1073741824.0,
        status.memory.acoustic_allocated_bytes.unwrap_or(0) as f64 / 1073741824.0,
        status.memory.local_allocated_bytes.unwrap_or(0) as f64 / 1073741824.0,
        status.memory.audiovae_dynamic_scratch_bytes as f64 / 1073741824.0,
    );
    eprintln!("[VoxGen] CPU fallback: DISABLED");

    if args.diagnostics_json {
        println!("{}", serde_json::to_string_pretty(&runtime.status())?);
        return Ok(());
    }

    if let Some(text)=args.tokenize.as_deref(){
        let tokens=runtime.tokenize(text)?;
        println!("{}",serde_json::to_string_pretty(&json!({"text":text,"tokens":tokens,"tokenizer":runtime.status().tokenizer}))?);
        return Ok(());
    }

    if args.vae_encode_wav.is_some() && args.vae_encode_pcm16k_f32.is_some() { bail!("use either --vae-encode-wav or --vae-encode-pcm16k-f32, not both"); }
    if let Some(path)=args.vae_encode_pcm16k_f32.as_ref() {
        let pcm=audiovae::read_f32_file(path)?;
        let (stats,latents)=runtime.audiovae_encode_pcm16k(&pcm,args.vae_pad_side.into())?;
        if let Some(out)=args.vae_output_latents_f32.as_ref(){audiovae::write_f32_file(out,&latents)?;}
        println!("{}",serde_json::to_string_pretty(&json!({"mode":"audiovae_encode_pcm16k","stats":stats,"output_latents_f32":args.vae_output_latents_f32}))?);
        return Ok(());
    }
    if let Some(path)=args.vae_encode_wav.as_ref() {
        let (stats,latents)=runtime.audiovae_encode_wav(path,args.vae_pad_side.into())?;
        if let Some(out)=args.vae_output_latents_f32.as_ref(){audiovae::write_f32_file(out,&latents)?;}
        println!("{}",serde_json::to_string_pretty(&json!({"mode":"audiovae_encode","stats":stats,"output_latents_f32":args.vae_output_latents_f32}))?);
        return Ok(());
    }
    if let Some(path)=args.vae_decode_latents_f32.as_ref() {
        let latents=audiovae::read_f32_file(path)?;
        let (stats,wave)=runtime.audiovae_decode_latents(&latents)?;
        if let Some(out)=args.vae_output_wav.as_ref(){audiovae::write_wav_f32(out,&wave,stats.output_sample_rate)?;}
        if let Some(out)=args.vae_output_pcm_f32.as_ref(){audiovae::write_f32_file(out,&wave)?;}
        println!("{}",serde_json::to_string_pretty(&json!({"mode":"audiovae_decode","stats":stats,"output_wav":args.vae_output_wav,"output_pcm_f32":args.vae_output_pcm_f32}))?);
        return Ok(());
    }
    if let Some(path)=args.vae_roundtrip_wav.as_ref() {
        let (encode,latents)=runtime.audiovae_encode_wav(path,args.vae_pad_side.into())?;
        if let Some(out)=args.vae_output_latents_f32.as_ref(){audiovae::write_f32_file(out,&latents)?;}
        let (decode,wave)=runtime.audiovae_decode_latents(&latents)?;
        if let Some(out)=args.vae_output_wav.as_ref(){audiovae::write_wav_f32(out,&wave,decode.output_sample_rate)?;}
        if let Some(out)=args.vae_output_pcm_f32.as_ref(){audiovae::write_f32_file(out,&wave)?;}
        println!("{}",serde_json::to_string_pretty(&json!({"mode":"audiovae_roundtrip","encode":encode,"decode":decode,"output_wav":args.vae_output_wav,"output_pcm_f32":args.vae_output_pcm_f32,"output_latents_f32":args.vae_output_latents_f32}))?);
        return Ok(());
    }

    if let Some(path) = args.locenc_patch_f32.as_ref() {
        let patch = read_exact_f32(path, 256, "LocEnc latent patch")?;
        let result = runtime.locenc_patch(&patch)?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    let cfm_default_cfg = runtime.default_cfm_cfg_rate();
    let cfm_options = CfmOptions { n_timesteps:args.cfm_steps, cfg_value:args.cfm_cfg.unwrap_or(cfm_default_cfg), temperature:args.cfm_temperature, sway_sampling_coef:args.cfm_sway, seed:args.cfm_seed, use_cfg_zero_star:!args.cfm_no_zero_star };

    if let Some(text)=args.text.as_deref(){
        if args.reference_latents_f32.is_some()||args.prompt_latents_f32.is_some(){bail!("end-to-end --text currently accepts WAV conditioning; latent smoke inputs remain under --conditioning-text-tokens");}
        let mut reference_wav=args.reference_wav.clone();
        let mut prompt_wav=args.prompt_wav.clone();
        match args.clone_mode {
            CloneModeArg::Auto => {}
            CloneModeArg::Reference => {
                if reference_wav.is_none(){bail!("--clone-mode reference requires --reference-wav");}
                if prompt_wav.is_some() || args.prompt_text.as_deref().is_some_and(|x|!x.trim().is_empty()){bail!("--clone-mode reference cannot use --prompt-wav/--prompt-text");}
            }
            CloneModeArg::Ultimate => {
                if args.prompt_text.as_deref().map(str::trim).unwrap_or("").is_empty(){bail!("--clone-mode ultimate requires --prompt-text with the exact reference transcript");}
                let source=prompt_wav.clone().or_else(||reference_wav.clone()).context("--clone-mode ultimate requires --prompt-wav or --reference-wav")?;
                if prompt_wav.is_none(){prompt_wav=Some(source.clone());}
                if reference_wav.is_none(){reference_wav=Some(source);}
            }
        }
        if args.style.is_some() && args.control.as_deref().is_some_and(|x|!x.trim().is_empty()) {
            bail!("--style cannot be combined with --control");
        }
        if !args.pace_percent.is_finite() || !(50.0..=200.0).contains(&args.pace_percent) {
            bail!("--pace-percent must be finite and between 50 and 200");
        }
        if args.style.is_none() && (args.intensity != "normal" || (args.pace_percent - 100.0).abs() > 1.0e-4) {
            bail!("--intensity/--pace-percent require --style");
        }
        if args.style.is_some() && (args.speed_percent - 100.0).abs() > 1.0e-4 {
            bail!("--speed must remain 100 when --style/--pace-percent managed prosody is active; use --pace-percent instead");
        }
        let managed_control = if let Some(style) = args.style.as_deref() {
            let mut effective = build_style_control(style, &args.intensity, "", text)
                .ok_or_else(|| anyhow::anyhow!("unsupported --style/--intensity combination"))?;
            if (args.pace_percent - 100.0).abs() >= 0.001 {
                effective.push_str(&format!(" Speaking pace: {:.0}%.", args.pace_percent));
            }
            let tuning = managed_style_tuning(style, &args.intensity)
                .ok_or_else(|| anyhow::anyhow!("unsupported --style/--intensity combination"))?;
            Some((effective, tuning.cfg_delta))
        } else { None };
        let effective_control = managed_control.as_ref().map(|(effective, _)| effective.as_str()).or(args.control.as_deref());
        if effective_control.is_some_and(|x|!x.trim().is_empty()) && prompt_wav.is_some(){
            bail!("--control/--style cannot be combined with prompt/ultimate cloning");
        }
        let mut text_cfm_options = cfm_options.clone();
        if args.cfm_cfg.is_none() {
            if let Some((_, cfg_delta)) = managed_control.as_ref() {
                text_cfm_options.cfg_value = (cfm_default_cfg + *cfg_delta).clamp(1.0, 3.0);
            }
        }
        let mut reports=Vec::new();
        for variation in 0..args.variations {
            let mut tts=TtsOptions{min_steps:args.min_steps,max_steps:args.max_steps,streaming_prefix_len:args.streaming_prefix_len,cfm:text_cfm_options.clone()};
            if variation>0 { tts.cfm.seed=tts.cfm.seed.wrapping_add((variation as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)); }
            let result=if stream_enabled {
                let mut streamed_samples=Vec::new();
                let mut result=runtime.synthesize(text,effective_control,args.prompt_text.as_deref(),reference_wav.as_deref(),prompt_wav.as_deref(),&tts,Some(|chunk:&[f32],_sr:u32|->Result<()>{streamed_samples.extend_from_slice(chunk);Ok(())}))?;
                result.samples=streamed_samples;
                result
            } else {
                runtime.synthesize::<fn(&[f32],u32)->Result<()>>(text,effective_control,args.prompt_text.as_deref(),reference_wav.as_deref(),prompt_wav.as_deref(),&tts,None)?
            };
            let playback_samples=StreamingPlaybackDsp::process_all(result.sample_rate, playback_controls, &result.samples)
                .map_err(anyhow::Error::msg)?;
            let output_samples=OutputPeakGuard::process_all(result.sample_rate, &playback_samples, args.gain)
                .map_err(anyhow::Error::msg)?;
            let output_path=args.output_wav.as_ref().map(|p|variation_output_path(p,variation,args.variations));
            if let Some(out)=output_path.as_ref(){audiovae::write_wav_f32(out,&output_samples,result.sample_rate)?;}
            reports.push(json!({"variation":variation+1,"seed":tts.cfm.seed,"output_wav":output_path,"result":result}));
        }
        let gpu_profile=runtime.status().gpu_profile;
        let managed_style_json = args.style.as_ref().map(|style| json!({
            "style": style,
            "intensity": args.intensity,
            "pace_percent": args.pace_percent,
        }));
        println!("{}",serde_json::to_string_pretty(&json!({"mode":"tts","streaming":stream_enabled,"gain":args.gain,"speed_percent":args.speed_percent,"pitch_semitones":args.pitch_semitones,"control":effective_control,"managed_style":managed_style_json,"clone_mode":format!("{:?}",args.clone_mode).to_ascii_lowercase(),"variations":reports,"gpu_profile":gpu_profile}))?);
        return Ok(());
    } else if args.output_wav.is_some() || args.prompt_text.is_some() || args.control.is_some() || args.style.is_some() || args.intensity != "normal" || (args.pace_percent - 100.0).abs() > 1.0e-4 { bail!("--output-wav/--prompt-text/--control/--style/--intensity/--pace-percent require --text"); }

    if args.cfm_mu_f32.is_some() || args.cfm_cond_f32.is_some() || args.cfm_bench {
        let m=args.cfm_mu_f32.as_ref().context("--cfm-mu-f32 and --cfm-cond-f32 must be provided together")?;
        let c=args.cfm_cond_f32.as_ref().context("--cfm-mu-f32 and --cfm-cond-f32 must be provided together")?;
        let mu=read_exact_f32(m,2048,"CFM mu")?; let cond=read_exact_f32(c,256,"CFM condition patch")?;
        let initial=if let Some(p)=args.cfm_initial_x_f32.as_ref(){Some(read_exact_f32(p,256,"CFM initial x")?)}else{None};
        if args.cfm_bench {
            if args.bench_iters==0 { bail!("--bench-iters must be >= 1"); }
            for _ in 0..args.bench_warmup { let _=runtime.cfm_cpu_mu(&cond,&mu,&cfm_options,initial.as_deref())?; }
            let mut ms=Vec::new(); let mut last=None;
            for _ in 0..args.bench_iters { let st=Instant::now(); last=Some(runtime.cfm_cpu_mu(&cond,&mu,&cfm_options,initial.as_deref())?); ms.push(st.elapsed().as_secs_f64()*1000.0); }
            let mean=ms.iter().sum::<f64>()/ms.len() as f64; let min=ms.iter().copied().fold(f64::INFINITY,f64::min); let max=ms.iter().copied().fold(f64::NEG_INFINITY,f64::max);
            println!("{}",serde_json::to_string_pretty(&json!({"mode":"cfm_benchmark","warmup":args.bench_warmup,"iterations":args.bench_iters,"mean_ms":mean,"min_ms":min,"max_ms":max,"options":cfm_options,"last":last}))?);
        } else {
            let result=runtime.cfm_cpu_mu(&cond,&mu,&cfm_options,initial.as_deref())?;
            if let Some(p)=args.cfm_output_f32.as_ref(){write_f32_values(p,&result.output_patch)?;}
            println!("{}",serde_json::to_string_pretty(&result)?);
        }
        return Ok(());
    }

    if args.locdit_x_f32.is_some() || args.locdit_cond_f32.is_some() || args.locdit_mu_f32.is_some() {
        let x_path=args.locdit_x_f32.as_ref().context("--locdit-x-f32, --locdit-cond-f32 and --locdit-mu-f32 must be provided together")?;
        let c_path=args.locdit_cond_f32.as_ref().context("--locdit-x-f32, --locdit-cond-f32 and --locdit-mu-f32 must be provided together")?;
        let m_path=args.locdit_mu_f32.as_ref().context("--locdit-x-f32, --locdit-cond-f32 and --locdit-mu-f32 must be provided together")?;
        let x=read_exact_f32(x_path,256,"LocDiT x patch")?;
        let cond=read_exact_f32(c_path,256,"LocDiT condition patch")?;
        let mu=read_exact_f32(m_path,2048,"LocDiT mu")?;
        let result=runtime.locdit_cpu_mu(&x,&cond,&mu,args.locdit_t,args.locdit_dt)?;
        println!("{}",serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    if let Some(tokens_text)=args.conditioning_text_tokens.as_deref() {
        if args.reference_wav.is_some() && args.reference_latents_f32.is_some(){bail!("use either --reference-wav or --reference-latents-f32, not both");}
        if args.prompt_wav.is_some() && args.prompt_latents_f32.is_some(){bail!("use either --prompt-wav or --prompt-latents-f32, not both");}
        let tokens=parse_tokens(tokens_text)?;
        let (reference,reference_audio_stats)=if let Some(p)=args.reference_wav.as_ref(){let(st,v)=runtime.audiovae_encode_wav_patches(p,AudioPadSide::Right)?;(v,Some(st))}else if let Some(p)=args.reference_latents_f32.as_ref(){(conditioning::split_patches(read_f32_values(p)?,"reference latent file")?,None)}else{(Vec::new(),None)};
        let (prompt,prompt_audio_stats)=if let Some(p)=args.prompt_wav.as_ref(){let(st,v)=runtime.audiovae_encode_wav_patches(p,AudioPadSide::Left)?;(v,Some(st))}else if let Some(p)=args.prompt_latents_f32.as_ref(){(conditioning::split_patches(read_f32_values(p)?,"prompt latent file")?,None)}else{(Vec::new(),None)};
        if args.cfm_after_conditioning {
            let initial=if let Some(p)=args.cfm_initial_x_f32.as_ref(){Some(read_exact_f32(p,256,"CFM initial x")?)}else{None};
            let result=runtime.prefill_latent_conditioning_and_cfm(&tokens,&reference,&prompt,&cfm_options,initial.as_deref())?;
            if let Some(p)=args.cfm_output_f32.as_ref(){write_f32_values(p,&result.cfm.output_patch)?;}
            println!("{}",serde_json::to_string_pretty(&json!({"conditioning":result,"reference_audio":reference_audio_stats,"prompt_audio":prompt_audio_stats}))?);
        } else {
            let result=runtime.prefill_latent_conditioning(&tokens,&reference,&prompt)?;
            println!("{}",serde_json::to_string_pretty(&json!({"conditioning":result,"reference_audio":reference_audio_stats,"prompt_audio":prompt_audio_stats}))?);
        }
        return Ok(());
    } else if args.reference_latents_f32.is_some() || args.prompt_latents_f32.is_some() || args.reference_wav.is_some() || args.prompt_wav.is_some() {
        bail!("--reference-wav/--prompt-wav/latent conditioning inputs require --conditioning-text-tokens");
    }

    if args.cfm_after_conditioning { bail!("--cfm-after-conditioning requires --conditioning-text-tokens"); }
    if args.vae_output_latents_f32.is_some() && args.vae_encode_wav.is_none() && args.vae_encode_pcm16k_f32.is_none() && args.vae_roundtrip_wav.is_none(){bail!("--vae-output-latents-f32 requires --vae-encode-wav, --vae-encode-pcm16k-f32 or --vae-roundtrip-wav");}
    if args.vae_output_wav.is_some() && args.vae_decode_latents_f32.is_none() && args.vae_roundtrip_wav.is_none(){bail!("--vae-output-wav requires --vae-decode-latents-f32 or --vae-roundtrip-wav");}
    if args.vae_output_pcm_f32.is_some() && args.vae_decode_latents_f32.is_none() && args.vae_roundtrip_wav.is_none(){bail!("--vae-output-pcm-f32 requires --vae-decode-latents-f32 or --vae-roundtrip-wav");}
    if args.cfm_initial_x_f32.is_some() || args.cfm_output_f32.is_some() || args.cfm_cfg.is_some() || args.cfm_no_zero_star { bail!("CFM options require --cfm-mu-f32/--cfm-cond-f32 or --cfm-after-conditioning"); }

    if let Some(path) = args.fsq_input_f32.as_ref() {
        let input = read_raw_embedding(path)?;
        let result = runtime.fsq_only(&input)?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    if args.residual_base_hidden_f32.is_some() || args.residual_current_embedding_f32.is_some() {
        let base_path = args.residual_base_hidden_f32.as_ref().context("--residual-base-hidden-f32 and --residual-current-embedding-f32 must be provided together")?;
        let current_path = args.residual_current_embedding_f32.as_ref().context("--residual-base-hidden-f32 and --residual-current-embedding-f32 must be provided together")?;
        let base_hidden = read_raw_embedding(base_path)?;
        let current = read_raw_embedding(current_path)?;
        runtime.reset_residual();
        if args.residual_bench {
            let result = runtime.benchmark_residual(&base_hidden, &current, args.bench_warmup, args.bench_iters)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            let result = runtime.residual_step(&base_hidden, &current)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        return Ok(());
    } else if args.residual_bench {
        bail!("--residual-bench requires --residual-base-hidden-f32 and --residual-current-embedding-f32");
    }

    if let Some(list) = args.base_residual_text_prefill.as_deref() {
        let tokens = parse_tokens(list)?;
        runtime.reset_pipeline();
        let mut last = None;
        for token in &tokens {
            last = Some(runtime.base_residual_text_token_step(*token)?);
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "mode": "voxcpm2_text_prefix_prefill",
                "tokens": tokens,
                "last_step": last,
                "baselm_position": runtime.status().baselm.position,
                "residual_position": runtime.status().residual_fsq.map(|x| x.residual_position)
            }))?
        );
        return Ok(());
    }

    if let Some(path) = args.base_residual_embedding_f32.as_ref() {
        let embedding = read_raw_embedding(path)?;
        runtime.reset_pipeline();
        let result = runtime.base_residual_step(&embedding)?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    if let Some(token) = args.baselm_token {
        runtime.reset_baselm();
        let result = runtime.decode_token(token, args.top_k)?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    if let Some(list) = args.baselm_prefill.as_deref() {
        let tokens = parse_tokens(list)?;
        runtime.reset_baselm();
        let result = runtime.prefill_tokens(&tokens, args.top_k)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "tokens": tokens,
                "last_step": result,
                "final_position": runtime.status().baselm.position
            }))?
        );
        return Ok(());
    }

    if let Some(path) = args.baselm_embedding_f32.as_ref() {
        let embedding = read_raw_embedding(path)?;
        runtime.reset_baselm();
        let result = runtime.decode_embedding(&embedding, args.top_k)?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    if let Some(token) = args.baselm_bench_token {
        let result = runtime.benchmark_baselm(token, args.bench_warmup, args.bench_iters)?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    http::serve(
        &args.host,
        args.port,
        Some(runtime),
        requested,
        args.gpu,
        execution_mode,
        xtx_tuning,
        args.max_context,
        stream_enabled,
        args.gain,
    )
}

fn parse_tokens(text: &str) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for (i, part) in text.split(',').enumerate() {
        let p = part.trim();
        if p.is_empty() { continue; }
        out.push(p.parse::<u32>().with_context(|| format!("invalid token at comma-separated field {}: {p:?}", i + 1))?);
    }
    if out.is_empty() { bail!("token list did not contain any token IDs"); }
    Ok(out)
}

fn read_f32_values(path: &PathBuf) -> Result<Vec<f32>> {
    let bytes=fs::read(path).with_context(||format!("read {}",path.display()))?;
    if bytes.len()%4!=0 { bail!("{} has {} bytes, not a whole number of little-endian f32 values",path.display(),bytes.len()); }
    Ok(bytes.chunks_exact(4).map(|b|f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect())
}
fn read_exact_f32(path:&PathBuf,count:usize,label:&str)->Result<Vec<f32>> { let v=read_f32_values(path)?;if v.len()!=count{bail!("{} contains {} floats; {label} requires exactly {count}",path.display(),v.len())}Ok(v) }
fn read_raw_embedding(path: &PathBuf) -> Result<Vec<f32>> { read_exact_f32(path,2048,"VoxGen embedding") }

fn write_f32_values(path:&PathBuf,values:&[f32])->Result<()> {
    let mut bytes=Vec::with_capacity(values.len()*4);for &v in values{bytes.extend_from_slice(&v.to_le_bytes());}
    fs::write(path,&bytes).with_context(||format!("write {}",path.display()))?;Ok(())
}
