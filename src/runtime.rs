use crate::{
    acoustic::{AcousticConfig, FsqResult, ResidualBenchmark, ResidualFsqEngine, ResidualStepResult, StopPrediction},
    audiovae::{AudioDecodeStats, AudioEncodeStats, AudioPadSide, AudioVaeEngine, AudioVaeState},
    baselm::{BaseLmBenchmark, BaseLmConfig, BaseLmEngine, BaseLmGpuStep, BaseLmStepResult},
    conditioning::{self, ConditioningPlanSummary, PrefixPosition},
    gguf::{self, BaseFormat, GgufSummary},
    local::{CfmOptions, CfmResult, LocDitResult, LocEncResult, LocalConfig, LocalEngine},
    profiler::Profiler,
    tokenizer::{TokenizerInfo, VoxTokenizer},
    vulkan::{ExecutionMode, VulkanContext, XtxTuning},
};
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, Serialize)]
pub struct MemoryPlan {
    pub baselm_gguf_data_bytes: u64,
    pub baselm_kv_cache_bytes: u64,
    pub baselm_allocated_bytes: u64,
    pub acoustic_gguf_data_bytes: Option<u64>,
    pub residual_kv_cache_bytes: Option<u64>,
    pub acoustic_allocated_bytes: Option<u64>,
    pub local_allocated_bytes: Option<u64>,
    pub audiovae_dynamic_scratch_bytes: u64,
    pub total_allocated_bytes: u64,
    pub device_local_heap_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BaseLmState {
    pub ready: bool,
    pub position: u32,
    pub config: BaseLmConfig,
}

#[derive(Debug, Clone, Serialize)]
pub struct AcousticState {
    pub ready: bool,
    pub residual_position: u32,
    pub config: AcousticConfig,
}

#[derive(Debug, Clone, Serialize)]
pub struct LocalState {
    pub ready: bool,
    pub config: LocalConfig,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConditioningPrefillResult {
    pub plan: ConditioningPlanSummary,
    pub baselm_position: u32,
    pub residual_position: u32,
    pub prefix_condition_checksum: f64,
    pub prefix_condition_l2: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConditioningCfmResult {
    pub prefill: ConditioningPrefillResult,
    pub cfm: CfmResult,
}

#[derive(Debug, Clone, Serialize)]
pub struct TtsOptions {
    pub min_steps:u32, pub max_steps:u32, pub streaming_prefix_len:usize, pub cfm:CfmOptions,
}
impl TtsOptions { pub fn validate(&self)->Result<()> { self.cfm.validate()?; if self.max_steps==0{bail!("TTS max_steps must be >=1");} if self.min_steps>=self.max_steps{bail!("TTS min_steps must be smaller than max_steps");} if self.streaming_prefix_len==0{bail!("streaming_prefix_len must be >=1");} Ok(()) } }

#[derive(Debug, Clone, Serialize)]
pub struct TtsStepTrace { pub step:u32, pub cfm_ms:f64, pub stop:StopPrediction, pub emitted_samples:usize }

#[derive(Debug, Clone, Serialize)]
pub struct TtsResult {
    pub text:String, pub control:Option<String>, pub token_count:usize, pub generated_patches:usize, pub stopped_by_predictor:bool,
    pub sample_rate:u32, pub sample_count:usize, pub audio_seconds:f64, pub elapsed_ms:f64, pub rtf:f64,
    pub first_pcm_ms:Option<f64>, pub conditioning:ConditioningPlanSummary, pub steps:Vec<TtsStepTrace>,
    #[serde(skip)] pub samples:Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct RuntimeStatus<'a> {
    pub engine: &'static str,
    pub implementation_iteration: u32,
    pub baselm_inference_ready: bool,
    pub fsq_inference_ready: bool,
    pub residual_lm_inference_ready: bool,
    pub locenc_inference_ready: bool,
    pub locdit_estimator_ready: bool,
    pub cfm_solver_ready: bool,
    pub audiovae_encoder_ready: bool,
    pub audiovae_decoder_ready: bool,
    pub latent_conditioning_ready: bool,
    pub wav_conditioning_ready: bool,
    pub speech_inference_ready: bool,
    pub base_format: BaseFormat,
    pub base_lm: &'a GgufSummary,
    pub acoustic: Option<&'a GgufSummary>,
    pub baselm: BaseLmState,
    pub residual_fsq: Option<AcousticState>,
    pub local: Option<LocalState>,
    pub audio_vae: Option<AudioVaeState>,
    pub gpu: &'a crate::vulkan::DeviceInfo,
    pub execution_mode: &'static str,
    pub memory: MemoryPlan,
    pub profile: crate::profiler::ProfileSnapshot,
    pub gpu_profile: crate::vulkan::GpuProfileSnapshot,
    pub tokenizer: TokenizerInfo,
    pub no_cpu_fallback: bool,
}


#[derive(Debug, Clone, PartialEq, Eq)]
struct AudioConditioningCacheKey {
    path: PathBuf,
    size: u64,
    modified_ns: Option<u128>,
    pad_side: AudioPadSide,
}

#[derive(Debug, Clone)]
struct AudioConditioningCacheEntry {
    key: AudioConditioningCacheKey,
    stats: AudioEncodeStats,
    patches: Vec<Vec<f32>>,
}

fn audio_conditioning_cache_key(path: &Path, pad_side: AudioPadSide) -> Result<AudioConditioningCacheKey> {
    let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let meta = fs::metadata(&canonical)
        .with_context(|| format!("read conditioning audio metadata {}", canonical.display()))?;
    if !meta.is_file() {
        bail!("conditioning audio is not a file: {}", canonical.display());
    }
    let modified_ns = meta.modified().ok().and_then(|t: SystemTime| {
        t.duration_since(UNIX_EPOCH).ok().map(|d| d.as_nanos())
    });
    Ok(AudioConditioningCacheKey {
        path: canonical,
        size: meta.len(),
        modified_ns,
        pad_side,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct BaseResidualStepResult {
    pub baselm: BaseLmGpuStep,
    pub residual_fsq: ResidualStepResult,
}

// Drop order is intentional. AudioVAE and Local descriptor sets reference the acoustic model-data buffer,
// so they are destroyed before Residual/FSQ, then BaseLM, then Vulkan.
pub struct Runtime {
    // Reference/prompt WAVs are normally reused across many Speak operations.
    // Cache their AudioVAE patches by stable file identity so request TTFA does
    // not repeatedly pay WAV read/resample/encoder cost. The demo prewarms the
    // selected reference after model load, so even the first Speak can hit it.
    conditioning_audio_cache: Mutex<Vec<AudioConditioningCacheEntry>>,
    audiovae_engine: Mutex<Option<AudioVaeEngine>>,
    local_engine: Mutex<Option<LocalEngine>>,
    acoustic_engine: Mutex<Option<ResidualFsqEngine>>,
    baselm: Mutex<Option<BaseLmEngine>>,
    pub gpu: VulkanContext,
    pub base: GgufSummary,
    pub acoustic: Option<GgufSummary>,
    pub base_format: BaseFormat,
    pub profiler: Arc<Profiler>,
    pub memory: MemoryPlan,
    tokenizer: VoxTokenizer,
}

impl Runtime {
    pub fn load(
        base: &Path,
        acoustic: Option<&Path>,
        requested: Option<BaseFormat>,
        gpu_index: Option<usize>,
        mode: ExecutionMode,
        xtx_tuning: XtxTuning,
        max_context: u32,
    ) -> Result<Self> {
        let profiler = Arc::new(Profiler::default());
        let base_summary = {
            let _s = profiler.scope("gguf_baselm_index");
            gguf::load_summary(base)?
        };
        let base_format = base_summary.validate_baselm(requested)?;
        let tokenizer = VoxTokenizer::from_gguf(&base_summary)?;
        let acoustic_summary = if let Some(path) = acoustic {
            let _s = profiler.scope("gguf_acoustic_index");
            let s = gguf::load_summary(path)?;
            s.validate_acoustic_f16()?;
            Some(s)
        } else {
            None
        };
        let gpu = {
            let _s = profiler.scope("vulkan_init");
            VulkanContext::new(gpu_index, mode, xtx_tuning)?
        };
        let base_engine = {
            let _s = profiler.scope("baselm_gpu_init");
            BaseLmEngine::new(&gpu, &base_summary, base_format, max_context)?
        };
        let baselm_allocated = base_engine.allocated_bytes();
        let baselm_kv = base_engine.config.kv_cache_bytes();

        let acoustic_engine = if let Some(s) = acoustic_summary.as_ref() {
            let _scope = profiler.scope("residual_fsq_gpu_init");
            Some(ResidualFsqEngine::new(&gpu, s, &base_engine.config, max_context)?)
        } else {
            None
        };
        let audiovae_engine = if let Some(s) = acoustic_summary.as_ref() {
            let _scope = profiler.scope("audiovae_gpu_init");
            Some(AudioVaeEngine::new(&gpu, s)?)
        } else { None };
        let local_engine = match (acoustic_summary.as_ref(), acoustic_engine.as_ref()) {
            (Some(s), Some(ae)) => {
                let _scope = profiler.scope("locenc_locdit_gpu_init");
                Some(LocalEngine::new(
                    &gpu, s, &base_summary, &base_engine.config,
                    ae.model_data_buffer(), base_engine.model_data_buffer(),
                    ae.current_lm_buffer(), ae.output_buffer(),
                )?)
            }
            _ => None,
        };
        let acoustic_allocated = acoustic_engine.as_ref().map(ResidualFsqEngine::allocated_bytes);
        let local_allocated = local_engine.as_ref().map(LocalEngine::allocated_bytes);
        let residual_kv = acoustic_engine.as_ref().map(|x| x.config.kv_cache_bytes());
        let total_allocated = baselm_allocated
            .saturating_add(acoustic_allocated.unwrap_or(0))
            .saturating_add(local_allocated.unwrap_or(0));
        let memory = MemoryPlan {
            baselm_gguf_data_bytes: base_summary.data_bytes()?,
            baselm_kv_cache_bytes: baselm_kv,
            baselm_allocated_bytes: baselm_allocated,
            acoustic_gguf_data_bytes: acoustic_summary.as_ref().map(GgufSummary::data_bytes).transpose()?,
            residual_kv_cache_bytes: residual_kv,
            acoustic_allocated_bytes: acoustic_allocated,
            local_allocated_bytes: local_allocated,
            audiovae_dynamic_scratch_bytes: 0,
            total_allocated_bytes: total_allocated,
            device_local_heap_bytes: gpu.info.local_heap_bytes,
        };
        if total_allocated > gpu.info.local_heap_bytes {
            bail!(
                "VoxGen allocations require {:.2} GiB but selected GPU '{}' reports {:.2} GiB device-local memory. CPU fallback is disabled.",
                total_allocated as f64 / 1073741824.0,
                gpu.info.name,
                gpu.info.local_heap_bytes as f64 / 1073741824.0
            );
        }
        Ok(Self {
            conditioning_audio_cache: Mutex::new(Vec::new()),
            audiovae_engine: Mutex::new(audiovae_engine),
            local_engine: Mutex::new(local_engine),
            acoustic_engine: Mutex::new(acoustic_engine),
            baselm: Mutex::new(Some(base_engine)),
            gpu,
            base: base_summary,
            acoustic: acoustic_summary,
            base_format,
            profiler,
            memory,
            tokenizer,
        })
    }

    pub fn status(&self) -> RuntimeStatus<'_> {
        let base_guard = self.baselm.lock().unwrap();
        let base_engine = base_guard.as_ref().expect("BaseLM engine invariant");
        let acoustic_guard = self.acoustic_engine.lock().unwrap();
        let acoustic_state = acoustic_guard.as_ref().map(|engine| AcousticState {
            ready: true,
            residual_position: engine.position(),
            config: engine.config.clone(),
        });
        let local_guard = self.local_engine.lock().unwrap();
        let local_state = local_guard.as_ref().map(|engine| LocalState { ready: true, config: engine.config.clone() });
        let acoustic_ready = acoustic_state.is_some();
        let local_ready = local_state.is_some();
        let audio_guard = self.audiovae_engine.lock().unwrap();
        let audio_state = audio_guard.as_ref().map(|e| AudioVaeState { ready: true, config: e.config.clone(), scratch_bytes: e.scratch_bytes() });
        let audio_ready = audio_state.is_some();
        let mut memory = self.memory.clone();
        let vae_scratch = audio_state.as_ref().map(|x| x.scratch_bytes).unwrap_or(0);
        memory.audiovae_dynamic_scratch_bytes = vae_scratch;
        memory.total_allocated_bytes = memory.total_allocated_bytes.saturating_add(vae_scratch);
        RuntimeStatus {
            engine: "VoxGen",
            implementation_iteration: 7,
            baselm_inference_ready: true,
            fsq_inference_ready: acoustic_ready,
            residual_lm_inference_ready: acoustic_ready,
            locenc_inference_ready: local_ready,
            locdit_estimator_ready: local_ready,
            cfm_solver_ready: local_ready,
            audiovae_encoder_ready: audio_ready,
            audiovae_decoder_ready: audio_ready,
            latent_conditioning_ready: local_ready,
            wav_conditioning_ready: audio_ready && local_ready,
            speech_inference_ready: audio_ready && local_ready && acoustic_ready,
            base_format: self.base_format,
            base_lm: &self.base,
            acoustic: self.acoustic.as_ref(),
            baselm: BaseLmState {
                ready: true,
                position: base_engine.position(),
                config: base_engine.config.clone(),
            },
            residual_fsq: acoustic_state,
            local: local_state,
            audio_vae: audio_state,
            gpu: &self.gpu.info,
            execution_mode: self.gpu.mode.as_str(),
            memory,
            profile: self.profiler.snapshot(),
            gpu_profile: self.gpu.gpu_profile_snapshot(),
            tokenizer: self.tokenizer.info(),
            no_cpu_fallback: true,
        }
    }

    pub fn tokenize(&self, text:&str) -> Result<Vec<u32>> { self.tokenizer.encode(text) }

    pub fn reset_gpu_profile(&self) {
        self.gpu.reset_gpu_profile();
    }

    pub fn reset_baselm(&self) {
        if let Some(engine) = self.baselm.lock().unwrap().as_mut() { engine.reset(); }
    }

    pub fn reset_residual(&self) {
        if let Some(engine) = self.acoustic_engine.lock().unwrap().as_mut() { engine.reset(); }
    }

    pub fn reset_pipeline(&self) {
        self.reset_baselm();
        self.reset_residual();
    }

    pub fn decode_token(&self, token_id: u32, top_k: usize) -> Result<BaseLmStepResult> {
        let mut guard = self.baselm.lock().unwrap();
        let engine = guard.as_mut().context("BaseLM engine unavailable")?;
        engine.decode_token(&self.gpu, &self.base, token_id, top_k)
    }

    pub fn decode_embedding(&self, embedding: &[f32], top_k: usize) -> Result<BaseLmStepResult> {
        let mut guard = self.baselm.lock().unwrap();
        let engine = guard.as_mut().context("BaseLM engine unavailable")?;
        engine.decode_embedding(&self.gpu, &self.base, embedding, top_k)
    }

    pub fn prefill_tokens(&self, tokens: &[u32], top_k_last: usize) -> Result<BaseLmStepResult> {
        let mut guard = self.baselm.lock().unwrap();
        let engine = guard.as_mut().context("BaseLM engine unavailable")?;
        engine.prefill_tokens(&self.gpu, &self.base, tokens, top_k_last)
    }

    pub fn benchmark_baselm(&self, token_id: u32, warmup: u32, iterations: u32) -> Result<BaseLmBenchmark> {
        let mut guard = self.baselm.lock().unwrap();
        let engine = guard.as_mut().context("BaseLM engine unavailable")?;
        engine.benchmark(&self.gpu, &self.base, token_id, warmup, iterations)
    }

    pub fn fsq_only(&self, base_hidden: &[f32]) -> Result<FsqResult> {
        let acoustic = self.acoustic.as_ref().context("--acoustic is required for FSQ")?;
        let mut guard = self.acoustic_engine.lock().unwrap();
        let engine = guard.as_mut().context("ResidualLM/FSQ engine unavailable")?;
        engine.fsq_only(&self.gpu, acoustic, base_hidden)
    }

    pub fn residual_step(&self, base_hidden: &[f32], current_embedding: &[f32]) -> Result<ResidualStepResult> {
        let acoustic = self.acoustic.as_ref().context("--acoustic is required for ResidualLM")?;
        let mut guard = self.acoustic_engine.lock().unwrap();
        let engine = guard.as_mut().context("ResidualLM/FSQ engine unavailable")?;
        engine.step(&self.gpu, acoustic, base_hidden, current_embedding)
    }

    /// Exact generation-loop handoff for the currently implemented stages:
    /// current acoustic embedding -> BaseLM -> FSQ -> fusion_concat_proj -> ResidualLM.
    /// BaseLM's normalized hidden remains on GPU and is consumed directly by FSQ.
    pub fn base_residual_step(&self, current_embedding: &[f32]) -> Result<BaseResidualStepResult> {
        let acoustic_summary = self.acoustic.as_ref().context("--acoustic is required for BaseLM->ResidualLM")?;
        let mut base_guard = self.baselm.lock().unwrap();
        let base_engine = base_guard.as_mut().context("BaseLM engine unavailable")?;
        let baselm = base_engine.decode_embedding_gpu_only(&self.gpu, &self.base, current_embedding)?;
        let base_output = base_engine.output_buffer();
        let mut acoustic_guard = self.acoustic_engine.lock().unwrap();
        let acoustic_engine = acoustic_guard.as_mut().context("ResidualLM/FSQ engine unavailable")?;
        let residual_fsq = acoustic_engine.step_from_gpu_base(
            &self.gpu,
            acoustic_summary,
            base_output,
            current_embedding,
        )?;
        Ok(BaseResidualStepResult { baselm, residual_fsq })
    }

    /// Exact VoxCPM2 text-prefix prefill position:
    /// token embedding -> BaseLM, then unquantized BaseLM hidden + zero acoustic half
    /// -> fusion_concat_proj -> ResidualLM. Both KV caches advance by one position.
    pub fn base_residual_text_token_step(&self, token_id: u32) -> Result<BaseResidualStepResult> {
        let acoustic_summary = self.acoustic.as_ref().context("--acoustic is required for BaseLM->ResidualLM text prefill")?;
        let mut base_guard = self.baselm.lock().unwrap();
        let base_engine = base_guard.as_mut().context("BaseLM engine unavailable")?;
        let baselm = base_engine.decode_token_gpu_only(&self.gpu, &self.base, token_id)?;
        let base_output = base_engine.output_buffer();
        let mut acoustic_guard = self.acoustic_engine.lock().unwrap();
        let acoustic_engine = acoustic_guard.as_mut().context("ResidualLM/FSQ engine unavailable")?;
        let residual_fsq = acoustic_engine.step_text_prefix_from_gpu_base(
            &self.gpu,
            acoustic_summary,
            base_output,
        )?;
        Ok(BaseResidualStepResult { baselm, residual_fsq })
    }

    pub fn locenc_patch(&self, patch: &[f32]) -> Result<LocEncResult> {
        let acoustic = self.acoustic.as_ref().context("--acoustic is required for LocEnc")?;
        let base_guard = self.baselm.lock().unwrap();
        let base = base_guard.as_ref().context("BaseLM engine unavailable")?;
        let mut local_guard = self.local_engine.lock().unwrap();
        let local = local_guard.as_mut().context("LocEnc/LocDiT engine unavailable")?;
        local.encode_patch(&self.gpu, acoustic, &base.config, patch)
    }

    pub fn locdit_cpu_mu(&self, x: &[f32], cond: &[f32], mu: &[f32], t: f32, dt: f32) -> Result<LocDitResult> {
        let acoustic = self.acoustic.as_ref().context("--acoustic is required for LocDiT")?;
        let base_guard = self.baselm.lock().unwrap();
        let base = base_guard.as_ref().context("BaseLM engine unavailable")?;
        let mut local_guard = self.local_engine.lock().unwrap();
        let local = local_guard.as_mut().context("LocEnc/LocDiT engine unavailable")?;
        local.locdit_from_cpu_mu(&self.gpu, acoustic, &base.config, x, cond, mu, t, dt)
    }

    /// Run the official latent-prefix semantics through LocEnc + BaseLM + ResidualLM.
    /// Reference/prompt inputs here are already AudioVAE latent patches; WAV encoding is step 6.
    pub fn prefill_latent_conditioning(&self, text_tokens: &[u32], reference: &[Vec<f32>], prompt: &[Vec<f32>]) -> Result<ConditioningPrefillResult> {
        let acoustic_summary = self.acoustic.as_ref().context("--acoustic is required for latent conditioning")?;
        let plan = conditioning::build_plan(text_tokens, reference, prompt)?;
        self.reset_pipeline();
        if self.gpu.mode == ExecutionMode::Xtx7900 {
            // Pass 3: record contiguous text-prefix positions from BaseLM and ResidualLM
            // into one shared command buffer. The autoregressive dependency is preserved by
            // barriers between positions, but the CPU only submits/waits once per batch.
            // Live streaming uses a wider submission batch to cut CPU/driver wait
            // boundaries on the TTFA-critical text prefill. Offline timestamp profiling
            // retains 16 positions so query/command-buffer accounting stays bounded.
            let xtx_prefill_batch_positions: usize = if self.gpu.gpu_profiling_enabled() { 16 } else { 32 };
            let mut index = 0usize;
            while index < plan.positions.len() {
                match &plan.positions[index] {
                    PrefixPosition::Text(_) => {
                        let run_start = index;
                        while index < plan.positions.len() && matches!(&plan.positions[index], PrefixPosition::Text(_)) { index += 1; }
                        let run_end = index;
                        let mut batch_start = run_start;
                        while batch_start < run_end {
                            let batch_end = (batch_start + xtx_prefill_batch_positions).min(run_end);
                            let mut base_guard = self.baselm.lock().unwrap();
                            let mut residual_guard = self.acoustic_engine.lock().unwrap();
                            let base = base_guard.as_mut().context("BaseLM unavailable")?;
                            let residual = residual_guard.as_mut().context("ResidualLM unavailable")?;
                            let cmd = base.prefill_command_buffer();
                            self.gpu.begin_one_time(cmd)?;
                            let batch_span = self.gpu.gpu_profile_begin(cmd, "prefill.cross_engine_batch");
                            for position in &plan.positions[batch_start..batch_end] {
                                let PrefixPosition::Text(token) = position else { unreachable!() };
                                let _ = base.record_token_gpu_only_in(&self.gpu, &self.base, *token, cmd)?;
                                self.gpu.compute_barrier(cmd);
                                let _ = residual.record_text_prefix_from_gpu_base_in(
                                    &self.gpu, acoustic_summary, base.output_buffer(), cmd,
                                )?;
                                self.gpu.compute_barrier(cmd);
                            }
                            self.gpu.gpu_profile_end(cmd, batch_span);
                            self.gpu.submit_and_wait(cmd)?;
                            batch_start = batch_end;
                        }
                    }
                    PrefixPosition::Audio(patch) => {
                        // Audio-prefix positions still use the proven LocEnc -> BaseLM ->
                        // ResidualLM path. Text prefill is normally the dominant startup run.
                        let mut base_guard = self.baselm.lock().unwrap();
                        let mut residual_guard = self.acoustic_engine.lock().unwrap();
                        let mut local_guard = self.local_engine.lock().unwrap();
                        let base = base_guard.as_mut().context("BaseLM unavailable")?;
                        let residual = residual_guard.as_mut().context("ResidualLM unavailable")?;
                        let local = local_guard.as_mut().context("LocEnc unavailable")?;
                        local.encode_patch_gpu_only(&self.gpu, acoustic_summary, &base.config, patch)?;
                        let _ = base.decode_embedding_from_gpu_only(&self.gpu, &self.base, local.locenc_output_buffer())?;
                        let _ = residual.step_from_gpu_base_and_embedding_gpu_only(&self.gpu, acoustic_summary, base.output_buffer(), local.locenc_output_buffer())?;
                        index += 1;
                    }
                }
            }
        } else {
            // Normal mode remains the correctness/reference implementation.
            for position in &plan.positions {
                match position {
                    PrefixPosition::Text(token) => {
                        let mut base_guard=self.baselm.lock().unwrap();let mut residual_guard=self.acoustic_engine.lock().unwrap();
                        let base=base_guard.as_mut().context("BaseLM unavailable")?;let residual=residual_guard.as_mut().context("ResidualLM unavailable")?;
                        let _=base.decode_token_gpu_only(&self.gpu,&self.base,*token)?;let _=residual.step_text_prefix_from_gpu_base_gpu_only(&self.gpu,acoustic_summary,base.output_buffer())?;
                    }
                    PrefixPosition::Audio(patch) => {
                        // Lock order matches status(): BaseLM -> ResidualLM -> Local.
                        let mut base_guard = self.baselm.lock().unwrap();
                        let mut residual_guard = self.acoustic_engine.lock().unwrap();
                        let mut local_guard = self.local_engine.lock().unwrap();
                        let base = base_guard.as_mut().context("BaseLM unavailable")?;
                        let residual = residual_guard.as_mut().context("ResidualLM unavailable")?;
                        let local = local_guard.as_mut().context("LocEnc unavailable")?;
                        local.encode_patch_gpu_only(&self.gpu, acoustic_summary, &base.config, patch)?;
                        let _ = base.decode_embedding_from_gpu_only(&self.gpu, &self.base, local.locenc_output_buffer())?;
                        let _ = residual.step_from_gpu_base_and_embedding_gpu_only(&self.gpu, acoustic_summary, base.output_buffer(), local.locenc_output_buffer())?;
                    }
                }
            }
        }
        let (prefix_condition_checksum, prefix_condition_l2) = stats(&plan.prefix_condition);
        let status = self.status();
        Ok(ConditioningPrefillResult {
            plan: plan.summary,
            baselm_position: status.baselm.position,
            residual_position: status.residual_fsq.map(|x| x.residual_position).unwrap_or(0),
            prefix_condition_checksum, prefix_condition_l2,
        })
    }

    /// LocDiT estimator using the current BaseLM/ResidualLM hidden states for the two mu tokens.
    /// This is one estimator call only; CFG duplication and Euler integration are step 5.
    pub fn locdit_from_current_hiddens(&self, x: &[f32], cond: &[f32], t: f32, dt: f32) -> Result<LocDitResult> {
        let acoustic = self.acoustic.as_ref().context("--acoustic is required for LocDiT")?;
        let base_guard = self.baselm.lock().unwrap();
        let _residual_guard = self.acoustic_engine.lock().unwrap();
        let base = base_guard.as_ref().context("BaseLM engine unavailable")?;
        let mut local_guard = self.local_engine.lock().unwrap();
        let local = local_guard.as_mut().context("LocEnc/LocDiT engine unavailable")?;
        local.locdit_from_model_hiddens(&self.gpu, acoustic, &base.config, x, cond, t, dt)
    }

    /// UnifiedCFM over an explicit 2048-float mu. The conditional and unconditional
    /// LocDiT passes plus CFG-Zero* reduction and Euler update remain on Vulkan.
    pub fn cfm_cpu_mu(&self, cond:&[f32], mu:&[f32], options:&CfmOptions, initial_x:Option<&[f32]>) -> Result<CfmResult> {
        let acoustic=self.acoustic.as_ref().context("--acoustic is required for CFM")?;
        let base_guard=self.baselm.lock().unwrap();
        let base=base_guard.as_ref().context("BaseLM engine unavailable")?;
        let mut local_guard=self.local_engine.lock().unwrap();
        let local=local_guard.as_mut().context("LocEnc/LocDiT/CFM engine unavailable")?;
        local.cfm_from_cpu_mu(&self.gpu,acoustic,&base.config,cond,mu,options,initial_x)
    }

    /// UnifiedCFM using the current BaseLM and ResidualLM hidden states as LocDiT mu.
    pub fn cfm_from_current_hiddens(&self, cond:&[f32], options:&CfmOptions, initial_x:Option<&[f32]>) -> Result<CfmResult> {
        let acoustic=self.acoustic.as_ref().context("--acoustic is required for CFM")?;
        let base_guard=self.baselm.lock().unwrap();
        let _residual_guard=self.acoustic_engine.lock().unwrap();
        let base=base_guard.as_ref().context("BaseLM engine unavailable")?;
        let mut local_guard=self.local_engine.lock().unwrap();
        let local=local_guard.as_mut().context("LocEnc/LocDiT/CFM engine unavailable")?;
        local.cfm_from_model_hiddens(&self.gpu,acoustic,&base.config,cond,options,initial_x)
    }

    /// Step-5 boundary: build the latent reference/prompt prefix exactly as VoxCPM2,
    /// then generate the next 4x64 acoustic latent patch with UnifiedCFM.
    pub fn prefill_latent_conditioning_and_cfm(&self,text_tokens:&[u32],reference:&[Vec<f32>],prompt:&[Vec<f32>],options:&CfmOptions,initial_x:Option<&[f32]>) -> Result<ConditioningCfmResult> {
        let plan=conditioning::build_plan(text_tokens,reference,prompt)?;
        let prefill=self.prefill_latent_conditioning(text_tokens,reference,prompt)?;
        let cfm=self.cfm_from_current_hiddens(&plan.prefix_condition,options,initial_x)?;
        Ok(ConditioningCfmResult{prefill,cfm})
    }

    /// AudioVAE V2 waveform encoder. WAV decoding/downmix/resampling are host preprocessing;
    /// all learned AudioVAE tensor operations execute on Vulkan. The returned latent layout is
    /// frame-major [T,64], directly splittable into VoxCPM2 4x64 patches.
    pub fn audiovae_encode_wav(&self, path:&Path, pad_side:AudioPadSide) -> Result<(AudioEncodeStats, Vec<f32>)> {
        let acoustic=self.acoustic.as_ref().context("--acoustic is required for AudioVAE")?;
        let acoustic_guard=self.acoustic_engine.lock().unwrap();
        let model=acoustic_guard.as_ref().context("acoustic model-data buffer unavailable")?.model_data_buffer();
        let mut vae_guard=self.audiovae_engine.lock().unwrap();
        let vae=vae_guard.as_mut().context("AudioVAE engine unavailable")?;
        vae.encode_wav(&self.gpu,model,acoustic,path,pad_side)
    }

    pub fn audiovae_encode_pcm16k(&self, samples:&[f32], pad_side:AudioPadSide) -> Result<(AudioEncodeStats, Vec<f32>)> {
        let acoustic=self.acoustic.as_ref().context("--acoustic is required for AudioVAE")?;
        let acoustic_guard=self.acoustic_engine.lock().unwrap();
        let model=acoustic_guard.as_ref().context("acoustic model-data buffer unavailable")?.model_data_buffer();
        let mut vae_guard=self.audiovae_engine.lock().unwrap();
        let vae=vae_guard.as_mut().context("AudioVAE engine unavailable")?;
        vae.encode_pcm16k(&self.gpu,model,acoustic,samples,pad_side)
    }

    pub fn audiovae_decode_latents(&self, latents:&[f32]) -> Result<(AudioDecodeStats, Vec<f32>)> {
        let acoustic=self.acoustic.as_ref().context("--acoustic is required for AudioVAE")?;
        let acoustic_guard=self.acoustic_engine.lock().unwrap();
        let model=acoustic_guard.as_ref().context("acoustic model-data buffer unavailable")?.model_data_buffer();
        let mut vae_guard=self.audiovae_engine.lock().unwrap();
        let vae=vae_guard.as_mut().context("AudioVAE engine unavailable")?;
        vae.decode_latents(&self.gpu,model,acoustic,latents)
    }

    pub fn audiovae_encode_wav_patches(&self,path:&Path,pad_side:AudioPadSide)->Result<(AudioEncodeStats,Vec<Vec<f32>>)> {
        let (stats,latents)=self.audiovae_encode_wav(path,pad_side)?;
        if latents.len()%256!=0{bail!("AudioVAE encoded latent count {} is not divisible by a 4x64 patch",latents.len());}
        Ok((stats,latents.chunks_exact(256).map(|x|x.to_vec()).collect()))
    }

    /// Cache the expensive AudioVAE conditioning encode by canonical file identity.
    /// A four-entry LRU-like window is enough for the demo's neutral + emotional
    /// references without making model lifetime depend on an unbounded file cache.
    pub fn audiovae_encode_wav_patches_cached(&self,path:&Path,pad_side:AudioPadSide)->Result<(AudioEncodeStats,Vec<Vec<f32>>)> {
        let key=audio_conditioning_cache_key(path,pad_side)?;
        if let Ok(cache)=self.conditioning_audio_cache.lock() {
            if let Some(hit)=cache.iter().rev().find(|entry|entry.key==key) {
                let mut stats=hit.stats.clone();
                // encode_ms is request-side work. A cache hit performs no encode.
                stats.encode_ms=0.0;
                return Ok((stats,hit.patches.clone()));
            }
        }
        let (stats,patches)=self.audiovae_encode_wav_patches(&key.path,pad_side)?;
        if let Ok(mut cache)=self.conditioning_audio_cache.lock() {
            cache.retain(|entry| !(entry.key.path==key.path && entry.key.pad_side==key.pad_side));
            cache.push(AudioConditioningCacheEntry{key,stats:stats.clone(),patches:patches.clone()});
            if cache.len()>4 { let excess=cache.len()-4; cache.drain(0..excess); }
        }
        Ok((stats,patches))
    }

    /// Pre-encode a local reference while the user is still preparing text.
    /// The HTTP/demo layer calls this after model load and reference selection.
    pub fn warm_reference_wav(&self,path:&Path)->Result<(AudioEncodeStats,usize)> {
        let (stats,patches)=self.audiovae_encode_wav_patches_cached(path,AudioPadSide::Right)?;
        Ok((stats,patches.len()))
    }

    /// Step-6 voice-conditioning bridge: encode reference/prompt WAVs with the exact VoxCPM2
    /// right/left patch-alignment semantics, then run the existing LocEnc/LM conditioning prefill.
    pub fn prefill_wav_conditioning(&self,text_tokens:&[u32],reference_wav:Option<&Path>,prompt_wav:Option<&Path>)->Result<(ConditioningPrefillResult,Option<AudioEncodeStats>,Option<AudioEncodeStats>)> {
        let (ref_stats,reference)=if let Some(p)=reference_wav{let(st,v)=self.audiovae_encode_wav_patches_cached(p,AudioPadSide::Right)?;(Some(st),v)}else{(None,Vec::new())};
        let (prompt_stats,prompt)=if let Some(p)=prompt_wav{let(st,v)=self.audiovae_encode_wav_patches_cached(p,AudioPadSide::Left)?;(Some(st),v)}else{(None,Vec::new())};
        let prefill=self.prefill_latent_conditioning(text_tokens,&reference,&prompt)?;
        Ok((prefill,ref_stats,prompt_stats))
    }

    pub fn predict_stop(&self)->Result<StopPrediction>{
        let acoustic=self.acoustic.as_ref().context("--acoustic is required for stop prediction")?;
        let mut residual_guard=self.acoustic_engine.lock().unwrap(); let residual=residual_guard.as_mut().context("stop predictor unavailable")?;
        residual.predict_stop_from_current_lm(&self.gpu,acoustic)
    }

    /// Feed one newly generated 4x64 patch back into LocEnc -> BaseLM -> FSQ/fusion -> ResidualLM.
    pub fn advance_generated_patch(&self,patch:&[f32])->Result<BaseResidualStepResult>{
        let acoustic=self.acoustic.as_ref().context("--acoustic is required for generation")?;
        let mut base_guard=self.baselm.lock().unwrap(); let mut residual_guard=self.acoustic_engine.lock().unwrap(); let mut local_guard=self.local_engine.lock().unwrap();
        let base=base_guard.as_mut().context("BaseLM unavailable")?; let residual=residual_guard.as_mut().context("ResidualLM unavailable")?; let local=local_guard.as_mut().context("LocEnc unavailable")?;
        local.encode_patch_gpu_only(&self.gpu,acoustic,&base.config,patch)?;
        let baselm=base.decode_embedding_from_gpu_only(&self.gpu,&self.base,local.locenc_output_buffer())?;
        let residual_fsq=residual.step_from_gpu_base_and_embedding(&self.gpu,acoustic,base.output_buffer(),local.locenc_output_buffer())?;
        Ok(BaseResidualStepResult{baselm,residual_fsq})
    }

    fn advance_generated_patch_gpu_only(&self,patch:&[f32])->Result<()> {
        let acoustic=self.acoustic.as_ref().context("--acoustic is required for generation")?;let mut base_guard=self.baselm.lock().unwrap();let mut residual_guard=self.acoustic_engine.lock().unwrap();let mut local_guard=self.local_engine.lock().unwrap();let base=base_guard.as_mut().context("BaseLM unavailable")?;let residual=residual_guard.as_mut().context("ResidualLM unavailable")?;let local=local_guard.as_mut().context("LocEnc unavailable")?;local.encode_patch_gpu_only(&self.gpu,acoustic,&base.config,patch)?;let _=base.decode_embedding_from_gpu_only(&self.gpu,&self.base,local.locenc_output_buffer())?;let _=residual.step_from_gpu_base_and_embedding_gpu_only(&self.gpu,acoustic,base.output_buffer(),local.locenc_output_buffer())?;Ok(())
    }

    /// Complete VoxCPM2 autoregressive TTS. Non-streaming synthesis decodes the full
    /// generated latent sequence. The streaming endpoint retains the compatibility rolling
    /// decoder until the fully stateful AudioVAE streaming path is enabled.
    pub fn synthesize<F>(&self,text:&str,control:Option<&str>,prompt_text:Option<&str>,reference_wav:Option<&Path>,prompt_wav:Option<&Path>,options:&TtsOptions,on_pcm:Option<F>)->Result<TtsResult>
    where F: FnMut(&[f32],u32)->Result<()> {
        self.synthesize_cancelable(text, control, prompt_text, reference_wav, prompt_wav, options, None, on_pcm)
    }

    /// Same synthesis pipeline as `synthesize`, with cooperative request cancellation.
    ///
    /// Cancellation is intentionally observed only between completed GPU operations.
    /// In particular, an in-flight CFM/AudioVAE dispatch is allowed to finish, then the
    /// next autoregressive acoustic patch is not started. This avoids trying to tear down
    /// Vulkan work mid-submit while still bounding stop latency to roughly one patch.
    pub fn synthesize_cancelable<F>(
        &self,
        text:&str,
        control:Option<&str>,
        prompt_text:Option<&str>,
        reference_wav:Option<&Path>,
        prompt_wav:Option<&Path>,
        options:&TtsOptions,
        cancel:Option<&AtomicBool>,
        mut on_pcm:Option<F>,
    )->Result<TtsResult>
    where F: FnMut(&[f32],u32)->Result<()> {
        let cancelled = || cancel.is_some_and(|flag| flag.load(Ordering::Acquire));
        options.validate()?; self.assert_speech_available()?;
        if cancelled() { bail!("speech synthesis cancelled"); }
        // End-to-end synthesis timing starts before tokenization/reference encoding
        // and prefill so TTFA/RTF diagnostics include all request-side inference work.
        let started=Instant::now();
        if prompt_wav.is_some() && prompt_text.map(str::trim).unwrap_or("").is_empty(){bail!("--prompt-wav requires --prompt-text with the exact transcript for continuation conditioning");}
        let control=control.map(str::trim).filter(|x|!x.is_empty());
        if control.is_some() && (prompt_wav.is_some() || prompt_text.map(str::trim).is_some_and(|x|!x.is_empty())) {
            bail!("VoxCPM2 style control cannot be combined with prompt-audio + prompt-text continuation/ultimate cloning");
        }
        // VoxCPM2's native style-control contract is textual: `(instruction)` is
        // prepended to the target text before tokenization.  Preserve the user's
        // punctuation and wording verbatim after the control prefix.
        let controlled_text=if let Some(c)=control{format!("({c}){text}")}else{text.to_owned()};
        let token_text=if prompt_wav.is_some(){format!("{}{}",prompt_text.unwrap_or(""),controlled_text)}else{controlled_text};
        let text_tokens=self.tokenizer.encode(&token_text)?;
        if cancelled() { bail!("speech synthesis cancelled"); }
        let (ref_stats,reference)=if let Some(p)=reference_wav{let(st,v)=self.audiovae_encode_wav_patches_cached(p,AudioPadSide::Right)?;(Some(st),v)}else{(None,Vec::new())};
        if cancelled() { bail!("speech synthesis cancelled"); }
        let (prompt_stats,prompt)=if let Some(p)=prompt_wav{let(st,v)=self.audiovae_encode_wav_patches_cached(p,AudioPadSide::Left)?;(Some(st),v)}else{(None,Vec::new())};
        let _=(ref_stats,prompt_stats); // retained by diagnostics in the individual VAE APIs.
        // Upstream streaming/full decode seeds AudioVAE with the last prompt patches so the
        // first generated waveform chunk has causal decoder context, then trims that context.
        let prefix_keep=options.streaming_prefix_len.saturating_sub(1).min(prompt.len());
        let decode_prefix=prompt[prompt.len().saturating_sub(prefix_keep)..].to_vec();
        let plan=conditioning::build_plan(&text_tokens,&reference,&prompt)?;
        let conditioning_summary=plan.summary.clone();
        let mut condition=plan.prefix_condition.clone();
        let _prefill=self.prefill_latent_conditioning(&text_tokens,&reference,&prompt)?;
        if cancelled() { bail!("speech synthesis cancelled"); }
        let mut first_pcm_ms=None; let mut generated:Vec<Vec<f32>>=Vec::new(); let mut streamed_samples=Vec::new(); let mut trace=Vec::new(); let mut stopped=false;
        for step in 0..options.max_steps {
            if cancelled() { bail!("speech synthesis cancelled"); }
            let mut cfm=options.cfm.clone(); cfm.seed=cfm.seed.wrapping_add((step as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let out=self.cfm_from_current_hiddens(&condition,&cfm,None)?; let patch=out.output_patch.clone();
            // Safe cancellation boundary: the current GPU patch has completed, but
            // nothing from it has to be published or advanced into the LM state.
            if cancelled() { bail!("speech synthesis cancelled"); }
            generated.push(patch.clone());
            let mut emitted=0usize;
            if on_pcm.is_some(){
                let mut context:Vec<&Vec<f32>>=decode_prefix.iter().chain(generated.iter()).collect(); if context.len()>options.streaming_prefix_len{context.drain(..context.len()-options.streaming_prefix_len);}
                let mut rolling=Vec::with_capacity(context.len()*256);for p in context{rolling.extend_from_slice(p);}
                let (_ds,pcm)=self.audiovae_decode_latents(&rolling)?; let take=7680.min(pcm.len()); let chunk=&pcm[pcm.len()-take..]; emitted=chunk.len();
                // The stop head decides whether another patch is needed; it cannot
                // retract the patch we just generated. Publish current PCM first so
                // socket/client playback can begin while stop prediction follows.
                if first_pcm_ms.is_none(){first_pcm_ms=Some(started.elapsed().as_secs_f64()*1000.0);} if let Some(cb)=on_pcm.as_mut(){cb(chunk,48000)?;} streamed_samples.extend_from_slice(chunk);
            }
            if cancelled() { bail!("speech synthesis cancelled"); }
            let stop=self.predict_stop()?;
            trace.push(TtsStepTrace{step,cfm_ms:out.elapsed_ms,stop:stop.clone(),emitted_samples:emitted});
            // Upstream checks the stop head on the current LM state after generating this patch.
            if step > options.min_steps && stop.stop { stopped=true; break; }
            if step+1>=options.max_steps { break; }
            self.advance_generated_patch_gpu_only(&patch)?; condition=patch;
        }
        let samples=if on_pcm.is_some(){streamed_samples}else{let mut lat=Vec::with_capacity((decode_prefix.len()+generated.len())*256);for p in &decode_prefix{lat.extend_from_slice(p);}for p in &generated{lat.extend_from_slice(p);}let(_ds,pcm)=self.audiovae_decode_latents(&lat)?;let trim=(decode_prefix.len()*7680).min(pcm.len());let pcm=pcm[trim..].to_vec();if first_pcm_ms.is_none(){first_pcm_ms=Some(started.elapsed().as_secs_f64()*1000.0);}pcm};
        if cancelled() { bail!("speech synthesis cancelled"); }
        let elapsed_ms=started.elapsed().as_secs_f64()*1000.0; let audio_seconds=samples.len() as f64/48000.0; let rtf=if audio_seconds>0.0{elapsed_ms/1000.0/audio_seconds}else{f64::INFINITY};
        Ok(TtsResult{text:text.to_owned(),control:control.map(str::to_owned),token_count:text_tokens.len(),generated_patches:generated.len(),stopped_by_predictor:stopped,sample_rate:48000,sample_count:samples.len(),audio_seconds,elapsed_ms,rtf,first_pcm_ms,conditioning:conditioning_summary,steps:trace,samples})
    }

    pub fn benchmark_residual(
        &self,
        base_hidden: &[f32],
        current_embedding: &[f32],
        warmup: u32,
        iterations: u32,
    ) -> Result<ResidualBenchmark> {
        let acoustic = self.acoustic.as_ref().context("--acoustic is required for ResidualLM benchmark")?;
        let mut guard = self.acoustic_engine.lock().unwrap();
        let engine = guard.as_mut().context("ResidualLM/FSQ engine unavailable")?;
        engine.benchmark(&self.gpu, acoustic, base_hidden, current_embedding, warmup, iterations)
    }

    pub fn assert_speech_available(&self) -> Result<()> {
        let st=self.status();
        if !(st.baselm_inference_ready&&st.residual_lm_inference_ready&&st.locenc_inference_ready&&st.cfm_solver_ready&&st.audiovae_encoder_ready&&st.audiovae_decoder_ready){
            bail!("VoxGen speech requires BaseLM + acoustic GGUF with ResidualLM/FSQ, LocEnc/LocDiT/CFM, and AudioVAE loaded");
        }
        Ok(())
    }
}

fn stats(v: &[f32]) -> (f64, f64) { let sum=v.iter().map(|&x|x as f64).sum(); let l2=v.iter().map(|&x|(x as f64)*(x as f64)).sum::<f64>().sqrt(); (sum,l2) }
