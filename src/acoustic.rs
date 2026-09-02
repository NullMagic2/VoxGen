use crate::{
    baselm::BaseLmConfig,
    gguf::{GgmlType, GgufSummary, TensorInfo},
    vulkan::{ComputePipeline, GpuBuffer, VulkanContext},
};
use anyhow::{bail, Context, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use serde::Serialize;
use std::{fs::File, time::Instant};

const RMSNORM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm.spv"));
const RMSNORM_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm_xtx7900.spv"));
const MATVEC_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matvec.spv"));
const MATVEC_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matvec_xtx7900.spv"));
const QKV_MATVEC_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qkv_matvec.spv"));
const QKV_MATVEC_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qkv_matvec_xtx7900.spv"));
const STORE_KV_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/store_kv.spv"));
const ATTN_SCORES_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_scores.spv"));
const ATTN_SCORES_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_scores_xtx7900.spv"));
const SOFTMAX_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/softmax.spv"));
const SOFTMAX_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/softmax_xtx7900.spv"));
const ATTN_VALUES_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attn_values.spv"));
const SWIGLU_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matvec_swiglu.spv"));
const SWIGLU_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matvec_swiglu_xtx7900.spv"));
const RESIDUAL_RMSNORM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/residual_rmsnorm.spv"));
const RESIDUAL_RMSNORM_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/residual_rmsnorm_xtx7900.spv"));
const LINEAR_BIAS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_bias.spv"));
const LINEAR_BIAS_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/linear_bias_xtx7900.spv"));
const FSQ_QUANTIZE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fsq_quantize.spv"));
const FUSION_LINEAR_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fusion_linear.spv"));
const FUSION_LINEAR_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fusion_linear_xtx7900.spv"));
const SILU_INPLACE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/silu_inplace.spv"));

const VOXCPM2_MAX_GENERATION_CONTEXT: u32 = 8192;

#[derive(Debug, Clone, Serialize)]
pub struct AcousticConfig {
    pub architecture: String,
    pub model_version: f32,
    pub residual_block_count: u32,
    pub embedding_length: u32,
    pub feed_forward_length: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub head_dim: u32,
    pub kv_dim: u32,
    pub no_rope: bool,
    pub active_context_length: u32,
    pub rms_epsilon: f32,
    pub residual_scale: f32,
    pub fsq_latent_dim: u32,
    pub fsq_scale: f32,
    pub fusion_input_dim: u32,
}

impl AcousticConfig {
    pub fn from_gguf(
        summary: &GgufSummary,
        base: &BaseLmConfig,
        active_context_length: u32,
    ) -> Result<Self> {
        let architecture = summary.metadata_str("general.architecture")?.to_owned();
        if architecture != "voxcpm-acoustic" {
            bail!("VoxGen expects general.architecture=voxcpm-acoustic for the acoustic GGUF, got {architecture:?}");
        }
        let model_version = summary.metadata_f32("voxcpm.model_version")?;
        if (model_version - 2.0).abs() > 0.01 {
            bail!("VoxGen iteration 7 is specialized for VoxCPM2 acoustic v2.0, got model_version={model_version}");
        }
        if active_context_length == 0 || active_context_length > VOXCPM2_MAX_GENERATION_CONTEXT {
            bail!(
                "ResidualLM --max-context must be in 1..={VOXCPM2_MAX_GENERATION_CONTEXT}; got {active_context_length}. VoxCPM2 config.max_length is 8192."
            );
        }

        let residual_block_count = summary.metadata_u32("voxcpm.residual_lm.n_layer")?;
        let embedding_length = summary.metadata_u32("voxcpm.residual_lm.n_embd")?;
        let no_rope = summary.metadata_bool("voxcpm.residual_lm.no_rope")?;
        let fsq_latent_dim = summary.metadata_u32("voxcpm.fsq.latent_dim")?;
        let fsq_scale = summary.metadata_f32("voxcpm.fsq.scale")?;

        if residual_block_count != 8
            || embedding_length != 2048
            || fsq_latent_dim != 512
            || !no_rope
        {
            bail!(
                "unsupported VoxCPM2 acoustic contract: residual_layers={residual_block_count}, embd={embedding_length}, fsq_latent={fsq_latent_dim}, no_rope={no_rope}; expected 8/2048/512/true"
            );
        }
        if base.embedding_length != embedding_length
            || base.feed_forward_length != 6144
            || base.head_count != 16
            || base.head_count_kv != 2
            || base.head_dim != 128
        {
            bail!("BaseLM geometry is incompatible with VoxCPM2 ResidualLM");
        }
        if !fsq_scale.is_finite() || fsq_scale <= 0.0 {
            bail!("invalid FSQ scale {fsq_scale}");
        }

        let cfg = Self {
            architecture,
            model_version,
            residual_block_count,
            embedding_length,
            feed_forward_length: base.feed_forward_length,
            head_count: base.head_count,
            head_count_kv: base.head_count_kv,
            head_dim: base.head_dim,
            kv_dim: base.kv_dim,
            no_rope,
            active_context_length,
            rms_epsilon: base.rms_epsilon,
            residual_scale: effective_scale(base.residual_scale),
            fsq_latent_dim,
            fsq_scale,
            fusion_input_dim: embedding_length * 2,
        };
        cfg.validate_tensors(summary)?;
        Ok(cfg)
    }

    fn validate_tensors(&self, s: &GgufSummary) -> Result<()> {
        // FSQ: 2048 -> 512 -> quantize -> 2048.
        require_matrix(s.tensor("fsq.in_proj.weight")?, self.embedding_length, self.fsq_latent_dim, true)?;
        require_vector(s.tensor("fsq.in_proj.bias")?, self.fsq_latent_dim, false)?;
        require_matrix(s.tensor("fsq.out_proj.weight")?, self.fsq_latent_dim, self.embedding_length, true)?;
        require_vector(s.tensor("fsq.out_proj.bias")?, self.embedding_length, false)?;

        // VoxCPM2 residual input = Linear(cat(FSQ(base_hidden), current_acoustic_embed)).
        require_matrix(
            s.tensor("projections.res_fusion_proj.weight")?,
            self.fusion_input_dim,
            self.embedding_length,
            true,
        )?;
        require_vector(s.tensor("projections.res_fusion_proj.bias")?, self.embedding_length, false)?;

        let output_norm = s.tensor("residual_lm.output_norm.weight")?;
        require_vector(output_norm, self.embedding_length, true)?;
        if output_norm.ggml_type != GgmlType::F32 {
            bail!("{} must be F32, got {}", output_norm.name, output_norm.ggml_type.name());
        }

        for layer in 0..self.residual_block_count {
            let p = |suffix: &str| format!("residual_lm.blk.{layer}.{suffix}");
            for norm_name in [p("attn_norm.weight"), p("ffn_norm.weight")] {
                let t = s.tensor(&norm_name)?;
                require_vector(t, self.embedding_length, true)?;
                if t.ggml_type != GgmlType::F32 {
                    bail!("{} must be F32, got {}", t.name, t.ggml_type.name());
                }
            }
            for (suffix, cols, rows) in [
                ("attn_q.weight", self.embedding_length, self.embedding_length),
                ("attn_k.weight", self.embedding_length, self.kv_dim),
                ("attn_v.weight", self.embedding_length, self.kv_dim),
                ("attn_output.weight", self.embedding_length, self.embedding_length),
                ("ffn_gate.weight", self.embedding_length, self.feed_forward_length),
                ("ffn_up.weight", self.embedding_length, self.feed_forward_length),
                ("ffn_down.weight", self.feed_forward_length, self.embedding_length),
            ] {
                require_matrix(s.tensor(&p(suffix))?, cols, rows, true)?;
            }
        }
        require_matrix(s.tensor("stop_predictor.linear1.weight")?, self.embedding_length, self.embedding_length, true)?;
        require_vector(s.tensor("stop_predictor.linear1.bias")?, self.embedding_length, true)?;
        require_matrix(s.tensor("stop_predictor.linear2.weight")?, self.embedding_length, 2, true)?;
        Ok(())
    }

    pub fn kv_cache_bytes(&self) -> u64 {
        self.residual_block_count as u64
            * self.active_context_length as u64
            * self.kv_dim as u64
            * 2
            * 2
    }

    pub fn per_cache_bytes(&self) -> u64 {
        self.kv_cache_bytes() / 2
    }
}

fn require_matrix(t: &TensorInfo, cols: u32, rows: u32, require_f16: bool) -> Result<()> {
    let expected = [cols as u64, rows as u64];
    if t.dims != expected {
        bail!("tensor {} has dimensions {:?}, expected {:?}", t.name, t.dims, expected);
    }
    if require_f16 && t.ggml_type != GgmlType::F16 {
        bail!(
            "tensor {} is {}, expected F16 from VoxCPM2-Acoustic-F16.gguf; acoustic CPU fallback/implicit conversion is disabled",
            t.name,
            t.ggml_type.name()
        );
    }
    if !matches!(t.ggml_type, GgmlType::F16 | GgmlType::F32) {
        bail!("tensor {} uses unsupported acoustic type {}", t.name, t.ggml_type.name());
    }
    Ok(())
}

fn require_vector(t: &TensorInfo, n: u32, allow_f32: bool) -> Result<()> {
    if t.dims != [n as u64] {
        bail!("tensor {} has dimensions {:?}, expected [{}]", t.name, t.dims, n);
    }
    let ok = t.ggml_type == GgmlType::F16 || (allow_f32 && t.ggml_type == GgmlType::F32);
    if !ok {
        bail!("tensor {} uses unsupported vector type {}", t.name, t.ggml_type.name());
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RmsPush {
    weight_offset: u32,
    n: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MatvecPush {
    weight_offset: u32,
    rows: u32,
    cols: u32,
    dtype: u32,
    row_base: u32,
    alpha: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearBiasPush {
    weight_offset: u32,
    bias_offset: u32,
    rows: u32,
    cols: u32,
    weight_dtype: u32,
    bias_dtype: u32,
    row_base: u32,
    alpha: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FsqPush {
    n: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FusionPush {
    weight_offset: u32,
    bias_offset: u32,
    rows: u32,
    left_cols: u32,
    right_cols: u32,
    weight_dtype: u32,
    bias_dtype: u32,
    row_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StoreKvPush {
    layer: u32,
    position: u32,
    context: u32,
    kv_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttentionPush {
    layer: u32,
    context: u32,
    position: u32,
    head_dim: u32,
    q_per_kv: u32,
    kv_heads: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SoftmaxPush {
    context: u32,
    position: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NPush {
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QkvPush {
    q_weight_offset: u32,
    k_weight_offset: u32,
    v_weight_offset: u32,
    q_rows: u32,
    kv_rows: u32,
    cols: u32,
    q_dtype: u32,
    k_dtype: u32,
    v_dtype: u32,
    row_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SwigluPush {
    gate_weight_offset: u32,
    up_weight_offset: u32,
    rows: u32,
    cols: u32,
    dtype: u32,
    row_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResidualRmsPush {
    weight_offset: u32,
    n: u32,
    eps: f32,
    scale: f32,
}

struct Pipelines {
    fsq_in: ComputePipeline,
    fsq_quantize: ComputePipeline,
    fsq_out: ComputePipeline,
    fusion: ComputePipeline,
    rms_attn: ComputePipeline,
    residual_rms: ComputePipeline,
    qkv: ComputePipeline,
    attn_out: ComputePipeline,
    swiglu: ComputePipeline,
    down: ComputePipeline,
    store_kv: ComputePipeline,
    attn_scores: ComputePipeline,
    softmax: ComputePipeline,
    attn_values: ComputePipeline,
    stop_linear1: ComputePipeline,
    stop_silu: ComputePipeline,
    stop_linear2: ComputePipeline,
}

#[derive(Debug, Clone, Serialize)]
pub struct FsqResult {
    pub elapsed_ms: f64,
    pub output_checksum: f64,
    pub output_l2: f64,
    pub latent_checksum: f64,
    pub latent_l2: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResidualStepResult {
    pub position: u32,
    pub elapsed_ms: f64,
    pub fsq_applied: bool,
    pub fsq_checksum: Option<f64>,
    pub fsq_l2: Option<f64>,
    pub fused_input_checksum: f64,
    pub fused_input_l2: f64,
    pub residual_hidden_checksum: f64,
    pub residual_hidden_l2: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResidualGpuStep { pub position:u32, pub elapsed_ms:f64, pub fsq_applied:bool }

#[derive(Debug, Clone, Serialize)]
pub struct ResidualBenchmark {
    pub iterations: u32,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub steps_per_second: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct StopPrediction {
    pub stop: bool,
    pub continue_logit: f32,
    pub stop_logit: f32,
    pub elapsed_ms: f64,
}

pub struct ResidualFsqEngine {
    pub config: AcousticConfig,
    model_data: GpuBuffer,
    base_input: GpuBuffer,
    current_embed: GpuBuffer,
    zero_embed: GpuBuffer,
    fsq_latent: GpuBuffer,
    fsq_output: GpuBuffer,
    // Canonical VoxCPM2 LM state used by LocDiT and the stop predictor.
    // Text prefix positions copy the raw BaseLM hidden here; acoustic/generation
    // positions copy the post-FSQ hidden here, matching upstream inference.
    current_lm: GpuBuffer,
    hidden: GpuBuffer,
    fusion_snapshot: GpuBuffer,
    norm: GpuBuffer,
    q: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    attention: GpuBuffer,
    gate: GpuBuffer,
    branch: GpuBuffer,
    scores: GpuBuffer,
    kv_k: GpuBuffer,
    kv_v: GpuBuffer,
    stop_hidden: GpuBuffer,
    stop_logits: GpuBuffer,
    pipelines: Pipelines,
    command_buffer: vk::CommandBuffer,
    position: u32,
}

impl ResidualFsqEngine {
    pub fn new(
        gpu: &VulkanContext,
        acoustic: &GgufSummary,
        base: &BaseLmConfig,
        max_context: u32,
    ) -> Result<Self> {
        acoustic.validate_acoustic_f16()?;
        let config = AcousticConfig::from_gguf(acoustic, base, max_context)?;
        let data_bytes = acoustic.data_bytes()?;
        if data_bytes > gpu.info.max_storage_buffer_range {
            bail!(
                "Acoustic GGUF data region is {:.2} GiB, larger than maxStorageBufferRange {:.2} GiB. VoxGen refuses CPU fallback or implicit segmentation.",
                data_bytes as f64 / 1073741824.0,
                gpu.info.max_storage_buffer_range as f64 / 1073741824.0
            );
        }
        if data_bytes > u32::MAX as u64 {
            bail!("Acoustic GGUF data region exceeds 4 GiB; iteration-3 shaders use 32-bit byte offsets");
        }
        let file = File::open(&acoustic.path)
            .with_context(|| format!("open acoustic GGUF {}", acoustic.path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap acoustic GGUF {}", acoustic.path.display()))?;
        let start = usize::try_from(acoustic.data_offset).context("acoustic GGUF data offset exceeds usize")?;
        let model_data = gpu.upload_device_local(&mmap[start..])?;

        let storage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let f32_bytes = |n: u32| n as u64 * 4;
        let base_input = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let current_embed = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let zero_embed = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        gpu.upload_f32(&zero_embed, &vec![0.0f32; config.embedding_length as usize])?;
        let fsq_latent = gpu.create_buffer(f32_bytes(config.fsq_latent_dim), storage, local)?;
        let fsq_output = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let current_lm = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let hidden = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let fusion_snapshot = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let norm = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let q = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let k = gpu.create_buffer(f32_bytes(config.kv_dim), storage, local)?;
        let v = gpu.create_buffer(f32_bytes(config.kv_dim), storage, local)?;
        let attention = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let gate = gpu.create_buffer(f32_bytes(config.feed_forward_length), storage, local)?;
        let branch = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let scores = gpu.create_buffer(
            config.head_count as u64 * config.active_context_length as u64 * 4,
            storage,
            local,
        )?;
        if config.per_cache_bytes() > gpu.info.max_storage_buffer_range {
            bail!("ResidualLM KV cache exceeds selected device maxStorageBufferRange; reduce --max-context");
        }
        let kv_k = gpu.create_buffer(config.per_cache_bytes(), storage, local)?;
        let kv_v = gpu.create_buffer(config.per_cache_bytes(), storage, local)?;
        let stop_hidden = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let stop_logits = gpu.create_buffer(2 * 4, storage, local)?;

        let fsq_in = gpu.create_compute_pipeline(gpu.select_spirv(LINEAR_BIAS_SPV, LINEAR_BIAS_XTX7900_SPV), 3, std::mem::size_of::<LinearBiasPush>() as u32)?;
        fsq_in.bind_buffers(&[&model_data, &base_input, &fsq_latent]);
        let fsq_quantize = gpu.create_compute_pipeline(FSQ_QUANTIZE_SPV, 1, std::mem::size_of::<FsqPush>() as u32)?;
        fsq_quantize.bind_buffers(&[&fsq_latent]);
        let fsq_out = gpu.create_compute_pipeline(gpu.select_spirv(LINEAR_BIAS_SPV, LINEAR_BIAS_XTX7900_SPV), 3, std::mem::size_of::<LinearBiasPush>() as u32)?;
        fsq_out.bind_buffers(&[&model_data, &fsq_latent, &fsq_output]);
        let fusion = gpu.create_compute_pipeline(gpu.select_spirv(FUSION_LINEAR_SPV, FUSION_LINEAR_XTX7900_SPV), 5, std::mem::size_of::<FusionPush>() as u32)?;
        fusion.bind_buffers(&[&model_data, &fsq_output, &current_embed, &hidden, &fusion_snapshot]);

        let rms_attn = rms_pipeline(gpu, &model_data, &hidden, &norm)?;
        let residual_rms = gpu.create_compute_pipeline(
            gpu.select_spirv(RESIDUAL_RMSNORM_SPV, RESIDUAL_RMSNORM_XTX7900_SPV),
            4,
            std::mem::size_of::<ResidualRmsPush>() as u32,
        )?;
        residual_rms.bind_buffers(&[&model_data, &hidden, &branch, &norm]);
        let qkv = gpu.create_compute_pipeline(
            gpu.select_spirv(QKV_MATVEC_SPV, QKV_MATVEC_XTX7900_SPV),
            5,
            std::mem::size_of::<QkvPush>() as u32,
        )?;
        qkv.bind_buffers(&[&model_data, &norm, &q, &k, &v]);
        let attn_out = matvec_pipeline(gpu, &model_data, &attention, &branch)?;
        let swiglu = gpu.create_compute_pipeline(
            gpu.select_spirv(SWIGLU_SPV, SWIGLU_XTX7900_SPV),
            3,
            std::mem::size_of::<SwigluPush>() as u32,
        )?;
        swiglu.bind_buffers(&[&model_data, &norm, &gate]);
        let down_pipe = matvec_pipeline(gpu, &model_data, &gate, &branch)?;
        let store_kv = gpu.create_compute_pipeline(STORE_KV_SPV, 4, std::mem::size_of::<StoreKvPush>() as u32)?;
        store_kv.bind_buffers(&[&k, &v, &kv_k, &kv_v]);
        let attn_scores = gpu.create_compute_pipeline(gpu.select_spirv(ATTN_SCORES_SPV, ATTN_SCORES_XTX7900_SPV), 3, std::mem::size_of::<AttentionPush>() as u32)?;
        attn_scores.bind_buffers(&[&q, &kv_k, &scores]);
        let softmax = gpu.create_compute_pipeline(gpu.select_spirv(SOFTMAX_SPV, SOFTMAX_XTX7900_SPV), 1, std::mem::size_of::<SoftmaxPush>() as u32)?;
        softmax.bind_buffers(&[&scores]);
        let attn_values = gpu.create_compute_pipeline(ATTN_VALUES_SPV, 3, std::mem::size_of::<AttentionPush>() as u32)?;
        attn_values.bind_buffers(&[&scores, &kv_v, &attention]);
        let stop_linear1 = gpu.create_compute_pipeline(gpu.select_spirv(LINEAR_BIAS_SPV, LINEAR_BIAS_XTX7900_SPV), 3, std::mem::size_of::<LinearBiasPush>() as u32)?;
        stop_linear1.bind_buffers(&[&model_data, &base_input, &stop_hidden]);
        let stop_silu = gpu.create_compute_pipeline(SILU_INPLACE_SPV, 1, std::mem::size_of::<NPush>() as u32)?;
        stop_silu.bind_buffers(&[&stop_hidden]);
        let stop_linear2 = gpu.create_compute_pipeline(gpu.select_spirv(MATVEC_SPV, MATVEC_XTX7900_SPV), 3, std::mem::size_of::<MatvecPush>() as u32)?;
        stop_linear2.bind_buffers(&[&model_data, &stop_hidden, &stop_logits]);
        let command_buffer = gpu.allocate_primary_command_buffer()?;

        Ok(Self {
            config,
            model_data,
            base_input,
            current_embed,
            zero_embed,
            fsq_latent,
            fsq_output,
            current_lm,
            hidden,
            fusion_snapshot,
            norm,
            q,
            k,
            v,
            attention,
            gate,
            branch,
            scores,
            kv_k,
            kv_v,
            stop_hidden,
            stop_logits,
            pipelines: Pipelines {
                fsq_in,
                fsq_quantize,
                fsq_out,
                fusion,
                rms_attn,
                residual_rms,
                qkv,
                attn_out,
                swiglu,
                down: down_pipe,
                store_kv,
                attn_scores,
                softmax,
                attn_values,
                stop_linear1,
                stop_silu,
                stop_linear2,
            },
            command_buffer,
            position: 0,
        })
    }

    pub fn position(&self) -> u32 { self.position }

    pub fn model_data_buffer(&self) -> &GpuBuffer { &self.model_data }
    pub fn current_lm_buffer(&self) -> &GpuBuffer { &self.current_lm }
    pub fn reset(&mut self) { self.position = 0; }

    pub fn allocated_bytes(&self) -> u64 {
        self.model_data.size
            + self.base_input.size
            + self.current_embed.size
            + self.zero_embed.size
            + self.fsq_latent.size
            + self.fsq_output.size
            + self.current_lm.size
            + self.hidden.size
            + self.fusion_snapshot.size
            + self.norm.size
            + self.q.size
            + self.k.size
            + self.v.size
            + self.attention.size
            + self.gate.size
            + self.branch.size
            + self.scores.size
            + self.kv_k.size
            + self.kv_v.size
            + self.stop_hidden.size
            + self.stop_logits.size
    }

    pub fn output_buffer(&self) -> &GpuBuffer { &self.norm }

    /// VoxCPM2 stop predictor: Linear(2048->2048)+SiLU+Linear(2048->2).
    /// The BaseLM hidden stays on Vulkan; only two logits are read back.
    pub fn predict_stop_from_gpu_base(&mut self, gpu:&VulkanContext, acoustic:&GgufSummary, base_hidden:&GpuBuffer) -> Result<StopPrediction> {
        if base_hidden.size < self.config.embedding_length as u64 * 4 { bail!("GPU BaseLM hidden is too small for stop predictor"); }
        self.pipelines.stop_linear1.bind_buffers(&[&self.model_data, base_hidden, &self.stop_hidden]);
        gpu.begin_one_time(self.command_buffer)?;
        gpu.compute_barrier(self.command_buffer);
        let w1=acoustic.tensor("stop_predictor.linear1.weight")?; let b1=acoustic.tensor("stop_predictor.linear1.bias")?;
        self.dispatch_linear_bias(gpu,&self.pipelines.stop_linear1,w1,b1,self.config.embedding_length,self.config.embedding_length)?;
        gpu.compute_barrier(self.command_buffer);
        self.pipelines.stop_silu.bind(self.command_buffer); self.pipelines.stop_silu.push(self.command_buffer,&NPush{n:self.config.embedding_length});
        unsafe{gpu.device.cmd_dispatch(self.command_buffer,div_ceil(self.config.embedding_length,256),1,1)};
        gpu.compute_barrier(self.command_buffer);
        let w2=acoustic.tensor("stop_predictor.linear2.weight")?;
        self.dispatch_matvec(gpu,&self.pipelines.stop_linear2,w2,2,self.config.embedding_length)?;
        let started=Instant::now(); let logits=gpu.submit_and_read_f32(self.command_buffer,&self.stop_logits,2)?; let elapsed_ms=started.elapsed().as_secs_f64()*1000.0;
        Ok(StopPrediction{stop:logits[1]>logits[0],continue_logit:logits[0],stop_logit:logits[1],elapsed_ms})
    }

    /// Stop predictor over the canonical VoxCPM2 LM state. The state is raw for
    /// text prefix positions and post-FSQ for acoustic/generated positions.
    pub fn predict_stop_from_current_lm(&mut self, gpu:&VulkanContext, acoustic:&GgufSummary) -> Result<StopPrediction> {
        self.pipelines.stop_linear1.bind_buffers(&[&self.model_data, &self.current_lm, &self.stop_hidden]);
        gpu.begin_one_time(self.command_buffer)?;
        gpu.compute_barrier(self.command_buffer);
        let w1=acoustic.tensor("stop_predictor.linear1.weight")?; let b1=acoustic.tensor("stop_predictor.linear1.bias")?;
        self.dispatch_linear_bias(gpu,&self.pipelines.stop_linear1,w1,b1,self.config.embedding_length,self.config.embedding_length)?;
        gpu.compute_barrier(self.command_buffer);
        self.pipelines.stop_silu.bind(self.command_buffer); self.pipelines.stop_silu.push(self.command_buffer,&NPush{n:self.config.embedding_length});
        unsafe{gpu.device.cmd_dispatch(self.command_buffer,div_ceil(self.config.embedding_length,256),1,1)};
        gpu.compute_barrier(self.command_buffer);
        let w2=acoustic.tensor("stop_predictor.linear2.weight")?;
        self.dispatch_matvec(gpu,&self.pipelines.stop_linear2,w2,2,self.config.embedding_length)?;
        let started=Instant::now(); let logits=gpu.submit_and_read_f32(self.command_buffer,&self.stop_logits,2)?; let elapsed_ms=started.elapsed().as_secs_f64()*1000.0;
        Ok(StopPrediction{stop:logits[1]>logits[0],continue_logit:logits[0],stop_logit:logits[1],elapsed_ms})
    }

    pub fn fsq_only(
        &mut self,
        gpu: &VulkanContext,
        acoustic: &GgufSummary,
        base_hidden: &[f32],
    ) -> Result<FsqResult> {
        self.check_embedding(base_hidden, "FSQ input")?;
        gpu.upload_f32(&self.base_input, base_hidden)?;
        self.pipelines.fsq_in.bind_buffers(&[&self.model_data, &self.base_input, &self.fsq_latent]);
        gpu.begin_one_time(self.command_buffer)?;
        // Covers the GPU-only BaseLM -> FSQ handoff when the BaseLM output was
        // produced by an earlier submission on the same compute queue.
        gpu.compute_barrier(self.command_buffer);
        self.record_fsq(gpu, acoustic)?;
        let started = Instant::now();
        gpu.submit_and_wait(self.command_buffer)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let latent = gpu.read_f32(&self.fsq_latent, self.config.fsq_latent_dim as usize)?;
        let output = gpu.read_f32(&self.fsq_output, self.config.embedding_length as usize)?;
        let (latent_checksum, latent_l2) = stats(&latent);
        let (output_checksum, output_l2) = stats(&output);
        Ok(FsqResult { elapsed_ms, output_checksum, output_l2, latent_checksum, latent_l2 })
    }

    pub fn step(
        &mut self,
        gpu: &VulkanContext,
        acoustic: &GgufSummary,
        base_hidden: &[f32],
        current_embedding: &[f32],
    ) -> Result<ResidualStepResult> {
        self.check_embedding(base_hidden, "ResidualLM base hidden")?;
        self.check_embedding(current_embedding, "ResidualLM current embedding")?;
        gpu.upload_f32(&self.base_input, base_hidden)?;
        gpu.upload_f32(&self.current_embed, current_embedding)?;
        self.pipelines.fsq_in.bind_buffers(&[&self.model_data, &self.base_input, &self.fsq_latent]);
        self.pipelines.fusion.bind_buffers(&[&self.model_data, &self.fsq_output, &self.current_embed, &self.hidden, &self.fusion_snapshot]);
        self.step_record_submit(gpu, acoustic, true, None)
    }

    pub fn step_from_gpu_base(
        &mut self,
        gpu: &VulkanContext,
        acoustic: &GgufSummary,
        base_hidden_gpu: &GpuBuffer,
        current_embedding: &[f32],
    ) -> Result<ResidualStepResult> {
        self.check_embedding(current_embedding, "ResidualLM current embedding")?;
        if base_hidden_gpu.size < self.config.embedding_length as u64 * 4 {
            bail!("GPU BaseLM output buffer is too small for 2048 f32 values");
        }
        gpu.upload_f32(&self.current_embed, current_embedding)?;
        self.pipelines.fsq_in.bind_buffers(&[&self.model_data, base_hidden_gpu, &self.fsq_latent]);
        self.pipelines.fusion.bind_buffers(&[&self.model_data, &self.fsq_output, &self.current_embed, &self.hidden, &self.fusion_snapshot]);
        self.step_record_submit(gpu, acoustic, true, None)
    }

    /// Audio-prefix/generation path with both inputs already resident on Vulkan.
    pub fn step_from_gpu_base_and_embedding(
        &mut self,
        gpu: &VulkanContext,
        acoustic: &GgufSummary,
        base_hidden_gpu: &GpuBuffer,
        current_embedding_gpu: &GpuBuffer,
    ) -> Result<ResidualStepResult> {
        let bytes = self.config.embedding_length as u64 * 4;
        if base_hidden_gpu.size < bytes || current_embedding_gpu.size < bytes {
            bail!("GPU BaseLM/current-embedding buffer is too small for ResidualLM input");
        }
        self.pipelines.fsq_in.bind_buffers(&[&self.model_data, base_hidden_gpu, &self.fsq_latent]);
        self.pipelines.fusion.bind_buffers(&[&self.model_data, &self.fsq_output, current_embedding_gpu, &self.hidden, &self.fusion_snapshot]);
        self.step_record_submit(gpu, acoustic, true, None)
    }

    pub fn step_from_gpu_base_and_embedding_gpu_only(&mut self,gpu:&VulkanContext,acoustic:&GgufSummary,base_hidden_gpu:&GpuBuffer,current_embedding_gpu:&GpuBuffer)->Result<ResidualGpuStep>{
        let bytes=self.config.embedding_length as u64*4;if base_hidden_gpu.size<bytes||current_embedding_gpu.size<bytes{bail!("GPU BaseLM/current-embedding buffer is too small for ResidualLM input");}
        self.pipelines.fsq_in.bind_buffers(&[&self.model_data,base_hidden_gpu,&self.fsq_latent]);self.pipelines.fusion.bind_buffers(&[&self.model_data,&self.fsq_output,current_embedding_gpu,&self.hidden,&self.fusion_snapshot]);self.step_record_submit_gpu_only(gpu,acoustic,true,None)
    }

    pub fn step_text_prefix_from_gpu_base_gpu_only(&mut self,gpu:&VulkanContext,acoustic:&GgufSummary,base_hidden_gpu:&GpuBuffer)->Result<ResidualGpuStep>{
        if base_hidden_gpu.size<self.config.embedding_length as u64*4{bail!("GPU BaseLM output buffer is too small for ResidualLM text prefix");}
        self.pipelines.fusion.bind_buffers(&[&self.model_data,base_hidden_gpu,&self.zero_embed,&self.hidden,&self.fusion_snapshot]);self.step_record_submit_gpu_only(gpu,acoustic,false,Some(base_hidden_gpu))
    }

    /// VoxCPM2 text-prefix path: BaseLM hidden is *not* FSQ-quantized and the
    /// acoustic half of fusion_concat_proj is zero because feat_mask=0 for text.
    pub fn step_text_prefix_from_gpu_base(
        &mut self,
        gpu: &VulkanContext,
        acoustic: &GgufSummary,
        base_hidden_gpu: &GpuBuffer,
    ) -> Result<ResidualStepResult> {
        if base_hidden_gpu.size < self.config.embedding_length as u64 * 4 {
            bail!("GPU BaseLM output buffer is too small for 2048 f32 values");
        }
        self.pipelines.fusion.bind_buffers(&[
            &self.model_data,
            base_hidden_gpu,
            &self.zero_embed,
            &self.hidden,
            &self.fusion_snapshot,
        ]);
        self.step_record_submit(gpu, acoustic, false, Some(base_hidden_gpu))
    }

    pub fn benchmark(
        &mut self,
        gpu: &VulkanContext,
        acoustic: &GgufSummary,
        base_hidden: &[f32],
        current_embedding: &[f32],
        warmup: u32,
        iterations: u32,
    ) -> Result<ResidualBenchmark> {
        if iterations == 0 { bail!("benchmark iterations must be > 0"); }
        if warmup.saturating_add(iterations) > self.config.active_context_length {
            bail!("ResidualLM benchmark exceeds --max-context");
        }
        self.reset();
        for _ in 0..warmup {
            let _ = self.step(gpu, acoustic, base_hidden, current_embedding)?;
        }
        let mut times = Vec::with_capacity(iterations as usize);
        for _ in 0..iterations {
            times.push(self.step(gpu, acoustic, base_hidden, current_embedding)?.elapsed_ms);
        }
        let mean_ms = times.iter().sum::<f64>() / times.len() as f64;
        let min_ms = times.iter().copied().fold(f64::INFINITY, f64::min);
        let max_ms = times.iter().copied().fold(0.0, f64::max);
        Ok(ResidualBenchmark {
            iterations,
            mean_ms,
            min_ms,
            max_ms,
            steps_per_second: 1000.0 / mean_ms,
        })
    }

    /// Record one ResidualLM prefill position into an already-begun command buffer.
    /// No submit/wait occurs here; this is paired with BaseLM recording by the
    /// xtx7900 cross-engine prefill batcher.
    pub(crate) fn record_text_prefix_from_gpu_base_in(
        &mut self,
        gpu: &VulkanContext,
        acoustic: &GgufSummary,
        base_hidden_gpu: &GpuBuffer,
        cmd: vk::CommandBuffer,
    ) -> Result<u32> {
        if base_hidden_gpu.size < self.config.embedding_length as u64 * 4 {
            bail!("GPU BaseLM output buffer is too small for ResidualLM text prefix");
        }
        if self.position >= self.config.active_context_length {
            bail!("ResidualLM KV cache is full at {} positions", self.config.active_context_length);
        }
        self.pipelines.fusion.bind_buffers(&[
            &self.model_data, base_hidden_gpu, &self.zero_embed, &self.hidden, &self.fusion_snapshot,
        ]);
        let position = self.position;
        let old_cmd = self.command_buffer;
        self.command_buffer = cmd;
        let recorded = (|| -> Result<()> {
            // BaseLM output was produced earlier in the same command buffer.
            gpu.compute_barrier(cmd);
            self.record_current_lm_copy(gpu, base_hidden_gpu);
            gpu.compute_barrier(cmd);
            self.record_fusion(gpu, acoustic)?;
            gpu.compute_barrier(cmd);
            self.record_residual_lm(gpu, acoustic, position)?;
            Ok(())
        })();
        self.command_buffer = old_cmd;
        recorded?;
        self.position += 1;
        Ok(position)
    }

    fn step_record_submit_gpu_only(&mut self,gpu:&VulkanContext,acoustic:&GgufSummary,fsq_applied:bool,raw_lm_state:Option<&GpuBuffer>)->Result<ResidualGpuStep>{
        if self.position>=self.config.active_context_length{bail!("ResidualLM KV cache is full at {} positions",self.config.active_context_length);}
        gpu.begin_one_time(self.command_buffer)?;
        gpu.compute_barrier(self.command_buffer);
        if fsq_applied {
            self.record_fsq(gpu,acoustic)?;
            self.record_current_lm_copy(gpu,&self.fsq_output);
        } else {
            let src=raw_lm_state.context("text-prefix ResidualLM step requires raw BaseLM state")?;
            self.record_current_lm_copy(gpu,src);
        }
        gpu.compute_barrier(self.command_buffer);
        self.record_fusion(gpu,acoustic)?;gpu.compute_barrier(self.command_buffer);
        let position=self.position;self.record_residual_lm(gpu,acoustic,position)?;
        let started=Instant::now();gpu.submit_and_wait(self.command_buffer)?;let elapsed_ms=started.elapsed().as_secs_f64()*1000.0;self.position+=1;Ok(ResidualGpuStep{position,elapsed_ms,fsq_applied})
    }

    fn step_record_submit(&mut self, gpu: &VulkanContext, acoustic: &GgufSummary, fsq_applied: bool, raw_lm_state: Option<&GpuBuffer>) -> Result<ResidualStepResult> {
        if self.position >= self.config.active_context_length {
            bail!("ResidualLM KV cache is full at {} positions", self.config.active_context_length);
        }
        gpu.begin_one_time(self.command_buffer)?;
        // Synchronize a possible BaseLM shader producer from the previous queue submission.
        gpu.compute_barrier(self.command_buffer);
        if fsq_applied {
            self.record_fsq(gpu, acoustic)?;
            self.record_current_lm_copy(gpu, &self.fsq_output);
        } else {
            let src = raw_lm_state.context("text-prefix ResidualLM step requires raw BaseLM state")?;
            self.record_current_lm_copy(gpu, src);
        }
        gpu.compute_barrier(self.command_buffer);
        self.record_fusion(gpu, acoustic)?;
        gpu.compute_barrier(self.command_buffer);
        let position = self.position;
        self.record_residual_lm(gpu, acoustic, position)?;
        let started = Instant::now();
        gpu.submit_and_wait(self.command_buffer)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.position += 1;

        let (fsq_checksum, fsq_l2) = if fsq_applied {
            let fsq = gpu.read_f32(&self.fsq_output, self.config.embedding_length as usize)?;
            let (checksum, l2) = stats(&fsq);
            (Some(checksum), Some(l2))
        } else {
            (None, None)
        };
        let fused = gpu.read_f32(&self.fusion_snapshot, self.config.embedding_length as usize)?;
        let output = gpu.read_f32(&self.norm, self.config.embedding_length as usize)?;
        let (fused_input_checksum, fused_input_l2) = stats(&fused);
        let (residual_hidden_checksum, residual_hidden_l2) = stats(&output);
        Ok(ResidualStepResult {
            position,
            elapsed_ms,
            fsq_applied,
            fsq_checksum,
            fsq_l2,
            fused_input_checksum,
            fused_input_l2,
            residual_hidden_checksum,
            residual_hidden_l2,
        })
    }

    fn record_current_lm_copy(&self, gpu:&VulkanContext, src:&GpuBuffer) {
        // MiniCPM/VoxCPM2 changes the semantic LM state contract after audio
        // positions: LocDiT and stop prediction consume FSQ(BaseLM), not the raw
        // BaseLM output. Keep one canonical device-local buffer for that state.
        gpu.compute_to_transfer_rw_barrier(self.command_buffer);
        unsafe {
            gpu.device.cmd_copy_buffer(
                self.command_buffer, src.buffer, self.current_lm.buffer,
                &[vk::BufferCopy{src_offset:0,dst_offset:0,size:self.config.embedding_length as u64*4}],
            );
        }
        gpu.transfer_to_compute_barrier(self.command_buffer);
    }

    fn record_fsq(&self, gpu: &VulkanContext, acoustic: &GgufSummary) -> Result<()> {
        let in_w = acoustic.tensor("fsq.in_proj.weight")?;
        let in_b = acoustic.tensor("fsq.in_proj.bias")?;
        self.dispatch_linear_bias(
            gpu,
            &self.pipelines.fsq_in,
            in_w,
            in_b,
            self.config.fsq_latent_dim,
            self.config.embedding_length,
        )?;
        gpu.compute_barrier(self.command_buffer);
        self.pipelines.fsq_quantize.bind(self.command_buffer);
        self.pipelines.fsq_quantize.push(
            self.command_buffer,
            &FsqPush { n: self.config.fsq_latent_dim, scale: self.config.fsq_scale },
        );
        unsafe { gpu.device.cmd_dispatch(self.command_buffer, div_ceil(self.config.fsq_latent_dim, 256), 1, 1) };
        gpu.compute_barrier(self.command_buffer);
        let out_w = acoustic.tensor("fsq.out_proj.weight")?;
        let out_b = acoustic.tensor("fsq.out_proj.bias")?;
        self.dispatch_linear_bias(
            gpu,
            &self.pipelines.fsq_out,
            out_w,
            out_b,
            self.config.embedding_length,
            self.config.fsq_latent_dim,
        )?;
        Ok(())
    }

    fn record_fusion(&self, gpu: &VulkanContext, acoustic: &GgufSummary) -> Result<()> {
        let w = acoustic.tensor("projections.res_fusion_proj.weight")?;
        let b = acoustic.tensor("projections.res_fusion_proj.bias")?;
        let max_x = gpu.info.max_compute_work_group_count_x.max(1);
        let mut row_base = 0u32;
        while row_base < self.config.embedding_length {
            let groups = (self.config.embedding_length - row_base).min(max_x);
            self.pipelines.fusion.bind(self.command_buffer);
            self.pipelines.fusion.push(
                self.command_buffer,
                &FusionPush {
                    weight_offset: tensor_offset_u32(w)?,
                    bias_offset: tensor_offset_u32(b)?,
                    rows: self.config.embedding_length,
                    left_cols: self.config.embedding_length,
                    right_cols: self.config.embedding_length,
                    weight_dtype: scalar_dtype_code(w.ggml_type)?,
                    bias_dtype: scalar_dtype_code(b.ggml_type)?,
                    row_base,
                },
            );
            unsafe { gpu.device.cmd_dispatch(self.command_buffer, groups, 1, 1) };
            row_base += groups;
        }
        Ok(())
    }

    fn record_residual_lm(&self, gpu: &VulkanContext, acoustic: &GgufSummary, position: u32) -> Result<()> {
        if !self.config.no_rope { bail!("VoxCPM2 ResidualLM must run with no_rope=true"); }
        let q_per_kv = self.config.head_count / self.config.head_count_kv;
        self.dispatch_rms(gpu, &self.pipelines.rms_attn, acoustic.tensor("residual_lm.blk.0.attn_norm.weight")?)?;
        gpu.compute_buffer_barrier(self.command_buffer, &[&self.norm]);
        for layer in 0..self.config.residual_block_count {
            let p = |suffix: &str| format!("residual_lm.blk.{layer}.{suffix}");
            self.dispatch_qkv(
                gpu,
                acoustic.tensor(&p("attn_q.weight"))?,
                acoustic.tensor(&p("attn_k.weight"))?,
                acoustic.tensor(&p("attn_v.weight"))?,
            )?;
            gpu.compute_buffer_barrier(self.command_buffer, &[&self.q, &self.k, &self.v]);

            self.pipelines.store_kv.bind(self.command_buffer);
            self.pipelines.store_kv.push(
                self.command_buffer,
                &StoreKvPush { layer, position, context: self.config.active_context_length, kv_dim: self.config.kv_dim },
            );
            unsafe { gpu.device.cmd_dispatch(self.command_buffer, 1, 1, 1) };
            gpu.compute_buffer_barrier(self.command_buffer, &[&self.kv_k, &self.kv_v]);

            let attn = AttentionPush {
                layer,
                context: self.config.active_context_length,
                position,
                head_dim: self.config.head_dim,
                q_per_kv,
                kv_heads: self.config.head_count_kv,
            };
            self.pipelines.attn_scores.bind(self.command_buffer);
            self.pipelines.attn_scores.push(self.command_buffer, &attn);
            unsafe { gpu.device.cmd_dispatch(self.command_buffer, position + 1, self.config.head_count, 1) };
            gpu.compute_buffer_barrier(self.command_buffer, &[&self.scores]);
            self.pipelines.softmax.bind(self.command_buffer);
            self.pipelines.softmax.push(self.command_buffer, &SoftmaxPush { context: self.config.active_context_length, position });
            unsafe { gpu.device.cmd_dispatch(self.command_buffer, self.config.head_count, 1, 1) };
            gpu.compute_buffer_barrier(self.command_buffer, &[&self.scores]);
            self.pipelines.attn_values.bind(self.command_buffer);
            self.pipelines.attn_values.push(self.command_buffer, &attn);
            unsafe { gpu.device.cmd_dispatch(self.command_buffer, self.config.head_count, 1, 1) };
            gpu.compute_buffer_barrier(self.command_buffer, &[&self.attention]);
            self.dispatch_matvec(gpu, &self.pipelines.attn_out, acoustic.tensor(&p("attn_output.weight"))?, self.config.embedding_length, self.config.embedding_length)?;
            gpu.compute_buffer_barrier(self.command_buffer, &[&self.branch]);
            self.dispatch_residual_rms(gpu, acoustic.tensor(&p("ffn_norm.weight"))?)?;
            gpu.compute_buffer_barrier(self.command_buffer, &[&self.norm]);

            self.dispatch_swiglu(
                gpu,
                acoustic.tensor(&p("ffn_gate.weight"))?,
                acoustic.tensor(&p("ffn_up.weight"))?,
                self.config.feed_forward_length,
                self.config.embedding_length,
            )?;
            gpu.compute_buffer_barrier(self.command_buffer, &[&self.gate]);
            self.dispatch_matvec(gpu, &self.pipelines.down, acoustic.tensor(&p("ffn_down.weight"))?, self.config.embedding_length, self.config.feed_forward_length)?;
            gpu.compute_buffer_barrier(self.command_buffer, &[&self.branch]);
            let next_norm = if layer + 1 < self.config.residual_block_count {
                acoustic.tensor(&format!("residual_lm.blk.{}.attn_norm.weight", layer + 1))?
            } else {
                acoustic.tensor("residual_lm.output_norm.weight")?
            };
            self.dispatch_residual_rms(gpu, next_norm)?;
            gpu.compute_buffer_barrier(self.command_buffer, &[&self.norm]);
        }
        Ok(())
    }

    fn dispatch_rms(&self, gpu: &VulkanContext, pipeline: &ComputePipeline, weight: &TensorInfo) -> Result<()> {
        let span = gpu.gpu_profile_begin(self.command_buffer, "residual.rmsnorm");
        pipeline.bind(self.command_buffer);
        pipeline.push(
            self.command_buffer,
            &RmsPush { weight_offset: tensor_offset_u32(weight)?, n: self.config.embedding_length, eps: self.config.rms_epsilon },
        );
        unsafe { gpu.device.cmd_dispatch(self.command_buffer, 1, 1, 1) };
        gpu.gpu_profile_end(self.command_buffer, span);
        Ok(())
    }

    fn dispatch_qkv(
        &self,
        gpu: &VulkanContext,
        q_weight: &TensorInfo,
        k_weight: &TensorInfo,
        v_weight: &TensorInfo,
    ) -> Result<()> {
        let cols = self.config.embedding_length;
        if q_weight.dims != [cols as u64, self.config.embedding_length as u64]
            || k_weight.dims != [cols as u64, self.config.kv_dim as u64]
            || v_weight.dims != [cols as u64, self.config.kv_dim as u64]
        {
            bail!("ResidualLM QKV projection shape mismatch");
        }
        if q_weight.ggml_type != GgmlType::F16 || k_weight.ggml_type != GgmlType::F16 || v_weight.ggml_type != GgmlType::F16 {
            bail!("ResidualLM QKV weights must be F16");
        }
        let total_rows = self.config.embedding_length + 2 * self.config.kv_dim;
        let max_x = gpu.info.max_compute_work_group_count_x.max(1);
        let span = gpu.gpu_profile_begin(self.command_buffer, "residual.qkv");
        let mut row_base = 0u32;
        while row_base < total_rows {
            let groups = (total_rows - row_base).min(max_x);
            self.pipelines.qkv.bind(self.command_buffer);
            self.pipelines.qkv.push(self.command_buffer, &QkvPush {
                q_weight_offset: tensor_offset_u32(q_weight)?,
                k_weight_offset: tensor_offset_u32(k_weight)?,
                v_weight_offset: tensor_offset_u32(v_weight)?,
                q_rows: self.config.embedding_length,
                kv_rows: self.config.kv_dim,
                cols,
                q_dtype: 1,
                k_dtype: 1,
                v_dtype: 1,
                row_base,
            });
            unsafe { gpu.device.cmd_dispatch(self.command_buffer, groups, 1, 1) };
            row_base += groups;
        }
        gpu.gpu_profile_end(self.command_buffer, span);
        Ok(())
    }

    fn dispatch_matvec(
        &self,
        gpu: &VulkanContext,
        pipeline: &ComputePipeline,
        weight: &TensorInfo,
        rows: u32,
        cols: u32,
    ) -> Result<()> {
        if weight.dims != [cols as u64, rows as u64] {
            bail!("matvec shape mismatch for {}", weight.name);
        }
        if weight.ggml_type != GgmlType::F16 {
            bail!("ResidualLM matrix {} must be F16, got {}", weight.name, weight.ggml_type.name());
        }
        let span = gpu.gpu_profile_begin(self.command_buffer, "residual.matvec");
        let max_x = gpu.info.max_compute_work_group_count_x.max(1);
        let mut row_base = 0u32;
        while row_base < rows {
            let groups = (rows - row_base).min(max_x);
            pipeline.bind(self.command_buffer);
            pipeline.push(
                self.command_buffer,
                &MatvecPush {
                    weight_offset: tensor_offset_u32(weight)?,
                    rows,
                    cols,
                    dtype: 1,
                    row_base,
                    alpha: 1.0,
                },
            );
            unsafe { gpu.device.cmd_dispatch(self.command_buffer, groups, 1, 1) };
            row_base += groups;
        }
        gpu.gpu_profile_end(self.command_buffer, span);
        Ok(())
    }

    fn dispatch_linear_bias(
        &self,
        gpu: &VulkanContext,
        pipeline: &ComputePipeline,
        weight: &TensorInfo,
        bias: &TensorInfo,
        rows: u32,
        cols: u32,
    ) -> Result<()> {
        let max_x = gpu.info.max_compute_work_group_count_x.max(1);
        let mut row_base = 0u32;
        while row_base < rows {
            let groups = (rows - row_base).min(max_x);
            pipeline.bind(self.command_buffer);
            pipeline.push(
                self.command_buffer,
                &LinearBiasPush {
                    weight_offset: tensor_offset_u32(weight)?,
                    bias_offset: tensor_offset_u32(bias)?,
                    rows,
                    cols,
                    weight_dtype: scalar_dtype_code(weight.ggml_type)?,
                    bias_dtype: scalar_dtype_code(bias.ggml_type)?,
                    row_base,
                    alpha: 1.0,
                },
            );
            unsafe { gpu.device.cmd_dispatch(self.command_buffer, groups, 1, 1) };
            row_base += groups;
        }
        Ok(())
    }

    fn dispatch_swiglu(
        &self,
        gpu: &VulkanContext,
        gate_weight: &TensorInfo,
        up_weight: &TensorInfo,
        rows: u32,
        cols: u32,
    ) -> Result<()> {
        if gate_weight.dims != [cols as u64, rows as u64] || up_weight.dims != [cols as u64, rows as u64] {
            bail!("ResidualLM SwiGLU projection shape mismatch");
        }
        if gate_weight.ggml_type != GgmlType::F16 || up_weight.ggml_type != GgmlType::F16 {
            bail!("ResidualLM SwiGLU matrices must be F16");
        }
        let span = gpu.gpu_profile_begin(self.command_buffer, "residual.swiglu");
        let max_x = gpu.info.max_compute_work_group_count_x.max(1);
        let mut row_base = 0u32;
        while row_base < rows {
            let groups = (rows - row_base).min(max_x);
            self.pipelines.swiglu.bind(self.command_buffer);
            self.pipelines.swiglu.push(self.command_buffer, &SwigluPush {
                gate_weight_offset: tensor_offset_u32(gate_weight)?,
                up_weight_offset: tensor_offset_u32(up_weight)?,
                rows,
                cols,
                dtype: 1,
                row_base,
            });
            unsafe { gpu.device.cmd_dispatch(self.command_buffer, groups, 1, 1) };
            row_base += groups;
        }
        gpu.gpu_profile_end(self.command_buffer, span);
        Ok(())
    }

    fn dispatch_residual_rms(&self, gpu: &VulkanContext, weight: &TensorInfo) -> Result<()> {
        let span = gpu.gpu_profile_begin(self.command_buffer, "residual.residual_rmsnorm");
        self.pipelines.residual_rms.bind(self.command_buffer);
        self.pipelines.residual_rms.push(self.command_buffer, &ResidualRmsPush {
            weight_offset: tensor_offset_u32(weight)?,
            n: self.config.embedding_length,
            eps: self.config.rms_epsilon,
            scale: self.config.residual_scale,
        });
        unsafe { gpu.device.cmd_dispatch(self.command_buffer, 1, 1, 1) };
        gpu.gpu_profile_end(self.command_buffer, span);
        Ok(())
    }

    fn check_embedding(&self, values: &[f32], what: &str) -> Result<()> {
        if values.len() != self.config.embedding_length as usize {
            bail!("{what} has {} floats; expected {}", values.len(), self.config.embedding_length);
        }
        Ok(())
    }
}

fn rms_pipeline(gpu: &VulkanContext, model: &GpuBuffer, input: &GpuBuffer, output: &GpuBuffer) -> Result<ComputePipeline> {
    let p = gpu.create_compute_pipeline(gpu.select_spirv(RMSNORM_SPV, RMSNORM_XTX7900_SPV), 3, std::mem::size_of::<RmsPush>() as u32)?;
    p.bind_buffers(&[model, input, output]);
    Ok(p)
}

fn matvec_pipeline(gpu: &VulkanContext, model: &GpuBuffer, input: &GpuBuffer, output: &GpuBuffer) -> Result<ComputePipeline> {
    let p = gpu.create_compute_pipeline(gpu.select_spirv(MATVEC_SPV, MATVEC_XTX7900_SPV), 3, std::mem::size_of::<MatvecPush>() as u32)?;
    p.bind_buffers(&[model, input, output]);
    Ok(p)
}

fn tensor_offset_u32(t: &TensorInfo) -> Result<u32> {
    u32::try_from(t.offset)
        .with_context(|| format!("tensor {} offset {} exceeds 32-bit shader address space", t.name, t.offset))
}

fn scalar_dtype_code(t: GgmlType) -> Result<u32> {
    match t {
        GgmlType::F32 => Ok(0),
        GgmlType::F16 => Ok(1),
        GgmlType::Q8_0 => Ok(8),
        other => bail!("unsupported scalar/matrix type {}", other.name()),
    }
}

fn effective_scale(v: f32) -> f32 { if v == 0.0 { 1.0 } else { v } }

fn stats(values: &[f32]) -> (f64, f64) {
    let mut checksum = 0.0f64;
    let mut squares = 0.0f64;
    for (i, &v) in values.iter().enumerate() {
        let x = v as f64;
        checksum += x * ((i % 251 + 1) as f64);
        squares += x * x;
    }
    (checksum, squares.sqrt())
}

fn div_ceil(n: u32, d: u32) -> u32 { (n + d - 1) / d }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsq_reference_quantizer_matches_scale_nine_grid() {
        let scale = 9.0f32;
        let input = 0.42f32;
        let got = (input.tanh() * scale).round() / scale;
        assert!((got - 0.44444445).abs() < 1e-6);
    }

    #[test]
    fn zero_residual_scale_means_identity_for_non_mup_voxcpm2() {
        assert_eq!(effective_scale(0.0), 1.0);
    }
}
