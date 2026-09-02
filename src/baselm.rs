use crate::{
    gguf::{BaseFormat, GgmlType, GgufSummary, TensorInfo},
    vulkan::{ComputePipeline, GpuBuffer, VulkanContext},
};
use anyhow::{bail, Context, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use serde::Serialize;
use std::{
    fs::File,
    path::Path,
    time::Instant,
};

const EMBEDDING_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/embedding.spv"));
const RMSNORM_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm.spv"));
const RMSNORM_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm_xtx7900.spv"));
const MATVEC_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matvec.spv"));
const MATVEC_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/matvec_xtx7900.spv"));
const QKV_MATVEC_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qkv_matvec.spv"));
const QKV_MATVEC_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qkv_matvec_xtx7900.spv"));
const ROPE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv"));
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

#[derive(Debug, Clone, Serialize)]
pub struct BaseLmConfig {
    pub architecture: String,
    pub context_length: u32,
    pub active_context_length: u32,
    pub embedding_length: u32,
    pub block_count: u32,
    pub feed_forward_length: u32,
    pub vocab_size: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub head_dim: u32,
    pub kv_dim: u32,
    pub rms_epsilon: f32,
    pub rope_theta: f32,
    pub rope_dimension_count: u32,
    pub rope_original_context_length: u32,
    pub rope_scaling_type: String,
    pub embedding_scale: f32,
    pub residual_scale: f32,
    pub logit_scale: f32,
    pub tied_output: bool,
    pub output_tensor: String,
}

impl BaseLmConfig {
    pub fn from_gguf(summary: &GgufSummary, active_context_length: u32) -> Result<Self> {
        let architecture = summary.metadata_str("general.architecture")?.to_owned();
        if architecture != "minicpm4" {
            bail!(
                "VoxGen BaseLM expects general.architecture=minicpm4, got {architecture:?}"
            );
        }
        let context_length = meta_u32(summary, &["minicpm4.context_length"])?;
        if active_context_length == 0 || active_context_length > context_length {
            bail!(
                "--max-context must be in 1..={context_length}, got {active_context_length}"
            );
        }
        let embedding_length = meta_u32(summary, &["minicpm4.embedding_length"])?;
        let block_count = meta_u32(summary, &["minicpm4.block_count"])?;
        let feed_forward_length = meta_u32(summary, &["minicpm4.feed_forward_length"])?;
        let head_count = meta_u32(summary, &["minicpm4.attention.head_count"])?;
        let head_count_kv = meta_u32(summary, &["minicpm4.attention.head_count_kv"])?;
        let head_dim = meta_u32(
            summary,
            &[
                "minicpm4.attention.key_length",
                "minicpm4.rope.dimension_count",
            ],
        )?;
        let rope_dimension_count = meta_u32(summary, &["minicpm4.rope.dimension_count"])?;
        let rms_epsilon = meta_f32(summary, &["minicpm4.attention.layer_norm_rms_epsilon"])?;
        let rope_theta = meta_f32(summary, &["minicpm4.rope.freq_base"])?;
        let rope_scaling_type = meta_string(summary, &["minicpm4.rope.scaling.type"])
            .unwrap_or_else(|| "none".to_string());
        let rope_original_context_length = meta_u32_optional(
            summary,
            &[
                "minicpm4.rope.scaling.original_context_length",
                "minicpm4.rope.scaling.original_context_len",
            ],
        )
        .unwrap_or(context_length);
        let embedding_scale = meta_f32_optional(summary, &["minicpm4.embedding_scale"]).unwrap_or(0.0);
        let residual_scale = meta_f32_optional(summary, &["minicpm4.residual_scale"]).unwrap_or(0.0);
        let logit_scale = meta_f32_optional(summary, &["minicpm4.logit_scale"]).unwrap_or(0.0);

        if embedding_length != 2048
            || block_count != 28
            || feed_forward_length != 6144
            || head_count != 16
            || head_count_kv != 2
            || head_dim != 128
            || rope_dimension_count != 128
        {
            bail!(
                "unsupported MiniCPM4 BaseLM geometry: embd={embedding_length}, layers={block_count}, ffn={feed_forward_length}, heads={head_count}, kv_heads={head_count_kv}, head_dim={head_dim}, rope_dim={rope_dimension_count}; VoxGen iteration 7 is specialized for the VoxCPM2 BaseLM geometry"
            );
        }
        if head_count % head_count_kv != 0 {
            bail!("head_count {head_count} is not divisible by head_count_kv {head_count_kv}");
        }
        if rope_dimension_count != head_dim {
            bail!("partial RoPE is not implemented: rope_dimension_count={rope_dimension_count}, head_dim={head_dim}");
        }
        if rope_scaling_type != "longrope" && rope_scaling_type != "none" {
            bail!("unsupported RoPE scaling type {rope_scaling_type:?}; expected longrope");
        }

        let token_embd = summary.tensor("token_embd.weight")?;
        require_dims(token_embd, &[embedding_length as u64, infer_vocab(token_embd)? as u64])?;
        require_matrix_type(token_embd)?;
        let vocab_size = infer_vocab(token_embd)?;
        let output = summary.tensors.iter().find(|t| t.name == "output.weight");
        let (output_tensor, tied_output) = match output {
            Some(t) => {
                require_dims(t, &[embedding_length as u64, vocab_size as u64])?;
                require_matrix_type(t)?;
                (t.name.clone(), false)
            }
            None => ("token_embd.weight".to_owned(), true),
        };
        require_dims(summary.tensor("output_norm.weight")?, &[embedding_length as u64])?;

        if rope_scaling_type == "longrope" {
            require_dims(
                summary.tensor("rope_factors_short.weight")?,
                &[(head_dim / 2) as u64],
            )?;
            require_dims(
                summary.tensor("rope_factors_long.weight")?,
                &[(head_dim / 2) as u64],
            )?;
            require_f32(summary.tensor("rope_factors_short.weight")?)?;
            require_f32(summary.tensor("rope_factors_long.weight")?)?;
        }

        for layer in 0..block_count {
            validate_layer(summary, layer, embedding_length, feed_forward_length, head_count_kv * head_dim)?;
        }
        require_f32(summary.tensor("output_norm.weight")?)?;

        Ok(Self {
            architecture,
            context_length,
            active_context_length,
            embedding_length,
            block_count,
            feed_forward_length,
            vocab_size,
            head_count,
            head_count_kv,
            head_dim,
            kv_dim: head_count_kv * head_dim,
            rms_epsilon,
            rope_theta,
            rope_dimension_count,
            rope_original_context_length,
            rope_scaling_type,
            embedding_scale,
            residual_scale,
            logit_scale,
            tied_output,
            output_tensor,
        })
    }

    pub fn kv_cache_bytes(&self) -> u64 {
        self.block_count as u64
            * self.active_context_length as u64
            * self.kv_dim as u64
            * 2 // fp16
            * 2 // K + V
    }

    pub fn per_cache_bytes(&self) -> u64 {
        self.kv_cache_bytes() / 2
    }

    pub fn rope_scaling_factor(&self) -> f32 {
        if self.rope_scaling_type != "longrope"
            || self.context_length <= self.rope_original_context_length
            || self.rope_original_context_length <= 1
        {
            1.0
        } else {
            let max = self.context_length as f32;
            let original = self.rope_original_context_length as f32;
            (1.0 + (max / original).ln() / original.ln()).sqrt()
        }
    }
}

fn infer_vocab(t: &TensorInfo) -> Result<u32> {
    if t.dims.len() != 2 {
        bail!("{} must be rank 2, got {:?}", t.name, t.dims);
    }
    u32::try_from(t.dims[1]).context("vocabulary size exceeds u32")
}

fn validate_layer(
    s: &GgufSummary,
    i: u32,
    embd: u32,
    ffn: u32,
    kv_dim: u32,
) -> Result<()> {
    let one = |suffix: &str| format!("blk.{i}.{suffix}");
    let attn_norm_name = one("attn_norm.weight");
    let ffn_norm_name = one("ffn_norm.weight");
    let attn_norm = s.tensor(&attn_norm_name)?;
    let ffn_norm = s.tensor(&ffn_norm_name)?;
    require_dims(attn_norm, &[embd as u64])?;
    require_dims(ffn_norm, &[embd as u64])?;
    require_f32(attn_norm)?;
    require_f32(ffn_norm)?;

    for (suffix, dims) in [
        ("attn_q.weight", vec![embd as u64, embd as u64]),
        ("attn_k.weight", vec![embd as u64, kv_dim as u64]),
        ("attn_v.weight", vec![embd as u64, kv_dim as u64]),
        ("attn_output.weight", vec![embd as u64, embd as u64]),
        ("ffn_gate.weight", vec![embd as u64, ffn as u64]),
        ("ffn_up.weight", vec![embd as u64, ffn as u64]),
        ("ffn_down.weight", vec![ffn as u64, embd as u64]),
    ] {
        let name = one(suffix);
        let t = s.tensor(&name)?;
        require_dims(t, &dims)?;
        require_matrix_type(t)?;
    }
    Ok(())
}

fn require_dims(t: &TensorInfo, dims: &[u64]) -> Result<()> {
    if t.dims != dims {
        bail!("tensor {} has dimensions {:?}, expected {:?}", t.name, t.dims, dims);
    }
    Ok(())
}
fn require_f32(t: &TensorInfo) -> Result<()> {
    if t.ggml_type != GgmlType::F32 {
        bail!("tensor {} is {}, expected F32", t.name, t.ggml_type.name());
    }
    Ok(())
}
fn require_matrix_type(t: &TensorInfo) -> Result<()> {
    if !matches!(t.ggml_type, GgmlType::F16 | GgmlType::Q8_0) {
        bail!("matrix {} uses unsupported {}", t.name, t.ggml_type.name());
    }
    if t.ggml_type == GgmlType::Q8_0 && t.dims[0] % 32 != 0 {
        bail!("Q8_0 matrix {} has row width {}, not divisible by 32", t.name, t.dims[0]);
    }
    Ok(())
}

fn meta_u32(s: &GgufSummary, keys: &[&str]) -> Result<u32> {
    meta_u32_optional(s, keys).with_context(|| format!("missing required metadata: {}", keys.join(" or ")))
}
fn meta_u32_optional(s: &GgufSummary, keys: &[&str]) -> Option<u32> {
    keys.iter().find_map(|k| s.metadata.get(*k)?.parse::<u32>().ok())
}
fn meta_f32(s: &GgufSummary, keys: &[&str]) -> Result<f32> {
    meta_f32_optional(s, keys).with_context(|| format!("missing required metadata: {}", keys.join(" or ")))
}
fn meta_f32_optional(s: &GgufSummary, keys: &[&str]) -> Option<f32> {
    keys.iter().find_map(|k| s.metadata.get(*k)?.parse::<f32>().ok())
}
fn meta_string(s: &GgufSummary, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| s.metadata.get(*k).cloned())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmbeddingPush {
    weight_offset: u32,
    cols: u32,
    token_id: u32,
    dtype: u32,
    alpha: f32,
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
struct RopePush {
    factor_offset: u32,
    position: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rope_theta: f32,
    scaling_factor: f32,
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
    embedding: ComputePipeline,
    rms_attn: ComputePipeline,
    residual_rms: ComputePipeline,
    qkv: ComputePipeline,
    attn_out: ComputePipeline,
    swiglu: ComputePipeline,
    down: ComputePipeline,
    logits: ComputePipeline,
    rope: ComputePipeline,
    store_kv: ComputePipeline,
    attn_scores: ComputePipeline,
    softmax: ComputePipeline,
    attn_values: ComputePipeline,
}

#[derive(Debug, Clone, Serialize)]
pub struct TopLogit {
    pub token_id: u32,
    pub logit: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct BaseLmStepResult {
    pub position: u32,
    pub elapsed_ms: f64,
    pub final_hidden_checksum: f64,
    pub final_hidden_l2: f64,
    pub top_logits: Vec<TopLogit>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BaseLmBenchmark {
    pub iterations: u32,
    pub token_id: u32,
    pub mean_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BaseLmGpuStep {
    pub position: u32,
    pub elapsed_ms: f64,
}

pub struct BaseLmEngine {
    pub config: BaseLmConfig,
    pub format: BaseFormat,
    model_data: GpuBuffer,
    hidden: GpuBuffer,
    norm: GpuBuffer,
    q: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    attention: GpuBuffer,
    gate: GpuBuffer,
    branch: GpuBuffer,
    logits: GpuBuffer,
    scores: GpuBuffer,
    kv_k: GpuBuffer,
    kv_v: GpuBuffer,
    pipelines: Pipelines,
    command_buffer: vk::CommandBuffer,
    position: u32,
}

impl BaseLmEngine {
    pub fn new(
        gpu: &VulkanContext,
        summary: &GgufSummary,
        format: BaseFormat,
        max_context: u32,
    ) -> Result<Self> {
        let config = BaseLmConfig::from_gguf(summary, max_context)?;
        validate_format_consistency(summary, format)?;

        let data_bytes = summary.data_bytes()?;
        if data_bytes > gpu.info.max_storage_buffer_range {
            bail!(
                "BaseLM GGUF data region is {:.2} GiB, larger than the selected Vulkan device maxStorageBufferRange {:.2} GiB. VoxGen refuses to segment/fallback silently.",
                data_bytes as f64 / 1073741824.0,
                gpu.info.max_storage_buffer_range as f64 / 1073741824.0
            );
        }
        if data_bytes > u32::MAX as u64 {
            bail!("BaseLM data region exceeds 4 GiB; iteration-2 shaders use 32-bit byte offsets");
        }

        let file = File::open(&summary.path)
            .with_context(|| format!("open BaseLM {}", summary.path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap BaseLM {}", summary.path.display()))?;
        let start = usize::try_from(summary.data_offset).context("GGUF data offset exceeds usize")?;
        let model_data = gpu.upload_device_local(&mmap[start..])?;

        let storage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let f32_bytes = |n: u32| n as u64 * 4;
        let hidden = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let norm = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let q = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let k = gpu.create_buffer(f32_bytes(config.kv_dim), storage, local)?;
        let v = gpu.create_buffer(f32_bytes(config.kv_dim), storage, local)?;
        let attention = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let gate = gpu.create_buffer(f32_bytes(config.feed_forward_length), storage, local)?;
        let branch = gpu.create_buffer(f32_bytes(config.embedding_length), storage, local)?;
        let logits = gpu.create_buffer(f32_bytes(config.vocab_size), storage, local)?;
        let scores = gpu.create_buffer(
            config.head_count as u64 * config.active_context_length as u64 * 4,
            storage,
            local,
        )?;
        if config.per_cache_bytes() > gpu.info.max_storage_buffer_range {
            bail!(
                "one BaseLM KV cache buffer requires {:.2} MiB, exceeding maxStorageBufferRange {:.2} MiB; reduce --max-context",
                config.per_cache_bytes() as f64 / 1048576.0,
                gpu.info.max_storage_buffer_range as f64 / 1048576.0
            );
        }
        let kv_k = gpu.create_buffer(config.per_cache_bytes(), storage, local)?;
        let kv_v = gpu.create_buffer(config.per_cache_bytes(), storage, local)?;

        let embedding = gpu.create_compute_pipeline(
            EMBEDDING_SPV,
            2,
            std::mem::size_of::<EmbeddingPush>() as u32,
        )?;
        embedding.bind_buffers(&[&model_data, &hidden]);

        let rms_attn = gpu.create_compute_pipeline(gpu.select_spirv(RMSNORM_SPV, RMSNORM_XTX7900_SPV), 3, std::mem::size_of::<RmsPush>() as u32)?;
        rms_attn.bind_buffers(&[&model_data, &hidden, &norm]);
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
        let logits_pipe = matvec_pipeline(gpu, &model_data, &norm, &logits)?;

        let rope = gpu.create_compute_pipeline(ROPE_SPV, 3, std::mem::size_of::<RopePush>() as u32)?;
        rope.bind_buffers(&[&model_data, &q, &k]);
        let store_kv = gpu.create_compute_pipeline(STORE_KV_SPV, 4, std::mem::size_of::<StoreKvPush>() as u32)?;
        store_kv.bind_buffers(&[&k, &v, &kv_k, &kv_v]);
        let attn_scores = gpu.create_compute_pipeline(gpu.select_spirv(ATTN_SCORES_SPV, ATTN_SCORES_XTX7900_SPV), 3, std::mem::size_of::<AttentionPush>() as u32)?;
        attn_scores.bind_buffers(&[&q, &kv_k, &scores]);
        let softmax = gpu.create_compute_pipeline(gpu.select_spirv(SOFTMAX_SPV, SOFTMAX_XTX7900_SPV), 1, std::mem::size_of::<SoftmaxPush>() as u32)?;
        softmax.bind_buffers(&[&scores]);
        let attn_values = gpu.create_compute_pipeline(ATTN_VALUES_SPV, 3, std::mem::size_of::<AttentionPush>() as u32)?;
        attn_values.bind_buffers(&[&scores, &kv_v, &attention]);

        let command_buffer = gpu.allocate_primary_command_buffer()?;
        Ok(Self {
            config,
            format,
            model_data,
            hidden,
            norm,
            q,
            k,
            v,
            attention,
            gate,
            branch,
            logits,
            scores,
            kv_k,
            kv_v,
            pipelines: Pipelines {
                embedding,
                rms_attn,
                residual_rms,
                qkv,
                attn_out,
                swiglu,
                down: down_pipe,
                logits: logits_pipe,
                rope,
                store_kv,
                attn_scores,
                softmax,
                attn_values,
            },
            command_buffer,
            position: 0,
        })
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.model_data.size
            + self.hidden.size
            + self.norm.size
            + self.q.size
            + self.k.size
            + self.v.size
            + self.attention.size
            + self.gate.size
            + self.branch.size
            + self.logits.size
            + self.scores.size
            + self.kv_k.size
            + self.kv_v.size
    }

    pub fn position(&self) -> u32 {
        self.position
    }

    /// Final normalized MiniCPM4 hidden state for the most recent step.
    /// Later VoxCPM2 stages bind this GPU buffer directly; no CPU inference fallback is involved.
    pub fn output_buffer(&self) -> &GpuBuffer {
        &self.norm
    }

    pub fn model_data_buffer(&self) -> &GpuBuffer { &self.model_data }

    pub fn reset(&mut self) {
        // Old packed cache contents are intentionally left in VRAM. They are unreachable
        // once position returns to zero and will be overwritten as decoding advances.
        self.position = 0;
    }

    pub fn decode_token(
        &mut self,
        gpu: &VulkanContext,
        summary: &GgufSummary,
        token_id: u32,
        top_k: usize,
    ) -> Result<BaseLmStepResult> {
        if token_id >= self.config.vocab_size {
            bail!("token id {token_id} is outside vocabulary 0..{}", self.config.vocab_size);
        }
        self.ensure_context()?;
        gpu.begin_one_time(self.command_buffer)?;
        let embedding = summary.tensor("token_embd.weight")?;
        self.pipelines.embedding.bind(self.command_buffer);
        self.pipelines.embedding.push(
            self.command_buffer,
            &EmbeddingPush {
                weight_offset: tensor_offset_u32(embedding)?,
                cols: self.config.embedding_length,
                token_id,
                dtype: dtype_code(embedding.ggml_type)?,
                alpha: effective_scale(self.config.embedding_scale),
            },
        );
        unsafe { gpu.device.cmd_dispatch(self.command_buffer, 1, 1, 1) };
        gpu.compute_barrier(self.command_buffer);
        self.record_transformer(gpu, summary, self.position, true)?;
        let position = self.position;
        let started = Instant::now();
        gpu.submit_and_wait(self.command_buffer)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.position += 1;
        self.collect_result(gpu, position, elapsed_ms, top_k)
    }

    pub fn decode_embedding(
        &mut self,
        gpu: &VulkanContext,
        summary: &GgufSummary,
        embedding: &[f32],
        top_k: usize,
    ) -> Result<BaseLmStepResult> {
        if embedding.len() != self.config.embedding_length as usize {
            bail!(
                "external embedding has {} floats; expected {}",
                embedding.len(),
                self.config.embedding_length
            );
        }
        self.ensure_context()?;
        gpu.upload_f32(&self.hidden, embedding)?;
        gpu.begin_one_time(self.command_buffer)?;
        self.record_transformer(gpu, summary, self.position, true)?;
        let position = self.position;
        let started = Instant::now();
        gpu.submit_and_wait(self.command_buffer)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.position += 1;
        self.collect_result(gpu, position, elapsed_ms, top_k)
    }

    /// Run one BaseLM step and leave the normalized hidden state resident in `output_buffer()`.
    /// This skips LM-head logits and all hidden/logit readback; it is the handoff used by
    /// ResidualLM/FSQ and later acoustic stages.
    /// Run a token-embedding BaseLM step without the LM head/readback and leave the
    /// normalized hidden state resident in `output_buffer()`. This is used by the
    /// VoxCPM2 text-prefix ResidualLM prefill path, where FSQ must *not* be applied.
    pub fn decode_token_gpu_only(
        &mut self,
        gpu: &VulkanContext,
        summary: &GgufSummary,
        token_id: u32,
    ) -> Result<BaseLmGpuStep> {
        if token_id >= self.config.vocab_size {
            bail!("token id {token_id} is outside vocabulary 0..{}", self.config.vocab_size);
        }
        self.ensure_context()?;
        gpu.begin_one_time(self.command_buffer)?;
        let embedding = summary.tensor("token_embd.weight")?;
        self.pipelines.embedding.bind(self.command_buffer);
        self.pipelines.embedding.push(
            self.command_buffer,
            &EmbeddingPush {
                weight_offset: tensor_offset_u32(embedding)?,
                cols: self.config.embedding_length,
                token_id,
                dtype: dtype_code(embedding.ggml_type)?,
                alpha: effective_scale(self.config.embedding_scale),
            },
        );
        unsafe { gpu.device.cmd_dispatch(self.command_buffer, 1, 1, 1) };
        gpu.compute_barrier(self.command_buffer);
        self.record_transformer(gpu, summary, self.position, false)?;
        let position = self.position;
        let started = Instant::now();
        gpu.submit_and_wait(self.command_buffer)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.position += 1;
        Ok(BaseLmGpuStep { position, elapsed_ms })
    }

    /// Record one token-only BaseLM prefill step into an already-begun command buffer.
    /// Used by the RX 7900 XTX prefill batcher so several BaseLM + ResidualLM
    /// positions can share one Vulkan submission. The caller owns submission/wait.
    pub(crate) fn record_token_gpu_only_in(
        &mut self,
        gpu: &VulkanContext,
        summary: &GgufSummary,
        token_id: u32,
        cmd: vk::CommandBuffer,
    ) -> Result<u32> {
        if token_id >= self.config.vocab_size {
            bail!("token id {token_id} is outside vocabulary 0..{}", self.config.vocab_size);
        }
        self.ensure_context()?;
        let position = self.position;
        let old_cmd = self.command_buffer;
        self.command_buffer = cmd;
        let recorded = (|| -> Result<()> {
            let embedding = summary.tensor("token_embd.weight")?;
            self.pipelines.embedding.bind(cmd);
            self.pipelines.embedding.push(
                cmd,
                &EmbeddingPush {
                    weight_offset: tensor_offset_u32(embedding)?,
                    cols: self.config.embedding_length,
                    token_id,
                    dtype: dtype_code(embedding.ggml_type)?,
                    alpha: effective_scale(self.config.embedding_scale),
                },
            );
            unsafe { gpu.device.cmd_dispatch(cmd, 1, 1, 1) };
            gpu.compute_barrier(cmd);
            self.record_transformer(gpu, summary, position, false)?;
            Ok(())
        })();
        self.command_buffer = old_cmd;
        recorded?;
        self.position += 1;
        Ok(position)
    }

    pub(crate) fn prefill_command_buffer(&self) -> vk::CommandBuffer { self.command_buffer }

    pub fn decode_embedding_gpu_only(
        &mut self,
        gpu: &VulkanContext,
        summary: &GgufSummary,
        embedding: &[f32],
    ) -> Result<BaseLmGpuStep> {
        if embedding.len() != self.config.embedding_length as usize {
            bail!(
                "external embedding has {} floats; expected {}",
                embedding.len(),
                self.config.embedding_length
            );
        }
        self.ensure_context()?;
        gpu.upload_f32(&self.hidden, embedding)?;
        gpu.begin_one_time(self.command_buffer)?;
        self.record_transformer(gpu, summary, self.position, false)?;
        let position = self.position;
        let started = Instant::now();
        gpu.submit_and_wait(self.command_buffer)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.position += 1;
        Ok(BaseLmGpuStep { position, elapsed_ms })
    }

    /// Consume a 2048-float embedding already resident on the selected Vulkan device.
    /// Used by LocEnc during audio-prefix conditioning; no CPU readback/upload is involved.
    pub fn decode_embedding_from_gpu_only(
        &mut self,
        gpu: &VulkanContext,
        summary: &GgufSummary,
        source: &GpuBuffer,
    ) -> Result<BaseLmGpuStep> {
        if source.size < self.config.embedding_length as u64 * 4 {
            bail!("GPU embedding source is too small for BaseLM 2048-float input");
        }
        self.ensure_context()?;
        gpu.begin_one_time(self.command_buffer)?;
        gpu.compute_to_transfer_barrier(self.command_buffer);
        unsafe {
            gpu.device.cmd_copy_buffer(
                self.command_buffer,
                source.buffer,
                self.hidden.buffer,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: self.config.embedding_length as u64 * 4 }],
            );
        }
        gpu.transfer_to_compute_barrier(self.command_buffer);
        self.record_transformer(gpu, summary, self.position, false)?;
        let position = self.position;
        let started = Instant::now();
        gpu.submit_and_wait(self.command_buffer)?;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.position += 1;
        Ok(BaseLmGpuStep { position, elapsed_ms })
    }

    pub fn prefill_tokens(
        &mut self,
        gpu: &VulkanContext,
        summary: &GgufSummary,
        tokens: &[u32],
        top_k_last: usize,
    ) -> Result<BaseLmStepResult> {
        if tokens.is_empty() {
            bail!("prefill token list is empty");
        }
        let mut last = None;
        for (i, &token) in tokens.iter().enumerate() {
            let top_k = if i + 1 == tokens.len() { top_k_last } else { 0 };
            last = Some(self.decode_token(gpu, summary, token, top_k)?);
        }
        Ok(last.unwrap())
    }

    pub fn benchmark(
        &mut self,
        gpu: &VulkanContext,
        summary: &GgufSummary,
        token_id: u32,
        warmup: u32,
        iterations: u32,
    ) -> Result<BaseLmBenchmark> {
        if iterations == 0 {
            bail!("benchmark iterations must be > 0");
        }
        let needed = warmup.saturating_add(iterations);
        if needed > self.config.active_context_length {
            bail!(
                "benchmark needs {needed} context positions but --max-context is {}",
                self.config.active_context_length
            );
        }
        self.reset();
        for _ in 0..warmup {
            self.decode_token(gpu, summary, token_id, 0)?;
        }
        let mut values = Vec::with_capacity(iterations as usize);
        for _ in 0..iterations {
            let r = self.decode_token(gpu, summary, token_id, 0)?;
            values.push(r.elapsed_ms);
        }
        let sum: f64 = values.iter().sum();
        let mean_ms = sum / values.len() as f64;
        let min_ms = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max_ms = values.iter().copied().fold(0.0, f64::max);
        Ok(BaseLmBenchmark {
            iterations,
            token_id,
            mean_ms,
            min_ms,
            max_ms,
            tokens_per_second: 1000.0 / mean_ms,
        })
    }

    fn ensure_context(&self) -> Result<()> {
        if self.position >= self.config.active_context_length {
            bail!(
                "BaseLM KV cache is full at {} tokens; reset or increase --max-context",
                self.config.active_context_length
            );
        }
        Ok(())
    }

    fn record_transformer(
        &self,
        gpu: &VulkanContext,
        summary: &GgufSummary,
        position: u32,
        compute_logits: bool,
    ) -> Result<()> {
        let cmd = self.command_buffer;
        let residual_scale = effective_scale(self.config.residual_scale);
        let q_per_kv = self.config.head_count / self.config.head_count_kv;
        let rope_factor_name = if self.config.rope_scaling_type == "longrope" {
            // MiniCPMLongRoPE builds one cache for max_position_embeddings and chooses
            // the factor set from that configured sequence length, not per token.
            if self.config.context_length > self.config.rope_original_context_length {
                "rope_factors_long.weight"
            } else {
                "rope_factors_short.weight"
            }
        } else {
            bail!("MiniCPM4 BaseLM without LongRoPE factors is not enabled in iteration 7");
        };
        let rope_factor = summary.tensor(rope_factor_name)?;

        self.dispatch_rms(
            gpu,
            &self.pipelines.rms_attn,
            summary.tensor("blk.0.attn_norm.weight")?,
        )?;
        gpu.compute_buffer_barrier(cmd, &[&self.norm]);

        for layer in 0..self.config.block_count {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}");

            self.dispatch_qkv(
                gpu,
                summary.tensor(&name("attn_q.weight"))?,
                summary.tensor(&name("attn_k.weight"))?,
                summary.tensor(&name("attn_v.weight"))?,
            )?;
            gpu.compute_buffer_barrier(cmd, &[&self.q, &self.k]);

            self.pipelines.rope.bind(cmd);
            self.pipelines.rope.push(
                cmd,
                &RopePush {
                    factor_offset: tensor_offset_u32(rope_factor)?,
                    position,
                    q_heads: self.config.head_count,
                    kv_heads: self.config.head_count_kv,
                    head_dim: self.config.head_dim,
                    rope_theta: self.config.rope_theta,
                    scaling_factor: self.config.rope_scaling_factor(),
                },
            );
            unsafe {
                gpu.device.cmd_dispatch(
                    cmd,
                    self.config.head_count + self.config.head_count_kv,
                    1,
                    1,
                )
            };
            gpu.compute_buffer_barrier(cmd, &[&self.q, &self.k, &self.v]);

            self.pipelines.store_kv.bind(cmd);
            self.pipelines.store_kv.push(
                cmd,
                &StoreKvPush {
                    layer,
                    position,
                    context: self.config.active_context_length,
                    kv_dim: self.config.kv_dim,
                },
            );
            unsafe { gpu.device.cmd_dispatch(cmd, 1, 1, 1) };
            gpu.compute_buffer_barrier(cmd, &[&self.kv_k, &self.kv_v]);

            let attn_push = AttentionPush {
                layer,
                context: self.config.active_context_length,
                position,
                head_dim: self.config.head_dim,
                q_per_kv,
                kv_heads: self.config.head_count_kv,
            };
            self.pipelines.attn_scores.bind(cmd);
            self.pipelines.attn_scores.push(cmd, &attn_push);
            unsafe { gpu.device.cmd_dispatch(cmd, position + 1, self.config.head_count, 1) };
            gpu.compute_buffer_barrier(cmd, &[&self.scores]);

            self.pipelines.softmax.bind(cmd);
            self.pipelines.softmax.push(
                cmd,
                &SoftmaxPush {
                    context: self.config.active_context_length,
                    position,
                },
            );
            unsafe { gpu.device.cmd_dispatch(cmd, self.config.head_count, 1, 1) };
            gpu.compute_buffer_barrier(cmd, &[&self.scores]);

            self.pipelines.attn_values.bind(cmd);
            self.pipelines.attn_values.push(cmd, &attn_push);
            unsafe { gpu.device.cmd_dispatch(cmd, self.config.head_count, 1, 1) };
            gpu.compute_buffer_barrier(cmd, &[&self.attention]);

            self.dispatch_matvec(
                gpu,
                &self.pipelines.attn_out,
                summary.tensor(&name("attn_output.weight"))?,
                self.config.embedding_length,
                self.config.embedding_length,
                1.0,
            )?;
            gpu.compute_buffer_barrier(cmd, &[&self.branch]);
            self.dispatch_residual_rms(
                gpu,
                summary.tensor(&name("ffn_norm.weight"))?,
                residual_scale,
            )?;
            gpu.compute_buffer_barrier(cmd, &[&self.norm]);

            self.dispatch_swiglu(
                gpu,
                summary.tensor(&name("ffn_gate.weight"))?,
                summary.tensor(&name("ffn_up.weight"))?,
                self.config.feed_forward_length,
                self.config.embedding_length,
            )?;
            gpu.compute_buffer_barrier(cmd, &[&self.gate]);
            self.dispatch_matvec(
                gpu,
                &self.pipelines.down,
                summary.tensor(&name("ffn_down.weight"))?,
                self.config.embedding_length,
                self.config.feed_forward_length,
                1.0,
            )?;
            gpu.compute_buffer_barrier(cmd, &[&self.branch]);

            let next_norm = if layer + 1 < self.config.block_count {
                summary.tensor(&format!("blk.{}.attn_norm.weight", layer + 1))?
            } else {
                summary.tensor("output_norm.weight")?
            };
            self.dispatch_residual_rms(gpu, next_norm, residual_scale)?;
            gpu.compute_buffer_barrier(cmd, &[&self.norm]);
        }
        if compute_logits {
            self.dispatch_matvec(
                gpu,
                &self.pipelines.logits,
                summary.tensor(&self.config.output_tensor)?,
                self.config.vocab_size,
                self.config.embedding_length,
                effective_scale(self.config.logit_scale),
            )?;
            gpu.compute_barrier(cmd);
        }
        Ok(())
    }

    fn dispatch_rms(
        &self,
        gpu: &VulkanContext,
        pipeline: &ComputePipeline,
        weight: &TensorInfo,
    ) -> Result<()> {
        let span = gpu.gpu_profile_begin(self.command_buffer, "baselm.rmsnorm");
        pipeline.bind(self.command_buffer);
        pipeline.push(
            self.command_buffer,
            &RmsPush {
                weight_offset: tensor_offset_u32(weight)?,
                n: self.config.embedding_length,
                eps: self.config.rms_epsilon,
            },
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
            bail!("QKV projection shape mismatch");
        }
        let total_rows = self.config.embedding_length + 2 * self.config.kv_dim;
        let max_x = gpu.info.max_compute_work_group_count_x.max(1);
        let span = gpu.gpu_profile_begin(self.command_buffer, "baselm.qkv");
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
                q_dtype: dtype_code(q_weight.ggml_type)?,
                k_dtype: dtype_code(k_weight.ggml_type)?,
                v_dtype: dtype_code(v_weight.ggml_type)?,
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
        alpha: f32,
    ) -> Result<()> {
        if weight.dims != [cols as u64, rows as u64] {
            bail!(
                "matvec shape mismatch for {}: GGUF {:?}, requested [{cols}, {rows}]",
                weight.name,
                weight.dims
            );
        }
        let span = gpu.gpu_profile_begin(self.command_buffer, "baselm.matvec");
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
                    dtype: dtype_code(weight.ggml_type)?,
                    row_base,
                    alpha,
                },
            );
            unsafe { gpu.device.cmd_dispatch(self.command_buffer, groups, 1, 1) };
            row_base += groups;
        }
        gpu.gpu_profile_end(self.command_buffer, span);
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
            bail!("SwiGLU projection shape mismatch");
        }
        if gate_weight.ggml_type != up_weight.ggml_type {
            bail!("SwiGLU gate/up dtypes differ: {} vs {}", gate_weight.ggml_type.name(), up_weight.ggml_type.name());
        }
        let span = gpu.gpu_profile_begin(self.command_buffer, "baselm.swiglu");
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
                dtype: dtype_code(gate_weight.ggml_type)?,
                row_base,
            });
            unsafe { gpu.device.cmd_dispatch(self.command_buffer, groups, 1, 1) };
            row_base += groups;
        }
        gpu.gpu_profile_end(self.command_buffer, span);
        Ok(())
    }

    fn dispatch_residual_rms(&self, gpu: &VulkanContext, weight: &TensorInfo, scale: f32) -> Result<()> {
        let span = gpu.gpu_profile_begin(self.command_buffer, "baselm.residual_rmsnorm");
        self.pipelines.residual_rms.bind(self.command_buffer);
        self.pipelines.residual_rms.push(self.command_buffer, &ResidualRmsPush {
            weight_offset: tensor_offset_u32(weight)?,
            n: self.config.embedding_length,
            eps: self.config.rms_epsilon,
            scale,
        });
        unsafe { gpu.device.cmd_dispatch(self.command_buffer, 1, 1, 1) };
        gpu.gpu_profile_end(self.command_buffer, span);
        Ok(())
    }

    fn collect_result(
        &self,
        gpu: &VulkanContext,
        position: u32,
        elapsed_ms: f64,
        top_k: usize,
    ) -> Result<BaseLmStepResult> {
        let hidden = gpu.read_f32(&self.norm, self.config.embedding_length as usize)?;
        let checksum = hidden.iter().map(|&x| x as f64).sum::<f64>();
        let l2 = hidden
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt();
        let top_logits = if top_k == 0 {
            Vec::new()
        } else {
            let logits = gpu.read_f32(&self.logits, self.config.vocab_size as usize)?;
            top_k_logits(&logits, top_k)
        };
        Ok(BaseLmStepResult {
            position,
            elapsed_ms,
            final_hidden_checksum: checksum,
            final_hidden_l2: l2,
            top_logits,
        })
    }
}

fn matvec_pipeline(
    gpu: &VulkanContext,
    model: &GpuBuffer,
    input: &GpuBuffer,
    output: &GpuBuffer,
) -> Result<ComputePipeline> {
    let p = gpu.create_compute_pipeline(
        gpu.select_spirv(MATVEC_SPV, MATVEC_XTX7900_SPV),
        3,
        std::mem::size_of::<MatvecPush>() as u32,
    )?;
    p.bind_buffers(&[model, input, output]);
    Ok(p)
}

fn dtype_code(t: GgmlType) -> Result<u32> {
    match t {
        GgmlType::F16 => Ok(1),
        GgmlType::Q8_0 => Ok(8),
        other => bail!("unsupported BaseLM matrix type {}", other.name()),
    }
}

fn tensor_offset_u32(t: &TensorInfo) -> Result<u32> {
    u32::try_from(t.offset)
        .with_context(|| format!("tensor {} offset {} exceeds 32-bit shader address space", t.name, t.offset))
}

fn effective_scale(v: f32) -> f32 {
    if v == 0.0 { 1.0 } else { v }
}

fn validate_format_consistency(summary: &GgufSummary, format: BaseFormat) -> Result<()> {
    let detected = summary.primary_base_format()?;
    if detected != format {
        bail!(
            "BaseLM format changed between validation and initialization: expected {}, detected {}",
            format.as_str(),
            detected.as_str()
        );
    }
    for t in &summary.tensors {
        if t.dims.len() < 2 || t.elements < 4096 {
            continue;
        }
        if t.name.contains("rope_factors") {
            continue;
        }
        match format {
            BaseFormat::F16 if t.ggml_type != GgmlType::F16 => {
                bail!("F16 BaseLM has non-F16 matrix {} ({})", t.name, t.ggml_type.name())
            }
            BaseFormat::Q8_0 if !matches!(t.ggml_type, GgmlType::Q8_0 | GgmlType::F16) => {
                bail!("Q8_0 BaseLM has unsupported matrix {} ({})", t.name, t.ggml_type.name())
            }
            _ => {}
        }
    }
    Ok(())
}

fn top_k_logits(logits: &[f32], k: usize) -> Vec<TopLogit> {
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, v)| (i as u32, v))
        .collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    indexed
        .into_iter()
        .take(k.min(logits.len()))
        .map(|(token_id, logit)| TopLogit { token_id, logit })
        .collect()
}

fn div_ceil(n: u32, d: u32) -> u32 {
    (n + d - 1) / d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_zero_scale_means_identity() {
        assert_eq!(effective_scale(0.0), 1.0);
        assert_eq!(effective_scale(0.125), 0.125);
    }

    #[test]
    fn top_k_is_descending() {
        let got = top_k_logits(&[1.0, -2.0, 5.0, 3.0], 3);
        assert_eq!(got.iter().map(|x| x.token_id).collect::<Vec<_>>(), vec![2, 3, 0]);
    }

    #[test]
    fn kv_bytes_match_packed_f16_layout() {
        let c = BaseLmConfig {
            architecture: "minicpm4".into(),
            context_length: 32768,
            active_context_length: 32768,
            embedding_length: 2048,
            block_count: 28,
            feed_forward_length: 6144,
            vocab_size: 73448,
            head_count: 16,
            head_count_kv: 2,
            head_dim: 128,
            kv_dim: 256,
            rms_epsilon: 1e-5,
            rope_theta: 10000.0,
            rope_dimension_count: 128,
            rope_original_context_length: 32768,
            rope_scaling_type: "longrope".into(),
            embedding_scale: 0.0,
            residual_scale: 0.0,
            logit_scale: 0.125,
            tied_output: true,
            output_tensor: "token_embd.weight".into(),
        };
        assert_eq!(c.per_cache_bytes(), 469_762_048);
        assert_eq!(c.kv_cache_bytes(), 939_524_096);
    }
}
