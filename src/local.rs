use crate::{
    baselm::BaseLmConfig,
    gguf::{GgmlType, GgufSummary, TensorInfo},
    vulkan::{ComputePipeline, GpuBuffer, VulkanContext},
};
use anyhow::{bail, Context, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use serde::Serialize;
use std::time::Instant;

const SEQ_LINEAR_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seq_linear_bias.spv"));
const SEQ_LINEAR_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seq_linear_bias_xtx7900.spv"));
const SEQ_LINEAR_COOPMAT_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seq_linear_bias_coopmat_xtx7900.spv"));
const SEQ_QKV_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seq_qkv.spv"));
const SEQ_QKV_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seq_qkv_xtx7900.spv"));
const SEQ_RMS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seq_rmsnorm.spv"));
const SEQ_RMS_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seq_rmsnorm_xtx7900.spv"));
const SEQ_RESIDUAL_RMS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seq_residual_rmsnorm.spv"));
const SEQ_RESIDUAL_RMS_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seq_residual_rmsnorm_xtx7900.spv"));
const SEQ_SWIGLU_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seq_swiglu.spv"));
const SEQ_SWIGLU_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seq_swiglu_xtx7900.spv"));
const SEQ_ROPE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seq_rope.spv"));
const DENSE_SCORES_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dense_attn_scores.spv"));
const DENSE_SCORES_XTX7900_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dense_attn_scores_xtx7900.spv"));
const DENSE_SOFTMAX_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dense_softmax.spv"));
const DENSE_VALUES_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dense_attn_values.spv"));
const SEQ_SILU_MUL_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seq_silu_mul.spv"));
const PACK_LOCENC_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pack_locenc.spv"));
const PACK_LOCDIT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pack_locdit.spv"));
const TIME_SIN_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/time_sinusoidal.spv"));
const SILU_INPLACE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/silu_inplace.spv"));
const ADD_VECTORS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_vectors.spv"));
const CFM_NOISE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cfm_noise.spv"));
const CFM_CFG_EULER_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cfm_cfg_euler.spv"));

pub const PATCH_SIZE: u32 = 4;
pub const FEAT_DIM: u32 = 64;
const LOCAL_HIDDEN: u32 = 1024;
const LOCAL_FFN: u32 = 4096;
const LOCAL_LAYERS: u32 = 12;
const LOCAL_Q_HEADS: u32 = 16;
const LOCAL_KV_HEADS: u32 = 2;
const HEAD_DIM: u32 = 128;
const Q_DIM: u32 = LOCAL_Q_HEADS * HEAD_DIM; // 2048
const KV_DIM: u32 = LOCAL_KV_HEADS * HEAD_DIM; // 256
const LOCENC_TOKENS: u32 = 1 + PATCH_SIZE; // CLS + four latent tokens
const LOCDIT_TOKENS: u32 = 2 + 1 + PATCH_SIZE + PATCH_SIZE; // mu1+mu2+time+cond+x

#[derive(Debug, Clone, Serialize)]
pub struct LocalConfig {
    pub patch_size: u32,
    pub feat_dim: u32,
    pub locenc_layers: u32,
    pub locenc_hidden: u32,
    pub locenc_tokens: u32,
    pub locdit_layers: u32,
    pub locdit_hidden: u32,
    pub locdit_tokens: u32,
    pub feed_forward_length: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub head_dim: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub rms_epsilon: f32,
    pub rope_theta: f32,
    pub cfm_sigma_min: f32,
    pub cfm_cfg_rate: f32,
}

impl LocalConfig {
    pub fn from_gguf(acoustic: &GgufSummary, base: &BaseLmConfig) -> Result<Self> {
        let patch_size = acoustic.metadata_u32("voxcpm.patch_size")?;
        let feat_dim = acoustic.metadata_u32("voxcpm.feat_dim")?;
        let locenc_layers = acoustic.metadata_u32("voxcpm.locenc.n_layer")?;
        let locenc_hidden = acoustic.metadata_u32("voxcpm.locenc.n_embd")?;
        let locdit_layers = acoustic.metadata_u32("voxcpm.locdit.n_layer")?;
        let locdit_hidden = acoustic.metadata_u32("voxcpm.locdit.n_embd")?;
        let cfm_sigma_min = acoustic.metadata_f32("voxcpm.cfm.sigma_min")?;
        let cfm_cfg_rate = acoustic.metadata_f32("voxcpm.cfm.cfg_rate")?;
        if !cfm_sigma_min.is_finite() || !cfm_cfg_rate.is_finite() {
            bail!("VoxCPM2 CFM metadata contains non-finite values: sigma_min={cfm_sigma_min}, cfg_rate={cfm_cfg_rate}");
        }
        if patch_size != PATCH_SIZE || feat_dim != FEAT_DIM || locenc_layers != LOCAL_LAYERS
            || locdit_layers != LOCAL_LAYERS || locenc_hidden != LOCAL_HIDDEN || locdit_hidden != LOCAL_HIDDEN
        {
            bail!("unsupported VoxCPM2 local contract: patch={patch_size}, feat={feat_dim}, LocEnc={locenc_layers}x{locenc_hidden}, LocDiT={locdit_layers}x{locdit_hidden}; expected 4/64/12x1024/12x1024");
        }
        if base.head_count != LOCAL_Q_HEADS || base.head_count_kv != LOCAL_KV_HEADS || base.head_dim != HEAD_DIM {
            bail!("BaseLM attention geometry is incompatible with VoxCPM2 LocEnc/LocDiT");
        }
        let c = Self {
            patch_size,
            feat_dim,
            locenc_layers,
            locenc_hidden,
            locenc_tokens: LOCENC_TOKENS,
            locdit_layers,
            locdit_hidden,
            locdit_tokens: LOCDIT_TOKENS,
            feed_forward_length: LOCAL_FFN,
            head_count: LOCAL_Q_HEADS,
            head_count_kv: LOCAL_KV_HEADS,
            head_dim: HEAD_DIM,
            q_dim: Q_DIM,
            kv_dim: KV_DIM,
            rms_epsilon: base.rms_epsilon,
            rope_theta: base.rope_theta,
            cfm_sigma_min,
            cfm_cfg_rate,
        };
        c.validate_tensors(acoustic)?;
        Ok(c)
    }

    fn validate_tensors(&self, s: &GgufSummary) -> Result<()> {
        require_matrix(s.tensor("locenc.in_proj.weight")?, FEAT_DIM, LOCAL_HIDDEN)?;
        require_vector(s.tensor("locenc.in_proj.bias")?, LOCAL_HIDDEN, false)?;
        require_vector(s.tensor("locenc.cls_token.weight")?, LOCAL_HIDDEN, false)?;
        require_vector(s.tensor("locenc.norm.weight")?, LOCAL_HIDDEN, true)?;
        require_matrix(s.tensor("projections.enc_to_lm_proj.weight")?, LOCAL_HIDDEN, 2048)?;
        require_vector(s.tensor("projections.enc_to_lm_proj.bias")?, 2048, false)?;
        validate_local_layers(s, "locenc", self.locenc_layers)?;

        for name in ["locdit.in_proj.weight", "locdit.cond_proj.weight"] {
            require_matrix(s.tensor(name)?, FEAT_DIM, LOCAL_HIDDEN)?;
        }
        for name in ["locdit.in_proj.bias", "locdit.cond_proj.bias", "locdit.out_proj.bias"] {
            let n = if name == "locdit.out_proj.bias" { FEAT_DIM } else { LOCAL_HIDDEN };
            require_vector(s.tensor(name)?, n, false)?;
        }
        require_matrix(s.tensor("locdit.out_proj.weight")?, LOCAL_HIDDEN, FEAT_DIM)?;
        require_vector(s.tensor("locdit.norm.weight")?, LOCAL_HIDDEN, true)?;
        for prefix in ["locdit.time_mlp", "locdit.delta_time_mlp"] {
            for i in [1, 2] {
                require_matrix(s.tensor(&format!("{prefix}.linear_{i}.weight"))?, LOCAL_HIDDEN, LOCAL_HIDDEN)?;
                require_vector(s.tensor(&format!("{prefix}.linear_{i}.bias"))?, LOCAL_HIDDEN, false)?;
            }
        }
        require_matrix(s.tensor("projections.lm_to_dit_proj.weight")?, 2048, LOCAL_HIDDEN)?;
        require_vector(s.tensor("projections.lm_to_dit_proj.bias")?, LOCAL_HIDDEN, false)?;
        require_matrix(s.tensor("projections.res_to_dit_proj.weight")?, 2048, LOCAL_HIDDEN)?;
        require_vector(s.tensor("projections.res_to_dit_proj.bias")?, LOCAL_HIDDEN, false)?;
        validate_local_layers(s, "locdit", self.locdit_layers)?;
        Ok(())
    }
}

fn validate_local_layers(s: &GgufSummary, prefix: &str, layers: u32) -> Result<()> {
    for i in 0..layers {
        let p = |suffix: &str| format!("{prefix}.blk.{i}.{suffix}");
        for n in [p("attn_norm.weight"), p("ffn_norm.weight")] {
            require_vector(s.tensor(&n)?, LOCAL_HIDDEN, true)?;
        }
        for (suffix, cols, rows) in [
            ("attn_q.weight", LOCAL_HIDDEN, Q_DIM),
            ("attn_k.weight", LOCAL_HIDDEN, KV_DIM),
            ("attn_v.weight", LOCAL_HIDDEN, KV_DIM),
            ("attn_output.weight", Q_DIM, LOCAL_HIDDEN),
            ("ffn_gate.weight", LOCAL_HIDDEN, LOCAL_FFN),
            ("ffn_up.weight", LOCAL_HIDDEN, LOCAL_FFN),
            ("ffn_down.weight", LOCAL_FFN, LOCAL_HIDDEN),
        ] {
            require_matrix(s.tensor(&p(suffix))?, cols, rows)?;
        }
    }
    Ok(())
}

fn require_matrix(t: &TensorInfo, cols: u32, rows: u32) -> Result<()> {
    if t.dims != [cols as u64, rows as u64] { bail!("{} has dimensions {:?}, expected [{cols}, {rows}]", t.name, t.dims); }
    if t.ggml_type != GgmlType::F16 { bail!("{} is {}, expected F16 acoustic weight", t.name, t.ggml_type.name()); }
    Ok(())
}
fn require_vector(t: &TensorInfo, n: u32, f32_required: bool) -> Result<()> {
    if t.dims != [n as u64] { bail!("{} has dimensions {:?}, expected [{n}]", t.name, t.dims); }
    if f32_required && t.ggml_type != GgmlType::F32 { bail!("{} is {}, expected F32 norm", t.name, t.ggml_type.name()); }
    if !f32_required && !matches!(t.ggml_type, GgmlType::F16 | GgmlType::F32) { bail!("{} has unsupported {}", t.name, t.ggml_type.name()); }
    Ok(())
}

#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct SeqLinearPush { weight_offset:u32,bias_offset:u32,rows:u32,cols:u32,weight_dtype:u32,bias_dtype:u32,tokens:u32,input_stride:u32,output_stride:u32,input_base:u32,output_base:u32,has_bias:u32,alpha:f32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct SeqRmsPush { weight_offset:u32,n:u32,tokens:u32,input_stride:u32,output_stride:u32,input_base:u32,output_base:u32,eps:f32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct SeqResidualRmsPush { weight_offset:u32,n:u32,tokens:u32,hidden_stride:u32,branch_stride:u32,output_stride:u32,hidden_base:u32,branch_base:u32,output_base:u32,eps:f32,scale:f32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct SeqQkvPush { q_weight_offset:u32,k_weight_offset:u32,v_weight_offset:u32,q_rows:u32,kv_rows:u32,cols:u32,tokens:u32,input_stride:u32,q_stride:u32,kv_stride:u32,input_base:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct SeqSwigluPush { gate_weight_offset:u32,up_weight_offset:u32,rows:u32,cols:u32,tokens:u32,input_stride:u32,output_stride:u32,input_base:u32,output_base:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct SeqRopePush { factor_offset:u32,tokens:u32,q_heads:u32,kv_heads:u32,head_dim:u32,q_stride:u32,k_stride:u32,rope_theta:f32,scaling_factor:f32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct DenseScoresPush { tokens:u32,q_heads:u32,kv_heads:u32,head_dim:u32,q_stride:u32,k_stride:u32,score_stride:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct DenseSoftmaxPush { tokens:u32,q_heads:u32,score_stride:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct DenseValuesPush { tokens:u32,q_heads:u32,kv_heads:u32,head_dim:u32,out_stride:u32,v_stride:u32,score_stride:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct NPush { n:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct PackLocEncPush { cls_offset:u32,cls_dtype:u32,hidden:u32,patch_tokens:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct PackLocDitPush { hidden:u32,patch_tokens:u32,zero_mu:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct TimePush { dim:u32,t:f32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct CfmNoisePush { n:u32, seed_lo:u32, seed_hi:u32, temperature:f32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct CfmEulerPush { n:u32, dt:f32, cfg:f32, use_zero_star:u32 }

struct TransformerPipes {
    rms_attn: ComputePipeline, residual_rms: ComputePipeline,
    q: ComputePipeline, k: ComputePipeline, v: ComputePipeline, qkv: ComputePipeline, attn_out: ComputePipeline,
    gate: ComputePipeline, up: ComputePipeline, swiglu: ComputePipeline, down: ComputePipeline,
    rope: ComputePipeline, scores: ComputePipeline, softmax: ComputePipeline, values: ComputePipeline,
    silu_mul: ComputePipeline,
}
struct LocalPipes {
    locenc_in: ComputePipeline, pack_locenc: ComputePipeline, enc_to_lm: ComputePipeline,
    locdit_in: ComputePipeline, locdit_cond: ComputePipeline, locdit_out: ComputePipeline,
    time1: ComputePipeline, time2: ComputePipeline, dt1: ComputePipeline, dt2: ComputePipeline,
    time_sin: ComputePipeline, dt_sin: ComputePipeline, time_silu: ComputePipeline, dt_silu: ComputePipeline, time_add: ComputePipeline,
    lm_to_dit: ComputePipeline, res_to_dit: ComputePipeline, pack_locdit: ComputePipeline,
    cfm_noise: ComputePipeline, cfm_cfg_euler: ComputePipeline,
    tr: TransformerPipes,
}

#[derive(Debug, Clone, Serialize)]
pub struct LocEncResult { pub elapsed_ms:f64, pub local_checksum:f64, pub local_l2:f64, pub lm_embedding_checksum:f64, pub lm_embedding_l2:f64 }
#[derive(Debug, Clone, Serialize)]
pub struct LocDitResult { pub elapsed_ms:f64, pub t:f32, pub dt:f32, pub output_checksum:f64, pub output_l2:f64 }

#[derive(Debug, Clone, Serialize)]
pub struct CfmOptions {
    pub n_timesteps:u32,
    pub cfg_value:f32,
    pub temperature:f32,
    pub sway_sampling_coef:f32,
    pub seed:u64,
    pub use_cfg_zero_star:bool,
}
#[derive(Debug, Clone, Serialize)]
pub struct CfmResult {
    pub elapsed_ms:f64,
    pub n_timesteps:u32,
    pub estimator_calls:u32,
    pub zero_init_steps:u32,
    pub cfg_value:f32,
    pub temperature:f32,
    pub sway_sampling_coef:f32,
    pub seed:u64,
    pub use_cfg_zero_star:bool,
    pub solver:&'static str,
    pub mean_mode:bool,
    pub output_checksum:f64,
    pub output_l2:f64,
    pub output_min:f32,
    pub output_max:f32,
    pub output_patch:Vec<f32>,
}

impl CfmOptions {
    pub fn validate(&self)->Result<()> {
        if self.n_timesteps==0 { bail!("CFM n_timesteps must be >= 1"); }
        if !self.cfg_value.is_finite() || !self.temperature.is_finite() || !self.sway_sampling_coef.is_finite() { bail!("CFM options must be finite"); }
        if self.temperature < 0.0 { bail!("CFM temperature must be >= 0"); }
        Ok(())
    }
}

pub struct LocalEngine {
    pub config: LocalConfig,
    patch_in:GpuBuffer, cond_in:GpuBuffer, patch_proj:GpuBuffer, cond_proj:GpuBuffer,
    hidden:GpuBuffer, norm:GpuBuffer, q:GpuBuffer, k:GpuBuffer, v:GpuBuffer, attention:GpuBuffer,
    gate:GpuBuffer, up:GpuBuffer, branch:GpuBuffer, scores:GpuBuffer,
    locenc_lm:GpuBuffer, mu1:GpuBuffer, mu2:GpuBuffer,
    time_sin:GpuBuffer, dt_sin:GpuBuffer, time_tmp:GpuBuffer, dt_tmp:GpuBuffer, time_vec:GpuBuffer, dt_vec:GpuBuffer,
    out_patch:GpuBuffer, cfm_positive:GpuBuffer, cfm_negative:GpuBuffer,
    pipes:LocalPipes, command_buffer:vk::CommandBuffer,
    rope_factor_offset:u32, rope_scaling_factor:f32,
    use_fused_swiglu:bool,
    use_fused_qkv:bool,
}

impl LocalEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(gpu:&VulkanContext, acoustic:&GgufSummary, base_summary:&GgufSummary, base:&BaseLmConfig,
               acoustic_model:&GpuBuffer, base_model:&GpuBuffer, base_hidden:&GpuBuffer, residual_hidden:&GpuBuffer) -> Result<Self> {
        let config=LocalConfig::from_gguf(acoustic,base)?;
        let storage=vk::BufferUsageFlags::STORAGE_BUFFER|vk::BufferUsageFlags::TRANSFER_SRC|vk::BufferUsageFlags::TRANSFER_DST;
        let mem=vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let b=|n:u32|gpu.create_buffer(n as u64*4,storage,mem);
        let patch_in=b(PATCH_SIZE*FEAT_DIM)?; let cond_in=b(PATCH_SIZE*FEAT_DIM)?;
        let patch_proj=b(PATCH_SIZE*LOCAL_HIDDEN)?; let cond_proj=b(PATCH_SIZE*LOCAL_HIDDEN)?;
        let hidden=b(LOCDIT_TOKENS*LOCAL_HIDDEN)?; let norm=b(LOCDIT_TOKENS*LOCAL_HIDDEN)?;
        let q=b(LOCDIT_TOKENS*Q_DIM)?; let k=b(LOCDIT_TOKENS*KV_DIM)?; let v=b(LOCDIT_TOKENS*KV_DIM)?;
        let attention=b(LOCDIT_TOKENS*Q_DIM)?; let gate=b(LOCDIT_TOKENS*LOCAL_FFN)?; let up=b(LOCDIT_TOKENS*LOCAL_FFN)?;
        let branch=b(LOCDIT_TOKENS*LOCAL_HIDDEN)?; let scores=b(LOCDIT_TOKENS*LOCAL_Q_HEADS*LOCDIT_TOKENS)?;
        let locenc_lm=b(2048)?; let mu1=b(LOCAL_HIDDEN)?; let mu2=b(LOCAL_HIDDEN)?;
        let time_sin=b(LOCAL_HIDDEN)?; let dt_sin=b(LOCAL_HIDDEN)?; let time_tmp=b(LOCAL_HIDDEN)?; let dt_tmp=b(LOCAL_HIDDEN)?;
        let time_vec=b(LOCAL_HIDDEN)?; let dt_vec=b(LOCAL_HIDDEN)?; let out_patch=b(PATCH_SIZE*FEAT_DIM)?; let cfm_positive=b(PATCH_SIZE*FEAT_DIM)?; let cfm_negative=b(PATCH_SIZE*FEAT_DIM)?;

        let seq = |input:&GpuBuffer, output:&GpuBuffer| -> Result<ComputePipeline> {
            if gpu.xtx_coopmat_enabled() {
                let p=gpu.create_compute_pipeline(SEQ_LINEAR_COOPMAT_XTX7900_SPV,3,std::mem::size_of::<SeqLinearPush>() as u32)?;
                p.bind_buffers(&[acoustic_model,input,output]);
                Ok(p)
            } else {
                let p=gpu.create_compute_pipeline(gpu.select_spirv(SEQ_LINEAR_SPV, SEQ_LINEAR_XTX7900_SPV),3,std::mem::size_of::<SeqLinearPush>() as u32)?;
                p.bind_buffers(&[acoustic_model,input,output]);
                Ok(p)
            }
        };
        let rms = || -> Result<ComputePipeline> { let p=gpu.create_compute_pipeline(gpu.select_spirv(SEQ_RMS_SPV, SEQ_RMS_XTX7900_SPV),3,std::mem::size_of::<SeqRmsPush>() as u32)?;p.bind_buffers(&[acoustic_model,&hidden,&norm]);Ok(p) };
        let locenc_in=seq(&patch_in,&patch_proj)?;
        let pack_locenc=gpu.create_compute_pipeline(PACK_LOCENC_SPV,3,std::mem::size_of::<PackLocEncPush>() as u32)?; pack_locenc.bind_buffers(&[acoustic_model,&patch_proj,&hidden]);
        let enc_to_lm=seq(&norm,&locenc_lm)?;
        let locdit_in=seq(&patch_in,&patch_proj)?; let locdit_cond=seq(&cond_in,&cond_proj)?; let locdit_out=seq(&norm,&out_patch)?;
        let time1=seq(&time_sin,&time_tmp)?; let time2=seq(&time_tmp,&time_vec)?; let dt1=seq(&dt_sin,&dt_tmp)?; let dt2=seq(&dt_tmp,&dt_vec)?;
        let time_sin_p=gpu.create_compute_pipeline(TIME_SIN_SPV,1,std::mem::size_of::<TimePush>() as u32)?; time_sin_p.bind_buffers(&[&time_sin]);
        let dt_sin_p=gpu.create_compute_pipeline(TIME_SIN_SPV,1,std::mem::size_of::<TimePush>() as u32)?; dt_sin_p.bind_buffers(&[&dt_sin]);
        let time_silu=gpu.create_compute_pipeline(SILU_INPLACE_SPV,1,std::mem::size_of::<NPush>() as u32)?; time_silu.bind_buffers(&[&time_tmp]);
        let dt_silu=gpu.create_compute_pipeline(SILU_INPLACE_SPV,1,std::mem::size_of::<NPush>() as u32)?; dt_silu.bind_buffers(&[&dt_tmp]);
        let time_add=gpu.create_compute_pipeline(ADD_VECTORS_SPV,2,std::mem::size_of::<NPush>() as u32)?; time_add.bind_buffers(&[&time_vec,&dt_vec]);
        let lm_to_dit=seq(base_hidden,&mu1)?; let res_to_dit=seq(residual_hidden,&mu2)?;
        let pack_locdit=gpu.create_compute_pipeline(PACK_LOCDIT_SPV,6,std::mem::size_of::<PackLocDitPush>() as u32)?; pack_locdit.bind_buffers(&[&mu1,&mu2,&time_vec,&cond_proj,&patch_proj,&hidden]);
        let cfm_noise=gpu.create_compute_pipeline(CFM_NOISE_SPV,1,std::mem::size_of::<CfmNoisePush>() as u32)?; cfm_noise.bind_buffers(&[&patch_in]);
        let cfm_cfg_euler=gpu.create_compute_pipeline(CFM_CFG_EULER_SPV,3,std::mem::size_of::<CfmEulerPush>() as u32)?; cfm_cfg_euler.bind_buffers(&[&patch_in,&cfm_positive,&cfm_negative]);

        let residual_rms={let p=gpu.create_compute_pipeline(gpu.select_spirv(SEQ_RESIDUAL_RMS_SPV,SEQ_RESIDUAL_RMS_XTX7900_SPV),4,std::mem::size_of::<SeqResidualRmsPush>() as u32)?;p.bind_buffers(&[acoustic_model,&hidden,&branch,&norm]);p};
        let swiglu={let p=gpu.create_compute_pipeline(gpu.select_spirv(SEQ_SWIGLU_SPV,SEQ_SWIGLU_XTX7900_SPV),3,std::mem::size_of::<SeqSwigluPush>() as u32)?;p.bind_buffers(&[acoustic_model,&norm,&gate]);p};
        let qkv={let p=gpu.create_compute_pipeline(gpu.select_spirv(SEQ_QKV_SPV,SEQ_QKV_XTX7900_SPV),5,std::mem::size_of::<SeqQkvPush>() as u32)?;p.bind_buffers(&[acoustic_model,&norm,&q,&k,&v]);p};
        let use_fused_swiglu=!gpu.xtx_coopmat_enabled();
        let use_fused_qkv=!gpu.xtx_coopmat_enabled();
        let tr=TransformerPipes{
            rms_attn:rms()?,residual_rms,
            q:seq(&norm,&q)?,k:seq(&norm,&k)?,v:seq(&norm,&v)?,qkv,attn_out:seq(&attention,&branch)?,
            gate:seq(&norm,&gate)?,up:seq(&norm,&up)?,swiglu,down:seq(&gate,&branch)?,
            rope:{let p=gpu.create_compute_pipeline(SEQ_ROPE_SPV,3,std::mem::size_of::<SeqRopePush>() as u32)?;p.bind_buffers(&[base_model,&q,&k]);p},
            scores:{let p=gpu.create_compute_pipeline(gpu.select_spirv(DENSE_SCORES_SPV, DENSE_SCORES_XTX7900_SPV),3,std::mem::size_of::<DenseScoresPush>() as u32)?;p.bind_buffers(&[&q,&k,&scores]);p},
            softmax:{let p=gpu.create_compute_pipeline(DENSE_SOFTMAX_SPV,1,std::mem::size_of::<DenseSoftmaxPush>() as u32)?;p.bind_buffers(&[&scores]);p},
            values:{let p=gpu.create_compute_pipeline(DENSE_VALUES_SPV,3,std::mem::size_of::<DenseValuesPush>() as u32)?;p.bind_buffers(&[&scores,&v,&attention]);p},
            silu_mul:{let p=gpu.create_compute_pipeline(SEQ_SILU_MUL_SPV,2,std::mem::size_of::<NPush>() as u32)?;p.bind_buffers(&[&gate,&up]);p},
        };
        let factor_name=if base.rope_scaling_type=="longrope" && base.context_length>base.rope_original_context_length {"rope_factors_long.weight"} else {"rope_factors_short.weight"};
        let rope_factor_offset=tensor_offset(base_summary.tensor(factor_name)?)?;
        Ok(Self{config,patch_in,cond_in,patch_proj,cond_proj,hidden,norm,q,k,v,attention,gate,up,branch,scores,locenc_lm,mu1,mu2,time_sin,dt_sin,time_tmp,dt_tmp,time_vec,dt_vec,out_patch,cfm_positive,cfm_negative,
            pipes:LocalPipes{locenc_in,pack_locenc,enc_to_lm,locdit_in,locdit_cond,locdit_out,time1,time2,dt1,dt2,time_sin:time_sin_p,dt_sin:dt_sin_p,time_silu,dt_silu,time_add,lm_to_dit,res_to_dit,pack_locdit,cfm_noise,cfm_cfg_euler,tr},
            command_buffer:gpu.allocate_primary_command_buffer()?,rope_factor_offset,rope_scaling_factor:base.rope_scaling_factor(),use_fused_swiglu,use_fused_qkv})
    }

    pub fn allocated_bytes(&self)->u64 { [&self.patch_in,&self.cond_in,&self.patch_proj,&self.cond_proj,&self.hidden,&self.norm,&self.q,&self.k,&self.v,&self.attention,&self.gate,&self.up,&self.branch,&self.scores,&self.locenc_lm,&self.mu1,&self.mu2,&self.time_sin,&self.dt_sin,&self.time_tmp,&self.dt_tmp,&self.time_vec,&self.dt_vec,&self.out_patch,&self.cfm_positive,&self.cfm_negative].iter().map(|x|x.size).sum() }
    pub fn locenc_output_buffer(&self)->&GpuBuffer { &self.locenc_lm }
    pub fn locdit_output_buffer(&self)->&GpuBuffer { &self.out_patch }

    pub fn encode_patch(&mut self,gpu:&VulkanContext,acoustic:&GgufSummary,base:&BaseLmConfig,patch:&[f32])->Result<LocEncResult>{
        check_patch(patch,"LocEnc patch")?;let started=Instant::now();gpu.begin_one_time(self.command_buffer)?;let upload_staging=gpu.record_upload_f32(self.command_buffer,&self.patch_in,patch)?;self.record_locenc_body(gpu,acoustic,base)?;gpu.submit_and_wait(self.command_buffer)?;drop(upload_staging);let elapsed_ms=started.elapsed().as_secs_f64()*1000.0;
        let local=gpu.read_f32(&self.norm,LOCAL_HIDDEN as usize)?;let lm=gpu.read_f32(&self.locenc_lm,2048)?;let(a,b)=stats(&local);let(c,d)=stats(&lm);Ok(LocEncResult{elapsed_ms,local_checksum:a,local_l2:b,lm_embedding_checksum:c,lm_embedding_l2:d})
    }
    pub fn encode_patch_gpu_only(&mut self,gpu:&VulkanContext,acoustic:&GgufSummary,base:&BaseLmConfig,patch:&[f32])->Result<f64>{check_patch(patch,"LocEnc patch")?;let t=Instant::now();gpu.begin_one_time(self.command_buffer)?;let upload_staging=gpu.record_upload_f32(self.command_buffer,&self.patch_in,patch)?;self.record_locenc_body(gpu,acoustic,base)?;gpu.submit_and_wait(self.command_buffer)?;drop(upload_staging);Ok(t.elapsed().as_secs_f64()*1000.0)}

    pub fn locdit_from_cpu_mu(&mut self,gpu:&VulkanContext,acoustic:&GgufSummary,base:&BaseLmConfig,x:&[f32],cond:&[f32],mu:&[f32],t:f32,dt:f32)->Result<LocDitResult>{
        check_patch(x,"LocDiT x")?;check_patch(cond,"LocDiT cond")?;check_time(t,dt)?;if mu.len()!=2048{bail!("LocDiT mu requires 2048 floats (two 1024-D tokens), got {}",mu.len())}if let Some((i,v))=mu.iter().copied().enumerate().find(|(_,v)|!v.is_finite()){bail!("LocDiT mu contains non-finite value at index {i}: {v}")}gpu.upload_f32(&self.patch_in,x)?;gpu.upload_f32(&self.cond_in,cond)?;gpu.upload_f32(&self.mu1,&mu[..1024])?;gpu.upload_f32(&self.mu2,&mu[1024..])?;let st=Instant::now();self.record_locdit(gpu,acoustic,base,t,dt,false)?;gpu.submit_and_wait(self.command_buffer)?;let elapsed_ms=st.elapsed().as_secs_f64()*1000.0;let out=gpu.read_f32(&self.out_patch,(PATCH_SIZE*FEAT_DIM)as usize)?;let(a,b)=stats(&out);Ok(LocDitResult{elapsed_ms,t,dt,output_checksum:a,output_l2:b})
    }
    pub fn locdit_from_model_hiddens(&mut self,gpu:&VulkanContext,acoustic:&GgufSummary,base:&BaseLmConfig,x:&[f32],cond:&[f32],t:f32,dt:f32)->Result<LocDitResult>{
        check_patch(x,"LocDiT x")?;check_patch(cond,"LocDiT cond")?;check_time(t,dt)?;gpu.upload_f32(&self.patch_in,x)?;gpu.upload_f32(&self.cond_in,cond)?;let st=Instant::now();self.record_locdit(gpu,acoustic,base,t,dt,true)?;gpu.submit_and_wait(self.command_buffer)?;let elapsed_ms=st.elapsed().as_secs_f64()*1000.0;let out=gpu.read_f32(&self.out_patch,256)?;let(a,b)=stats(&out);Ok(LocDitResult{elapsed_ms,t,dt,output_checksum:a,output_l2:b})
    }

    pub fn cfm_from_cpu_mu(&mut self,gpu:&VulkanContext,a:&GgufSummary,base:&BaseLmConfig,cond:&[f32],mu:&[f32],options:&CfmOptions,initial_x:Option<&[f32]>)->Result<CfmResult>{
        options.validate()?; check_patch(cond,"CFM condition")?;
        if mu.len()!=2048 { bail!("CFM mu requires 2048 floats, got {}",mu.len()); }
        if let Some((i,v))=mu.iter().copied().enumerate().find(|(_,v)|!v.is_finite()){bail!("CFM mu contains non-finite value at index {i}: {v}")}
        self.cfm_solve(gpu,a,base,cond,Some(mu),options,initial_x,false)
    }

    pub fn cfm_from_model_hiddens(&mut self,gpu:&VulkanContext,a:&GgufSummary,base:&BaseLmConfig,cond:&[f32],options:&CfmOptions,initial_x:Option<&[f32]>)->Result<CfmResult>{
        options.validate()?;check_patch(cond,"CFM condition")?;self.cfm_solve(gpu,a,base,cond,None,options,initial_x,true)
    }

    fn cfm_solve(&mut self,gpu:&VulkanContext,a:&GgufSummary,base:&BaseLmConfig,cond:&[f32],cpu_mu:Option<&[f32]>,options:&CfmOptions,initial_x:Option<&[f32]>,project_model_mu:bool)->Result<CfmResult>{
        // CFM is mathematically sequential, but it does not require a CPU round-trip between
        // Euler steps. Record the condition upload, mu preparation, noise/initial state, every
        // CFG estimator pass, every Euler update, and the final readback in one command buffer.
        // Compute/transfer barriers retain the exact dependency chain on the GPU.
        let started=Instant::now();
        gpu.begin_one_time(self.command_buffer)?;
        let mut staging_uploads=Vec::with_capacity(4);
        staging_uploads.push(gpu.record_upload_f32(self.command_buffer,&self.cond_in,cond)?);
        if let Some(mu)=cpu_mu {
            // Keep the positive CFG mu vectors resident for the complete solve.
            // The negative pass now zeros them logically in pack_locdit, so no save/restore
            // buffers or per-step device copies are necessary.
            staging_uploads.push(gpu.record_upload_f32(self.command_buffer,&self.mu1,&mu[..1024])?);
            staging_uploads.push(gpu.record_upload_f32(self.command_buffer,&self.mu2,&mu[1024..])?);
        }
        if project_model_mu {
            self.record_project_mu(gpu,a)?;
        }
        if let Some(x)=initial_x { check_patch(x,"CFM initial x")?;staging_uploads.push(gpu.record_upload_f32(self.command_buffer,&self.patch_in,x)?); }
        else {
            self.pipes.cfm_noise.bind(self.command_buffer);self.pipes.cfm_noise.push(self.command_buffer,&CfmNoisePush{n:PATCH_SIZE*FEAT_DIM,seed_lo:options.seed as u32,seed_hi:(options.seed>>32)as u32,temperature:options.temperature});unsafe{gpu.device.cmd_dispatch(self.command_buffer,1,1,1)};gpu.compute_barrier(self.command_buffer);
        }
        let span=cfm_time_span(options.n_timesteps,options.sway_sampling_coef)?;
        let zero_init_steps=if options.use_cfg_zero_star { std::cmp::max(1,((span.len() as f32)*0.04).floor() as u32) } else { 0 };
        let mut estimator_calls=0u32;
        for step in 1..span.len() {
            if step as u32 <= zero_init_steps { continue; }
            let t=span[step-1];let dt=t-span[step];
            // UnifiedCFM mean_mode=false: estimator delta-time input is always zero.
            self.record_locdit_common(gpu,a,t,0.0)?;
            self.record_locdit_body(gpu,a,base,false)?;gpu.compute_to_transfer_rw_barrier(self.command_buffer);
            unsafe{gpu.device.cmd_copy_buffer(self.command_buffer,self.out_patch.buffer,self.cfm_positive.buffer,&[vk::BufferCopy{src_offset:0,dst_offset:0,size:(PATCH_SIZE*FEAT_DIM)as u64*4}]);}
            // The unconditional CFG pass consumes zero mu tokens directly in pack_locdit;
            // avoid two device-buffer fills on every estimator pair. The barrier is still
            // required because the positive-copy reads out_patch before we overwrite it.
            gpu.transfer_to_compute_barrier(self.command_buffer);
            self.record_locdit_body(gpu,a,base,true)?;gpu.compute_to_transfer_barrier(self.command_buffer);
            unsafe{gpu.device.cmd_copy_buffer(self.command_buffer,self.out_patch.buffer,self.cfm_negative.buffer,&[vk::BufferCopy{src_offset:0,dst_offset:0,size:(PATCH_SIZE*FEAT_DIM)as u64*4}]);}
            gpu.transfer_to_compute_barrier(self.command_buffer);
            self.pipes.cfm_cfg_euler.bind(self.command_buffer);self.pipes.cfm_cfg_euler.push(self.command_buffer,&CfmEulerPush{n:PATCH_SIZE*FEAT_DIM,dt,cfg:options.cfg_value,use_zero_star:options.use_cfg_zero_star as u32});unsafe{gpu.device.cmd_dispatch(self.command_buffer,1,1,1)};gpu.compute_barrier(self.command_buffer);
            estimator_calls+=2;
        }
        let out=gpu.submit_and_read_f32(self.command_buffer,&self.patch_in,(PATCH_SIZE*FEAT_DIM)as usize)?;
        drop(staging_uploads);
        let elapsed_ms=started.elapsed().as_secs_f64()*1000.0;let(sum,l2)=stats(&out);let(min,max)=out.iter().copied().fold((f32::INFINITY,f32::NEG_INFINITY),|(mn,mx),v|(mn.min(v),mx.max(v)));
        Ok(CfmResult{elapsed_ms,n_timesteps:options.n_timesteps,estimator_calls,zero_init_steps,cfg_value:options.cfg_value,temperature:options.temperature,sway_sampling_coef:options.sway_sampling_coef,seed:options.seed,use_cfg_zero_star:options.use_cfg_zero_star,solver:"euler",mean_mode:false,output_checksum:sum,output_l2:l2,output_min:min,output_max:max,output_patch:out})
    }

    fn record_locenc(&self,gpu:&VulkanContext,a:&GgufSummary,base:&BaseLmConfig)->Result<()> {
        gpu.begin_one_time(self.command_buffer)?;
        self.record_locenc_body(gpu,a,base)
    }

    fn record_locenc_body(&self,gpu:&VulkanContext,a:&GgufSummary,base:&BaseLmConfig)->Result<()> {
        record_linear(gpu,self.command_buffer,&self.pipes.locenc_in,a.tensor("locenc.in_proj.weight")?,Some(a.tensor("locenc.in_proj.bias")?),PATCH_SIZE,FEAT_DIM,LOCAL_HIDDEN,FEAT_DIM,LOCAL_HIDDEN,0,0)?;gpu.compute_barrier(self.command_buffer);
        let cls=a.tensor("locenc.cls_token.weight")?;self.pipes.pack_locenc.bind(self.command_buffer);self.pipes.pack_locenc.push(self.command_buffer,&PackLocEncPush{cls_offset:tensor_offset(cls)?,cls_dtype:dtype(cls.ggml_type)?,hidden:LOCAL_HIDDEN,patch_tokens:PATCH_SIZE});unsafe{gpu.device.cmd_dispatch(self.command_buffer,div_up(LOCENC_TOKENS*LOCAL_HIDDEN,256),1,1)};gpu.compute_barrier(self.command_buffer);
        self.record_transformer(gpu,a,base,"locenc",LOCAL_LAYERS,LOCENC_TOKENS)?;gpu.compute_barrier(self.command_buffer);
        record_linear(gpu,self.command_buffer,&self.pipes.enc_to_lm,a.tensor("projections.enc_to_lm_proj.weight")?,Some(a.tensor("projections.enc_to_lm_proj.bias")?),1,LOCAL_HIDDEN,2048,LOCAL_HIDDEN,2048,0,0)?;
        Ok(())
    }

    fn record_locdit(&self,gpu:&VulkanContext,a:&GgufSummary,base:&BaseLmConfig,t:f32,dt:f32,project_mu:bool)->Result<()> {
        gpu.begin_one_time(self.command_buffer)?;
        self.record_locdit_common(gpu,a,t,dt)?;
        if project_mu { self.record_project_mu(gpu,a)?; }
        self.record_locdit_body(gpu,a,base,false)?;
        Ok(())
    }

    fn record_locdit_common(&self,gpu:&VulkanContext,a:&GgufSummary,t:f32,dt:f32)->Result<()> {
        record_linear(gpu,self.command_buffer,&self.pipes.locdit_in,a.tensor("locdit.in_proj.weight")?,Some(a.tensor("locdit.in_proj.bias")?),PATCH_SIZE,FEAT_DIM,LOCAL_HIDDEN,FEAT_DIM,LOCAL_HIDDEN,0,0)?;
        record_linear(gpu,self.command_buffer,&self.pipes.locdit_cond,a.tensor("locdit.cond_proj.weight")?,Some(a.tensor("locdit.cond_proj.bias")?),PATCH_SIZE,FEAT_DIM,LOCAL_HIDDEN,FEAT_DIM,LOCAL_HIDDEN,0,0)?;gpu.compute_barrier(self.command_buffer);
        self.record_time(gpu,a,t,dt)
    }

    fn record_project_mu(&self,gpu:&VulkanContext,a:&GgufSummary)->Result<()> {
        record_linear(gpu,self.command_buffer,&self.pipes.lm_to_dit,a.tensor("projections.lm_to_dit_proj.weight")?,Some(a.tensor("projections.lm_to_dit_proj.bias")?),1,2048,LOCAL_HIDDEN,2048,LOCAL_HIDDEN,0,0)?;
        record_linear(gpu,self.command_buffer,&self.pipes.res_to_dit,a.tensor("projections.res_to_dit_proj.weight")?,Some(a.tensor("projections.res_to_dit_proj.bias")?),1,2048,LOCAL_HIDDEN,2048,LOCAL_HIDDEN,0,0)?;gpu.compute_buffer_barrier(self.command_buffer,&[&self.mu1,&self.mu2]);Ok(())
    }

    fn record_locdit_body(&self,gpu:&VulkanContext,a:&GgufSummary,base:&BaseLmConfig,zero_mu:bool)->Result<()> {
        self.pipes.pack_locdit.bind(self.command_buffer);self.pipes.pack_locdit.push(self.command_buffer,&PackLocDitPush{hidden:LOCAL_HIDDEN,patch_tokens:PATCH_SIZE,zero_mu:zero_mu as u32});unsafe{gpu.device.cmd_dispatch(self.command_buffer,div_up(LOCDIT_TOKENS*LOCAL_HIDDEN,256),1,1)};gpu.compute_barrier(self.command_buffer);
        self.record_transformer(gpu,a,base,"locdit",LOCAL_LAYERS,LOCDIT_TOKENS)?;gpu.compute_barrier(self.command_buffer);
        record_linear(gpu,self.command_buffer,&self.pipes.locdit_out,a.tensor("locdit.out_proj.weight")?,Some(a.tensor("locdit.out_proj.bias")?),PATCH_SIZE,LOCAL_HIDDEN,FEAT_DIM,LOCAL_HIDDEN,FEAT_DIM,7*LOCAL_HIDDEN,0)?;Ok(())
    }

    fn record_time(&self,gpu:&VulkanContext,a:&GgufSummary,t:f32,dt:f32)->Result<()> {
        self.pipes.time_sin.bind(self.command_buffer);self.pipes.time_sin.push(self.command_buffer,&TimePush{dim:LOCAL_HIDDEN,t});unsafe{gpu.device.cmd_dispatch(self.command_buffer,4,1,1)};
        self.pipes.dt_sin.bind(self.command_buffer);self.pipes.dt_sin.push(self.command_buffer,&TimePush{dim:LOCAL_HIDDEN,t:dt});unsafe{gpu.device.cmd_dispatch(self.command_buffer,4,1,1)};gpu.compute_barrier(self.command_buffer);
        record_linear(gpu,self.command_buffer,&self.pipes.time1,a.tensor("locdit.time_mlp.linear_1.weight")?,Some(a.tensor("locdit.time_mlp.linear_1.bias")?),1,LOCAL_HIDDEN,LOCAL_HIDDEN,LOCAL_HIDDEN,LOCAL_HIDDEN,0,0)?;
        record_linear(gpu,self.command_buffer,&self.pipes.dt1,a.tensor("locdit.delta_time_mlp.linear_1.weight")?,Some(a.tensor("locdit.delta_time_mlp.linear_1.bias")?),1,LOCAL_HIDDEN,LOCAL_HIDDEN,LOCAL_HIDDEN,LOCAL_HIDDEN,0,0)?;gpu.compute_barrier(self.command_buffer);
        self.pipes.time_silu.bind(self.command_buffer);self.pipes.time_silu.push(self.command_buffer,&NPush{n:LOCAL_HIDDEN});unsafe{gpu.device.cmd_dispatch(self.command_buffer,4,1,1)};
        self.pipes.dt_silu.bind(self.command_buffer);self.pipes.dt_silu.push(self.command_buffer,&NPush{n:LOCAL_HIDDEN});unsafe{gpu.device.cmd_dispatch(self.command_buffer,4,1,1)};gpu.compute_barrier(self.command_buffer);
        record_linear(gpu,self.command_buffer,&self.pipes.time2,a.tensor("locdit.time_mlp.linear_2.weight")?,Some(a.tensor("locdit.time_mlp.linear_2.bias")?),1,LOCAL_HIDDEN,LOCAL_HIDDEN,LOCAL_HIDDEN,LOCAL_HIDDEN,0,0)?;
        record_linear(gpu,self.command_buffer,&self.pipes.dt2,a.tensor("locdit.delta_time_mlp.linear_2.weight")?,Some(a.tensor("locdit.delta_time_mlp.linear_2.bias")?),1,LOCAL_HIDDEN,LOCAL_HIDDEN,LOCAL_HIDDEN,LOCAL_HIDDEN,0,0)?;gpu.compute_barrier(self.command_buffer);
        self.pipes.time_add.bind(self.command_buffer);self.pipes.time_add.push(self.command_buffer,&NPush{n:LOCAL_HIDDEN});unsafe{gpu.device.cmd_dispatch(self.command_buffer,4,1,1)};gpu.compute_barrier(self.command_buffer);Ok(())
    }

    fn record_transformer(&self,gpu:&VulkanContext,a:&GgufSummary,base:&BaseLmConfig,prefix:&str,layers:u32,tokens:u32)->Result<()> {
        record_rms(gpu,self.command_buffer,&self.pipes.tr.rms_attn,a.tensor(&format!("{prefix}.blk.0.attn_norm.weight"))?,tokens,base.rms_epsilon)?;
        gpu.compute_buffer_barrier(self.command_buffer,&[&self.norm]);
        for layer in 0..layers {
            let n=|s:&str|format!("{prefix}.blk.{layer}.{s}");
            if self.use_fused_qkv {
                record_qkv(gpu,self.command_buffer,&self.pipes.tr.qkv,
                    a.tensor(&n("attn_q.weight"))?,a.tensor(&n("attn_k.weight"))?,a.tensor(&n("attn_v.weight"))?,tokens)?;
            } else {
                // Keep the explicit cooperative-matrix projections when that experimental
                // XTX path is enabled; the portable fused QKV shader must not replace it.
                record_linear(gpu,self.command_buffer,&self.pipes.tr.q,a.tensor(&n("attn_q.weight"))?,None,tokens,LOCAL_HIDDEN,Q_DIM,LOCAL_HIDDEN,Q_DIM,0,0)?;
                record_linear(gpu,self.command_buffer,&self.pipes.tr.k,a.tensor(&n("attn_k.weight"))?,None,tokens,LOCAL_HIDDEN,KV_DIM,LOCAL_HIDDEN,KV_DIM,0,0)?;
                record_linear(gpu,self.command_buffer,&self.pipes.tr.v,a.tensor(&n("attn_v.weight"))?,None,tokens,LOCAL_HIDDEN,KV_DIM,LOCAL_HIDDEN,KV_DIM,0,0)?;
            }
            gpu.compute_buffer_barrier(self.command_buffer,&[&self.q,&self.k]);
            self.pipes.tr.rope.bind(self.command_buffer);self.pipes.tr.rope.push(self.command_buffer,&SeqRopePush{factor_offset:self.rope_factor_offset,tokens,q_heads:LOCAL_Q_HEADS,kv_heads:LOCAL_KV_HEADS,head_dim:HEAD_DIM,q_stride:Q_DIM,k_stride:KV_DIM,rope_theta:base.rope_theta,scaling_factor:self.rope_scaling_factor});unsafe{gpu.device.cmd_dispatch(self.command_buffer,LOCAL_Q_HEADS+LOCAL_KV_HEADS,tokens,1)};
            gpu.compute_buffer_barrier(self.command_buffer,&[&self.q,&self.k,&self.v]);
            let score_span=gpu.gpu_profile_begin(self.command_buffer,"local.attn_scores");self.pipes.tr.scores.bind(self.command_buffer);self.pipes.tr.scores.push(self.command_buffer,&DenseScoresPush{tokens,q_heads:LOCAL_Q_HEADS,kv_heads:LOCAL_KV_HEADS,head_dim:HEAD_DIM,q_stride:Q_DIM,k_stride:KV_DIM,score_stride:LOCDIT_TOKENS});unsafe{gpu.device.cmd_dispatch(self.command_buffer,tokens,LOCAL_Q_HEADS,tokens)};gpu.gpu_profile_end(self.command_buffer,score_span);
            gpu.compute_buffer_barrier(self.command_buffer,&[&self.scores]);
            let softmax_span=gpu.gpu_profile_begin(self.command_buffer,"local.softmax");self.pipes.tr.softmax.bind(self.command_buffer);self.pipes.tr.softmax.push(self.command_buffer,&DenseSoftmaxPush{tokens,q_heads:LOCAL_Q_HEADS,score_stride:LOCDIT_TOKENS});unsafe{gpu.device.cmd_dispatch(self.command_buffer,LOCAL_Q_HEADS,tokens,1)};gpu.gpu_profile_end(self.command_buffer,softmax_span);
            gpu.compute_buffer_barrier(self.command_buffer,&[&self.scores]);
            let values_span=gpu.gpu_profile_begin(self.command_buffer,"local.attn_values");self.pipes.tr.values.bind(self.command_buffer);self.pipes.tr.values.push(self.command_buffer,&DenseValuesPush{tokens,q_heads:LOCAL_Q_HEADS,kv_heads:LOCAL_KV_HEADS,head_dim:HEAD_DIM,out_stride:Q_DIM,v_stride:KV_DIM,score_stride:LOCDIT_TOKENS});unsafe{gpu.device.cmd_dispatch(self.command_buffer,LOCAL_Q_HEADS,tokens,1)};gpu.gpu_profile_end(self.command_buffer,values_span);
            gpu.compute_buffer_barrier(self.command_buffer,&[&self.attention]);
            record_linear(gpu,self.command_buffer,&self.pipes.tr.attn_out,a.tensor(&n("attn_output.weight"))?,None,tokens,Q_DIM,LOCAL_HIDDEN,Q_DIM,LOCAL_HIDDEN,0,0)?;
            gpu.compute_buffer_barrier(self.command_buffer,&[&self.branch]);
            record_residual_rms(gpu,self.command_buffer,&self.pipes.tr.residual_rms,a.tensor(&n("ffn_norm.weight"))?,tokens,base.rms_epsilon,1.0)?;
            gpu.compute_buffer_barrier(self.command_buffer,&[&self.norm]);
            if self.use_fused_swiglu {
                record_swiglu(gpu,self.command_buffer,&self.pipes.tr.swiglu,a.tensor(&n("ffn_gate.weight"))?,a.tensor(&n("ffn_up.weight"))?,tokens,LOCAL_HIDDEN,LOCAL_FFN,LOCAL_HIDDEN,LOCAL_FFN,0,0)?;
                gpu.compute_buffer_barrier(self.command_buffer,&[&self.gate]);
            } else {
                record_linear(gpu,self.command_buffer,&self.pipes.tr.gate,a.tensor(&n("ffn_gate.weight"))?,None,tokens,LOCAL_HIDDEN,LOCAL_FFN,LOCAL_HIDDEN,LOCAL_FFN,0,0)?;
                record_linear(gpu,self.command_buffer,&self.pipes.tr.up,a.tensor(&n("ffn_up.weight"))?,None,tokens,LOCAL_HIDDEN,LOCAL_FFN,LOCAL_HIDDEN,LOCAL_FFN,0,0)?;
                gpu.compute_buffer_barrier(self.command_buffer,&[&self.gate,&self.up]);
                self.pipes.tr.silu_mul.bind(self.command_buffer);self.pipes.tr.silu_mul.push(self.command_buffer,&NPush{n:tokens*LOCAL_FFN});unsafe{gpu.device.cmd_dispatch(self.command_buffer,div_up(tokens*LOCAL_FFN,256),1,1)};
                gpu.compute_buffer_barrier(self.command_buffer,&[&self.gate]);
            }
            record_linear(gpu,self.command_buffer,&self.pipes.tr.down,a.tensor(&n("ffn_down.weight"))?,None,tokens,LOCAL_FFN,LOCAL_HIDDEN,LOCAL_FFN,LOCAL_HIDDEN,0,0)?;
            gpu.compute_buffer_barrier(self.command_buffer,&[&self.branch]);
            let next_norm=if layer+1<layers {a.tensor(&format!("{prefix}.blk.{}.attn_norm.weight",layer+1))?} else {a.tensor(&format!("{prefix}.norm.weight"))?};
            record_residual_rms(gpu,self.command_buffer,&self.pipes.tr.residual_rms,next_norm,tokens,base.rms_epsilon,1.0)?;
            gpu.compute_buffer_barrier(self.command_buffer,&[&self.norm]);
        }
        Ok(())
    }

}

fn cfm_time_span(n:u32,sway:f32)->Result<Vec<f32>> {
    if n==0 { bail!("CFM n_timesteps must be >= 1"); }
    let mut out=Vec::with_capacity(n as usize+1);
    for i in 0..=n { let t=1.0-(i as f32)/(n as f32); let w=t+sway*((std::f32::consts::FRAC_PI_2*t).cos()-1.0+t); if !w.is_finite(){bail!("non-finite CFM time grid at {i}")} out.push(w); }
    for i in 1..out.len(){ if out[i]>out[i-1]+1.0e-6 { bail!("CFM sway schedule is not descending at {}: {} -> {}",i,out[i-1],out[i]); } }
    Ok(out)
}

fn record_qkv(gpu:&VulkanContext,cmd:vk::CommandBuffer,p:&ComputePipeline,q:&TensorInfo,k:&TensorInfo,v:&TensorInfo,tokens:u32)->Result<()> {
    if q.dims!=[LOCAL_HIDDEN as u64,Q_DIM as u64]||k.dims!=[LOCAL_HIDDEN as u64,KV_DIM as u64]||v.dims!=[LOCAL_HIDDEN as u64,KV_DIM as u64]{bail!("local QKV shape mismatch")};
    if q.ggml_type!=GgmlType::F16||k.ggml_type!=GgmlType::F16||v.ggml_type!=GgmlType::F16{bail!("local QKV weights must be F16")};
    let span=gpu.gpu_profile_begin(cmd,"local.seq_qkv");
    p.bind(cmd);p.push(cmd,&SeqQkvPush{q_weight_offset:tensor_offset(q)?,k_weight_offset:tensor_offset(k)?,v_weight_offset:tensor_offset(v)?,q_rows:Q_DIM,kv_rows:KV_DIM,cols:LOCAL_HIDDEN,tokens,input_stride:LOCAL_HIDDEN,q_stride:Q_DIM,kv_stride:KV_DIM,input_base:0});
    unsafe{gpu.device.cmd_dispatch(cmd,Q_DIM+2*KV_DIM,tokens,1)};
    gpu.gpu_profile_end(cmd,span);Ok(())
}

fn record_linear(gpu:&VulkanContext,cmd:vk::CommandBuffer,p:&ComputePipeline,w:&TensorInfo,bias:Option<&TensorInfo>,tokens:u32,cols:u32,rows:u32,input_stride:u32,output_stride:u32,input_base:u32,output_base:u32)->Result<()> {
    let (bo,bd,hb)=if let Some(b)=bias{(tensor_offset(b)?,dtype(b.ggml_type)?,1)}else{(0,0,0)};
    let span=gpu.gpu_profile_begin(cmd,"local.seq_linear");
    p.bind(cmd);p.push(cmd,&SeqLinearPush{weight_offset:tensor_offset(w)?,bias_offset:bo,rows,cols,weight_dtype:dtype(w.ggml_type)?,bias_dtype:bd,tokens,input_stride,output_stride,input_base,output_base,has_bias:hb,alpha:1.0});
    unsafe{
        if gpu.xtx_coopmat_enabled(){gpu.device.cmd_dispatch(cmd,div_up(rows,16),1,1)}
        else{gpu.device.cmd_dispatch(cmd,rows,tokens,1)}
    };
    gpu.gpu_profile_end(cmd,span);Ok(())
}
fn record_rms(gpu:&VulkanContext,cmd:vk::CommandBuffer,p:&ComputePipeline,w:&TensorInfo,tokens:u32,eps:f32)->Result<()> {let span=gpu.gpu_profile_begin(cmd,"local.seq_rmsnorm");p.bind(cmd);p.push(cmd,&SeqRmsPush{weight_offset:tensor_offset(w)?,n:LOCAL_HIDDEN,tokens,input_stride:LOCAL_HIDDEN,output_stride:LOCAL_HIDDEN,input_base:0,output_base:0,eps});unsafe{gpu.device.cmd_dispatch(cmd,tokens,1,1)};gpu.gpu_profile_end(cmd,span);Ok(())}
fn record_residual_rms(gpu:&VulkanContext,cmd:vk::CommandBuffer,p:&ComputePipeline,w:&TensorInfo,tokens:u32,eps:f32,scale:f32)->Result<()> {let span=gpu.gpu_profile_begin(cmd,"local.seq_residual_rmsnorm");p.bind(cmd);p.push(cmd,&SeqResidualRmsPush{weight_offset:tensor_offset(w)?,n:LOCAL_HIDDEN,tokens,hidden_stride:LOCAL_HIDDEN,branch_stride:LOCAL_HIDDEN,output_stride:LOCAL_HIDDEN,hidden_base:0,branch_base:0,output_base:0,eps,scale});unsafe{gpu.device.cmd_dispatch(cmd,tokens,1,1)};gpu.gpu_profile_end(cmd,span);Ok(())}
fn record_swiglu(gpu:&VulkanContext,cmd:vk::CommandBuffer,p:&ComputePipeline,gate:&TensorInfo,up:&TensorInfo,tokens:u32,cols:u32,rows:u32,input_stride:u32,output_stride:u32,input_base:u32,output_base:u32)->Result<()> {if gate.dims!=[cols as u64,rows as u64]||up.dims!=[cols as u64,rows as u64]{bail!("local SwiGLU shape mismatch")};if gate.ggml_type!=GgmlType::F16||up.ggml_type!=GgmlType::F16{bail!("local SwiGLU weights must be F16")};let span=gpu.gpu_profile_begin(cmd,"local.seq_swiglu");p.bind(cmd);p.push(cmd,&SeqSwigluPush{gate_weight_offset:tensor_offset(gate)?,up_weight_offset:tensor_offset(up)?,rows,cols,tokens,input_stride,output_stride,input_base,output_base});unsafe{gpu.device.cmd_dispatch(cmd,rows,tokens,1)};gpu.gpu_profile_end(cmd,span);Ok(())}
fn tensor_offset(t:&TensorInfo)->Result<u32>{u32::try_from(t.offset).with_context(||format!("tensor {} offset exceeds u32",t.name))}
fn dtype(t:GgmlType)->Result<u32>{match t{GgmlType::F32=>Ok(0),GgmlType::F16=>Ok(1),GgmlType::Q8_0=>Ok(8),GgmlType::Other(x)=>bail!("unsupported dtype {x}")}}
fn div_up(n:u32,d:u32)->u32{(n+d-1)/d}
fn check_patch(v:&[f32],name:&str)->Result<()>{if v.len()!=(PATCH_SIZE*FEAT_DIM)as usize{bail!("{name} requires exactly {} floats (4x64), got {}",PATCH_SIZE*FEAT_DIM,v.len())}if let Some((i,x))=v.iter().copied().enumerate().find(|(_,x)|!x.is_finite()){bail!("{name} contains non-finite value at index {i}: {x}")}Ok(())}
fn check_time(t:f32,dt:f32)->Result<()>{if !t.is_finite()||!dt.is_finite(){bail!("LocDiT t/dt must be finite, got t={t}, dt={dt}")}Ok(())}
fn stats(v:&[f32])->(f64,f64){let sum=v.iter().map(|&x|x as f64).sum();let l2=v.iter().map(|&x|(x as f64)*(x as f64)).sum::<f64>().sqrt();(sum,l2)}
