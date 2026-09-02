use crate::{
    gguf::{GgmlType, GgufSummary, TensorInfo},
    vulkan::{ComputePipeline, GpuBuffer, VulkanContext},
};
use anyhow::{bail, Context, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use serde::Serialize;
use std::{f64::consts::PI, path::Path, time::Instant};

const CONV1D_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vae_conv1d.spv"));
const CONVT1D_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vae_convtranspose1d.spv"));
const SNAKE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vae_snake.spv"));
const SCALE_BIAS_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vae_scale_bias.spv"));
const TANH_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vae_tanh.spv"));
const ADD_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vae_add.spv"));

const ENCODER_RATES: [u32; 4] = [2, 5, 8, 8];
const DECODER_RATES: [u32; 6] = [8, 6, 5, 2, 2, 2];
const SR_BINS: [u32; 3] = [20_000, 30_000, 40_000];
const DECODER_SR_BUCKETS: u32 = 4;
const PATCH_SIZE: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioPadSide { Left, Right }

#[derive(Debug, Clone, Serialize)]
pub struct AudioVaeConfig {
    pub encoder_dim: u32,
    pub latent_dim: u32,
    pub decoder_dim: u32,
    pub sample_rate: u32,
    pub out_sample_rate: u32,
    pub encoder_rates: Vec<u32>,
    pub decoder_rates: Vec<u32>,
    pub encoder_hop: u32,
    pub decoder_hop: u32,
    pub patch_size: u32,
    pub input_samples_per_patch: u32,
    pub output_samples_per_patch: u32,
    pub sr_bin_boundaries: Vec<u32>,
    pub decoder_sr_bucket: u32,
    pub cond_type: String,
    pub depthwise: bool,
}

impl AudioVaeConfig {
    pub fn from_gguf(s: &GgufSummary) -> Result<Self> {
        let version = s.metadata_f32("voxcpm.model_version")?;
        if (version - 2.0).abs() > 0.01 { bail!("AudioVAE V2 requires VoxCPM2 acoustic model_version=2.0, got {version}"); }
        let encoder_dim = s.metadata_u32("voxcpm.audiovae.encoder_dim")?;
        let latent_dim = s.metadata_u32("voxcpm.audiovae.latent_dim")?;
        let decoder_dim = s.metadata_u32("voxcpm.audiovae.decoder_dim")?;
        let sample_rate = s.metadata_u32("voxcpm.audiovae.sample_rate")?;
        let out_sample_rate = s.metadata_u32("voxcpm.audiovae.out_sample_rate")?;
        let cond_type = s.metadata_str("voxcpm.audiovae.cond_type")?.to_owned();
        let patch_size = s.metadata_u32("voxcpm.patch_size")?;
        if (encoder_dim, latent_dim, decoder_dim, sample_rate, out_sample_rate, patch_size)
            != (128, 64, 2048, 16_000, 48_000, 4)
        {
            bail!("unsupported VoxCPM2 AudioVAE geometry: enc={encoder_dim}, latent={latent_dim}, dec={decoder_dim}, sr={sample_rate}, out_sr={out_sample_rate}, patch={patch_size}");
        }
        if cond_type != "scale_bias" { bail!("VoxGen step 6 requires AudioVAE cond_type=scale_bias, got {cond_type:?}"); }
        let encoder_hop = ENCODER_RATES.iter().product();
        let decoder_hop = DECODER_RATES.iter().product();
        let decoder_sr_bucket = SR_BINS.iter().take_while(|&&x| out_sample_rate > x).count() as u32;
        let cfg = Self {
            encoder_dim, latent_dim, decoder_dim, sample_rate, out_sample_rate,
            encoder_rates: ENCODER_RATES.to_vec(), decoder_rates: DECODER_RATES.to_vec(),
            encoder_hop, decoder_hop, patch_size,
            input_samples_per_patch: encoder_hop * patch_size,
            output_samples_per_patch: decoder_hop * patch_size,
            sr_bin_boundaries: SR_BINS.to_vec(), decoder_sr_bucket, cond_type,
            depthwise: true,
        };
        cfg.validate_tensors(s)?;
        Ok(cfg)
    }

    fn validate_tensors(&self, s: &GgufSummary) -> Result<()> {
        // Encoder first convolution.
        require_conv(s, "audio_vae.encoder.block.0", 1, 128, 7, 1, false)?;
        let mut c = self.encoder_dim;
        for (stage, &stride) in ENCODER_RATES.iter().enumerate() {
            let block = stage + 1;
            for (ri, dilation) in [1u32, 3, 9].into_iter().enumerate() {
                let p = format!("audio_vae.encoder.block.{block}.block.{ri}.block");
                require_alpha(s, &format!("{p}.0.alpha"), c)?;
                require_conv(s, &format!("{p}.1"), c, c, 7, c, false)?;
                // Dilation is encoded by execution, not tensor geometry.
                let _ = dilation;
                require_alpha(s, &format!("{p}.2.alpha"), c)?;
                require_conv(s, &format!("{p}.3"), c, c, 1, 1, false)?;
            }
            require_alpha(s, &format!("audio_vae.encoder.block.{block}.block.3.alpha"), c)?;
            require_conv(s, &format!("audio_vae.encoder.block.{block}.block.4"), c, c * 2, stride * 2, 1, false)?;
            c *= 2;
        }
        require_conv(s, "audio_vae.encoder.fc_mu", c, self.latent_dim, 3, 1, false)?;
        // fc_logvar is part of the checkpoint although inference uses mu only; validate it so malformed VAE files fail early.
        require_conv(s, "audio_vae.encoder.fc_logvar", c, self.latent_dim, 3, 1, false)?;

        // Depthwise decoder stem.
        require_conv(s, "audio_vae.decoder.model.0", self.latent_dim, self.latent_dim, 7, self.latent_dim, false)?;
        require_conv(s, "audio_vae.decoder.model.1", self.latent_dim, self.decoder_dim, 1, 1, false)?;
        let mut din = self.decoder_dim;
        for (i, &stride) in DECODER_RATES.iter().enumerate() {
            let mi = i + 2;
            let dout = din / 2;
            require_sr_embed(s, mi, din)?;
            require_alpha(s, &format!("audio_vae.decoder.model.{mi}.block.0.alpha"), din)?;
            require_conv(s, &format!("audio_vae.decoder.model.{mi}.block.1"), din, dout, stride * 2, 1, true)?;
            for (rj, _dilation) in [1u32, 3, 9].into_iter().enumerate() {
                let bi = rj + 2; // no NoiseBlock in VoxCPM2
                let p = format!("audio_vae.decoder.model.{mi}.block.{bi}.block");
                require_alpha(s, &format!("{p}.0.alpha"), dout)?;
                require_conv(s, &format!("{p}.1"), dout, dout, 7, dout, false)?;
                require_alpha(s, &format!("{p}.2.alpha"), dout)?;
                require_conv(s, &format!("{p}.3"), dout, dout, 1, 1, false)?;
            }
            din = dout;
        }
        require_alpha(s, "audio_vae.decoder.model.8.alpha", din)?;
        require_conv(s, "audio_vae.decoder.model.9", din, 1, 7, 1, false)?;
        Ok(())
    }
}

fn require_f16(t: &TensorInfo) -> Result<()> {
    if t.ggml_type != GgmlType::F16 { bail!("AudioVAE tensor {} is {}, expected F16", t.name, t.ggml_type.name()); }
    Ok(())
}
fn require_alpha(s: &GgufSummary, name: &str, channels: u32) -> Result<()> {
    let t = s.tensor(name)?; require_f16(t)?;
    if t.elements != channels as u64 { bail!("AudioVAE alpha {} has {} values, expected {channels}", name, t.elements); }
    Ok(())
}
fn require_conv(s: &GgufSummary, base: &str, in_c: u32, out_c: u32, kernel: u32, groups: u32, transpose: bool) -> Result<()> {
    let w = s.tensor(base)?; require_f16(w)?;
    let expected = if transpose {
        vec![kernel as u64, (out_c / groups) as u64, in_c as u64]
    } else {
        vec![kernel as u64, (in_c / groups) as u64, out_c as u64]
    };
    if w.dims != expected { bail!("AudioVAE tensor {} dimensions {:?}, expected {:?}", w.name, w.dims, expected); }
    let b = s.tensor(&format!("{base}.bias"))?; require_f16(b)?;
    if b.elements != out_c as u64 { bail!("AudioVAE bias {} has {} values, expected {out_c}", b.name, b.elements); }
    Ok(())
}
fn require_sr_embed(s: &GgufSummary, model_index: usize, channels: u32) -> Result<()> {
    for kind in ["scale_embed", "bias_embed"] {
        let name = format!("audio_vae.decoder.sr_cond_model.{model_index}.{kind}");
        let t = s.tensor(&name)?; require_f16(t)?;
        if t.dims != [channels as u64, DECODER_SR_BUCKETS as u64] {
            bail!("AudioVAE {} dimensions {:?}, expected [{channels}, {DECODER_SR_BUCKETS}]", name, t.dims);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioEncodeStats {
    pub source_sample_rate: u32,
    pub source_channels: u16,
    pub source_samples_mono: usize,
    pub resampled_samples: usize,
    pub padded_samples: usize,
    pub pad_side: AudioPadSide,
    pub latent_frames: usize,
    pub latent_patches: usize,
    pub checksum: f64,
    pub l2: f64,
    pub encode_ms: f64,
    pub scratch_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioDecodeStats {
    pub latent_frames: usize,
    pub latent_patches: usize,
    pub output_samples: usize,
    pub output_sample_rate: u32,
    pub duration_seconds: f64,
    pub checksum: f64,
    pub l2: f64,
    pub peak: f32,
    pub decode_ms: f64,
    pub scratch_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioVaeState {
    pub ready: bool,
    pub config: AudioVaeConfig,
    pub scratch_bytes: u64,
}

#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct ConvPush {
    weight_offset:u32,bias_offset:u32,in_channels:u32,out_channels:u32,in_length:u32,out_length:u32,
    kernel:u32,stride:u32,dilation:u32,groups:u32,left_pad:u32,weight_dtype:u32,bias_dtype:u32,has_bias:u32,
}
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct ConvTPush { weight_offset:u32,bias_offset:u32,in_channels:u32,out_channels:u32,in_length:u32,out_length:u32,kernel:u32,stride:u32,groups:u32,weight_dtype:u32,bias_dtype:u32,has_bias:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct SnakePush { alpha_offset:u32,channels:u32,length:u32,dtype:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)]
struct ScalePush { scale_offset:u32,bias_offset:u32,channels:u32,length:u32,bucket:u32,buckets:u32,scale_dtype:u32,bias_dtype:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)] struct NPush { n:u32 }
#[repr(C)] #[derive(Clone, Copy, Pod, Zeroable)] struct GridPush { channels:u32, length:u32 }

struct Scratch {
    capacity_elements: usize,
    a: GpuBuffer, b: GpuBuffer, c: GpuBuffer,
    conv_ab: ComputePipeline, conv_ba: ComputePipeline,
    convt_ab: ComputePipeline, convt_ba: ComputePipeline,
    snake_ab: ComputePipeline, snake_ba: ComputePipeline,
    add_a_c: ComputePipeline, add_b_c: ComputePipeline,
    scale_a: ComputePipeline, scale_b: ComputePipeline,
    tanh_a: ComputePipeline, tanh_b: ComputePipeline,
}
impl Scratch {
    fn bytes(&self) -> u64 { self.a.size + self.b.size + self.c.size }
    fn new(gpu:&VulkanContext, model:&GpuBuffer, elements:usize)->Result<Self>{
        let bytes=(elements.max(1) as u64).checked_mul(4).context("AudioVAE scratch overflow")?;
        if bytes>gpu.info.max_storage_buffer_range { bail!("AudioVAE scratch buffer requires {:.2} GiB, exceeding Vulkan maxStorageBufferRange {:.2} GiB",bytes as f64/1073741824.0,gpu.info.max_storage_buffer_range as f64/1073741824.0); }
        let usage=vk::BufferUsageFlags::STORAGE_BUFFER|vk::BufferUsageFlags::TRANSFER_SRC|vk::BufferUsageFlags::TRANSFER_DST;
        let mem=vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let a=gpu.create_buffer(bytes,usage,mem)?; let b=gpu.create_buffer(bytes,usage,mem)?; let c=gpu.create_buffer(bytes,usage,mem)?;
        let conv_ab=pipe3(gpu,CONV1D_SPV,std::mem::size_of::<ConvPush>() as u32,model,&a,&b)?;
        let conv_ba=pipe3(gpu,CONV1D_SPV,std::mem::size_of::<ConvPush>() as u32,model,&b,&a)?;
        let convt_ab=pipe3(gpu,CONVT1D_SPV,std::mem::size_of::<ConvTPush>() as u32,model,&a,&b)?;
        let convt_ba=pipe3(gpu,CONVT1D_SPV,std::mem::size_of::<ConvTPush>() as u32,model,&b,&a)?;
        let snake_ab=pipe3(gpu,SNAKE_SPV,std::mem::size_of::<SnakePush>() as u32,model,&a,&b)?;
        let snake_ba=pipe3(gpu,SNAKE_SPV,std::mem::size_of::<SnakePush>() as u32,model,&b,&a)?;
        let add_a_c=pipe2(gpu,ADD_SPV,std::mem::size_of::<GridPush>() as u32,&a,&c)?;
        let add_b_c=pipe2(gpu,ADD_SPV,std::mem::size_of::<GridPush>() as u32,&b,&c)?;
        let scale_a=pipe2(gpu,SCALE_BIAS_SPV,std::mem::size_of::<ScalePush>() as u32,model,&a)?;
        let scale_b=pipe2(gpu,SCALE_BIAS_SPV,std::mem::size_of::<ScalePush>() as u32,model,&b)?;
        let tanh_a=pipe1(gpu,TANH_SPV,std::mem::size_of::<NPush>() as u32,&a)?;
        let tanh_b=pipe1(gpu,TANH_SPV,std::mem::size_of::<NPush>() as u32,&b)?;
        Ok(Self{capacity_elements:elements,a,b,c,conv_ab,conv_ba,convt_ab,convt_ba,snake_ab,snake_ba,add_a_c,add_b_c,scale_a,scale_b,tanh_a,tanh_b})
    }
}
fn pipe1(g:&VulkanContext,spv:&[u8],pc:u32,a:&GpuBuffer)->Result<ComputePipeline>{let p=g.create_compute_pipeline(spv,1,pc)?;p.bind_buffers(&[a]);Ok(p)}
fn pipe2(g:&VulkanContext,spv:&[u8],pc:u32,a:&GpuBuffer,b:&GpuBuffer)->Result<ComputePipeline>{let p=g.create_compute_pipeline(spv,2,pc)?;p.bind_buffers(&[a,b]);Ok(p)}
fn pipe3(g:&VulkanContext,spv:&[u8],pc:u32,a:&GpuBuffer,b:&GpuBuffer,c:&GpuBuffer)->Result<ComputePipeline>{let p=g.create_compute_pipeline(spv,3,pc)?;p.bind_buffers(&[a,b,c]);Ok(p)}

#[derive(Clone,Copy,PartialEq,Eq)] enum Buf { A, B }
impl Buf { fn other(self)->Self{if self==Self::A{Self::B}else{Self::A}} }

pub struct AudioVaeEngine { pub config: AudioVaeConfig, scratch: Option<Scratch>, command_buffer: vk::CommandBuffer }

impl AudioVaeEngine {
    pub fn new(gpu:&VulkanContext, acoustic:&GgufSummary)->Result<Self>{
        let config=AudioVaeConfig::from_gguf(acoustic)?;
        let command_buffer=gpu.allocate_primary_command_buffer()?;
        Ok(Self{config,scratch:None,command_buffer})
    }
    pub fn scratch_bytes(&self)->u64{self.scratch.as_ref().map(Scratch::bytes).unwrap_or(0)}
    fn ensure_scratch(&mut self,gpu:&VulkanContext,model:&GpuBuffer,elements:usize)->Result<()> {
        if self.scratch.as_ref().map(|s|s.capacity_elements>=elements).unwrap_or(false){return Ok(());}
        // Grow geometrically to avoid rebuilding descriptor/pipeline state for small length changes.
        let cap=elements.checked_next_power_of_two().context("AudioVAE scratch capacity overflow")?;
        self.scratch=Some(Scratch::new(gpu,model,cap)?);
        Ok(())
    }

    pub fn encode_wav(&mut self,gpu:&VulkanContext,model:&GpuBuffer,acoustic:&GgufSummary,path:&Path,pad_side:AudioPadSide)->Result<(AudioEncodeStats,Vec<f32>)>{
        let (samples,src_rate,channels)=read_wav_mono(path)?;
        if samples.is_empty(){bail!("WAV {} contains no samples",path.display());}
        if samples.iter().any(|x|!x.is_finite()){bail!("WAV {} contains NaN/inf samples",path.display());}
        let resampled=resample_sinc(&samples,src_rate,self.config.sample_rate);
        let resampled_len=resampled.len();
        let mut aligned=align_samples(resampled,self.config.input_samples_per_patch as usize,pad_side);
        let source_len=samples.len();
        let t0=Instant::now();
        let latent_channel_major=self.encode_samples(gpu,model,acoustic,&aligned)?;
        let latent_frames=latent_channel_major.len()/self.config.latent_dim as usize;
        let latents=channel_to_frame_major(&latent_channel_major,self.config.latent_dim as usize,latent_frames);
        let (checksum,l2)=stats(&latents);
        aligned.clear();
        let result=AudioEncodeStats{source_sample_rate:src_rate,source_channels:channels,source_samples_mono:source_len,resampled_samples:resampled_len,padded_samples:latent_frames*self.config.encoder_hop as usize,pad_side,latent_frames,latent_patches:latent_frames/self.config.patch_size as usize,checksum,l2,encode_ms:t0.elapsed().as_secs_f64()*1000.0,scratch_bytes:self.scratch_bytes()};
        Ok((result,latents))
    }

    pub fn encode_pcm16k(&mut self,gpu:&VulkanContext,model:&GpuBuffer,acoustic:&GgufSummary,samples:&[f32],pad_side:AudioPadSide)->Result<(AudioEncodeStats,Vec<f32>)>{
        if samples.is_empty(){bail!("AudioVAE PCM input is empty");}
        if samples.iter().any(|x|!x.is_finite()){bail!("AudioVAE PCM input contains NaN/inf");}
        let aligned=align_samples(samples.to_vec(),self.config.input_samples_per_patch as usize,pad_side);
        let t0=Instant::now(); let cm=self.encode_samples(gpu,model,acoustic,&aligned)?;
        let frames=cm.len()/self.config.latent_dim as usize; let latents=channel_to_frame_major(&cm,self.config.latent_dim as usize,frames); let(checksum,l2)=stats(&latents);
        Ok((AudioEncodeStats{source_sample_rate:self.config.sample_rate,source_channels:1,source_samples_mono:samples.len(),resampled_samples:samples.len(),padded_samples:aligned.len(),pad_side,latent_frames:frames,latent_patches:frames/4,checksum,l2,encode_ms:t0.elapsed().as_secs_f64()*1000.0,scratch_bytes:self.scratch_bytes()},latents))
    }

    fn encode_samples(&mut self,gpu:&VulkanContext,model:&GpuBuffer,s:&GgufSummary,samples:&[f32])->Result<Vec<f32>>{
        if samples.len()%self.config.encoder_hop as usize!=0{bail!("AudioVAE encoder input must be aligned to {} samples",self.config.encoder_hop);}
        let max_elements=encoder_max_elements(samples.len(),self.config.encoder_dim as usize);
        if max_elements > u32::MAX as usize { bail!("AudioVAE encoder activation exceeds the current 32-bit shader index range"); }
        self.ensure_scratch(gpu,model,max_elements)?;
        let sc=self.scratch.as_ref().unwrap();
        let cmd=self.command_buffer; gpu.begin_one_time(cmd)?;
        let upload_staging=gpu.record_upload_f32(cmd,&sc.a,samples)?;
        let mut cur=Buf::A; let mut len=u32::try_from(samples.len()).context("AudioVAE encoder input exceeds u32 sample count")?; let mut channels=1u32;
        // stem A -> B
        record_conv(gpu,cmd,sc,s,cur,"audio_vae.encoder.block.0",channels,128,len,len,7,1,1,1,6)?; cur=cur.other(); channels=128;
        for (stage,&stride) in ENCODER_RATES.iter().enumerate(){
            let block=stage+1;
            for (ri,dilation) in [1u32,3,9].into_iter().enumerate(){
                let p=format!("audio_vae.encoder.block.{block}.block.{ri}.block");
                record_residual(gpu,cmd,sc,s,cur,&p,channels,len,dilation,channels)?;
            }
            let alpha=s.tensor(&format!("audio_vae.encoder.block.{block}.block.3.alpha"))?;
            record_snake(gpu,cmd,sc,cur,alpha,channels,len)?; let src=cur.other();
            let new_len=len/stride;
            record_conv(gpu,cmd,sc,s,src,&format!("audio_vae.encoder.block.{block}.block.4"),channels,channels*2,len,new_len,stride*2,stride,1,1,stride)?;
            // record_conv writes to opposite of src == cur
            channels*=2; len=new_len;
        }
        // fc_mu current -> opposite
        record_conv(gpu,cmd,sc,s,cur,"audio_vae.encoder.fc_mu",channels,self.config.latent_dim,len,len,3,1,1,1,2)?; cur=cur.other();
        let count=self.config.latent_dim as usize*len as usize;
        let out=gpu.submit_and_read_f32(cmd,buffer(sc,cur),count)?;
        drop(upload_staging);
        if out.iter().any(|x|!x.is_finite()){bail!("AudioVAE encoder produced NaN/inf");}
        Ok(out)
    }

    pub fn decode_latents(&mut self,gpu:&VulkanContext,model:&GpuBuffer,s:&GgufSummary,frame_major:&[f32])->Result<(AudioDecodeStats,Vec<f32>)>{
        let ld=self.config.latent_dim as usize;
        if frame_major.is_empty()||frame_major.len()%ld!=0{bail!("AudioVAE latent input must contain N*{ld} f32 values");}
        if frame_major.iter().any(|x|!x.is_finite()){bail!("AudioVAE latent input contains NaN/inf");}
        let frames=frame_major.len()/ld;
        if frames % self.config.patch_size as usize != 0 { bail!("VoxCPM2 AudioVAE decode expects latent frames in groups of {}, got {frames}", self.config.patch_size); }
        let cm=frame_to_channel_major(frame_major,ld,frames);
        let max_elements=decoder_max_elements(frames,self.config.decoder_dim as usize);
        if max_elements > u32::MAX as usize { bail!("AudioVAE decoder activation exceeds the current 32-bit shader index range"); }
        self.ensure_scratch(gpu,model,max_elements)?;
        let sc=self.scratch.as_ref().unwrap();
        let cmd=self.command_buffer; gpu.begin_one_time(cmd)?;
        let upload_staging=gpu.record_upload_f32(cmd,&sc.a,&cm)?;
        let mut cur=Buf::A; let mut len=u32::try_from(frames).context("AudioVAE latent frame count exceeds u32")?; let mut channels=self.config.latent_dim;
        record_conv(gpu,cmd,sc,s,cur,"audio_vae.decoder.model.0",channels,channels,len,len,7,1,1,channels,6)?; cur=cur.other();
        record_conv(gpu,cmd,sc,s,cur,"audio_vae.decoder.model.1",channels,self.config.decoder_dim,len,len,1,1,1,1,0)?; cur=cur.other(); channels=self.config.decoder_dim;
        let t0=Instant::now();
        for (i,&stride) in DECODER_RATES.iter().enumerate(){
            let mi=i+2; let out_channels=channels/2;
            record_scale(gpu,cmd,sc,s,cur,mi,channels,len,self.config.decoder_sr_bucket)?;
            let alpha=s.tensor(&format!("audio_vae.decoder.model.{mi}.block.0.alpha"))?;
            record_snake(gpu,cmd,sc,cur,alpha,channels,len)?; let src=cur.other();
            let new_len=len.checked_mul(stride).context("AudioVAE decoder length overflow")?;
            record_convt(gpu,cmd,sc,s,src,&format!("audio_vae.decoder.model.{mi}.block.1"),channels,out_channels,len,new_len,stride*2,stride,1)?;
            // output is opposite src == cur
            channels=out_channels; len=new_len;
            for (rj,dilation) in [1u32,3,9].into_iter().enumerate(){
                let bi=rj+2; let p=format!("audio_vae.decoder.model.{mi}.block.{bi}.block");
                record_residual(gpu,cmd,sc,s,cur,&p,channels,len,dilation,channels)?;
            }
        }
        let alpha=s.tensor("audio_vae.decoder.model.8.alpha")?;
        record_snake(gpu,cmd,sc,cur,alpha,channels,len)?; let src=cur.other();
        record_conv(gpu,cmd,sc,s,src,"audio_vae.decoder.model.9",channels,1,len,len,7,1,1,1,6)?; // output opposite(src) = cur
        record_tanh(gpu,cmd,sc,cur,len)?;
        let wave=gpu.submit_and_read_f32(cmd,buffer(sc,cur),len as usize)?;
        drop(upload_staging);
        if wave.iter().any(|x|!x.is_finite()){bail!("AudioVAE decoder produced NaN/inf");}
        let(checksum,l2)=stats(&wave); let peak=wave.iter().fold(0f32,|m,&v|m.max(v.abs()));
        let result=AudioDecodeStats{latent_frames:frames,latent_patches:frames/self.config.patch_size as usize,output_samples:wave.len(),output_sample_rate:self.config.out_sample_rate,duration_seconds:wave.len() as f64/self.config.out_sample_rate as f64,checksum,l2,peak,decode_ms:t0.elapsed().as_secs_f64()*1000.0,scratch_bytes:self.scratch_bytes()};
        Ok((result,wave))
    }
}

fn dtype(t:&TensorInfo)->Result<u32>{match t.ggml_type{GgmlType::F32=>Ok(0),GgmlType::F16=>Ok(1),_=>bail!("AudioVAE tensor {} has unsupported type {}",t.name,t.ggml_type.name())}}
fn off(t:&TensorInfo)->Result<u32>{u32::try_from(t.offset).with_context(||format!("tensor {} offset exceeds u32",t.name))}
fn buffer(sc:&Scratch,b:Buf)->&GpuBuffer{match b{Buf::A=>&sc.a,Buf::B=>&sc.b}}
fn groups256(n:u32)->u32{n/256+u32::from(n%256!=0)}
fn conv_pipe(sc:&Scratch,src:Buf)->&ComputePipeline{match src{Buf::A=>&sc.conv_ab,Buf::B=>&sc.conv_ba}}
fn convt_pipe(sc:&Scratch,src:Buf)->&ComputePipeline{match src{Buf::A=>&sc.convt_ab,Buf::B=>&sc.convt_ba}}
fn snake_pipe(sc:&Scratch,src:Buf)->&ComputePipeline{match src{Buf::A=>&sc.snake_ab,Buf::B=>&sc.snake_ba}}

fn record_conv(g:&VulkanContext,cmd:vk::CommandBuffer,sc:&Scratch,s:&GgufSummary,src:Buf,base:&str,in_c:u32,out_c:u32,in_len:u32,out_len:u32,kernel:u32,stride:u32,dilation:u32,groups:u32,left_pad:u32)->Result<()> {
    let w=s.tensor(base)?; let b=s.tensor(&format!("{base}.bias"))?; let p=conv_pipe(sc,src);
    let span=g.gpu_profile_begin(cmd,"audiovae.conv1d");
    p.bind(cmd);p.push(cmd,&ConvPush{weight_offset:off(w)?,bias_offset:off(b)?,in_channels:in_c,out_channels:out_c,in_length:in_len,out_length:out_len,kernel,stride,dilation,groups,left_pad,weight_dtype:dtype(w)?,bias_dtype:dtype(b)?,has_bias:1});
    let gx=groups256(out_len); if gx>g.info.max_compute_work_group_count_x{bail!("AudioVAE Conv1d dispatch x={gx} exceeds device limit {}",g.info.max_compute_work_group_count_x);} unsafe{g.device.cmd_dispatch(cmd,gx,out_c,1)};g.gpu_profile_end(cmd,span);g.compute_barrier(cmd);Ok(())
}
fn record_convt(g:&VulkanContext,cmd:vk::CommandBuffer,sc:&Scratch,s:&GgufSummary,src:Buf,base:&str,in_c:u32,out_c:u32,in_len:u32,out_len:u32,kernel:u32,stride:u32,groups:u32)->Result<()> {
    let w=s.tensor(base)?;let b=s.tensor(&format!("{base}.bias"))?;let p=convt_pipe(sc,src);
    let span=g.gpu_profile_begin(cmd,"audiovae.convtranspose1d");
    p.bind(cmd);p.push(cmd,&ConvTPush{weight_offset:off(w)?,bias_offset:off(b)?,in_channels:in_c,out_channels:out_c,in_length:in_len,out_length:out_len,kernel,stride,groups,weight_dtype:dtype(w)?,bias_dtype:dtype(b)?,has_bias:1});
    let gx=groups256(out_len); if gx>g.info.max_compute_work_group_count_x{bail!("AudioVAE ConvTranspose1d dispatch x={gx} exceeds device limit {}",g.info.max_compute_work_group_count_x);} unsafe{g.device.cmd_dispatch(cmd,gx,out_c,1)};g.gpu_profile_end(cmd,span);g.compute_barrier(cmd);Ok(())
}
fn record_snake(g:&VulkanContext,cmd:vk::CommandBuffer,sc:&Scratch,src:Buf,alpha:&TensorInfo,c:u32,len:u32)->Result<()> {
    let p=snake_pipe(sc,src);let span=g.gpu_profile_begin(cmd,"audiovae.snake");p.bind(cmd);p.push(cmd,&SnakePush{alpha_offset:off(alpha)?,channels:c,length:len,dtype:dtype(alpha)?});let gx=groups256(len); if gx>g.info.max_compute_work_group_count_x{bail!("AudioVAE Snake dispatch x={gx} exceeds device limit {}",g.info.max_compute_work_group_count_x);} unsafe{g.device.cmd_dispatch(cmd,gx,c,1)};g.gpu_profile_end(cmd,span);g.compute_barrier(cmd);Ok(())
}
fn snapshot(g:&VulkanContext,cmd:vk::CommandBuffer,sc:&Scratch,src:Buf,n:u32){
    g.compute_to_transfer_rw_barrier(cmd); unsafe{g.device.cmd_copy_buffer(cmd,buffer(sc,src).buffer,sc.c.buffer,&[vk::BufferCopy{src_offset:0,dst_offset:0,size:n as u64*4}]);} g.transfer_to_compute_barrier(cmd);
}
fn record_residual(g:&VulkanContext,cmd:vk::CommandBuffer,sc:&Scratch,s:&GgufSummary,cur:Buf,prefix:&str,c:u32,len:u32,dilation:u32,groups:u32)->Result<()> {
    snapshot(g,cmd,sc,cur,c*len);
    let a1=s.tensor(&format!("{prefix}.0.alpha"))?;record_snake(g,cmd,sc,cur,a1,c,len)?;let other=cur.other();
    record_conv(g,cmd,sc,s,other,&format!("{prefix}.1"),c,c,len,len,7,1,dilation,groups,6*dilation)?;
    let a2=s.tensor(&format!("{prefix}.2.alpha"))?;record_snake(g,cmd,sc,cur,a2,c,len)?;
    record_conv(g,cmd,sc,s,other,&format!("{prefix}.3"),c,c,len,len,1,1,1,1,0)?;
    let add=match cur{Buf::A=>&sc.add_a_c,Buf::B=>&sc.add_b_c};add.bind(cmd);add.push(cmd,&GridPush{channels:c,length:len});let gx=groups256(len);if gx>g.info.max_compute_work_group_count_x{bail!("AudioVAE residual-add dispatch x={gx} exceeds device limit {}",g.info.max_compute_work_group_count_x);}unsafe{g.device.cmd_dispatch(cmd,gx,c,1)};g.compute_barrier(cmd);Ok(())
}
fn record_scale(g:&VulkanContext,cmd:vk::CommandBuffer,sc:&Scratch,s:&GgufSummary,cur:Buf,mi:usize,c:u32,len:u32,bucket:u32)->Result<()> {
    let sw=s.tensor(&format!("audio_vae.decoder.sr_cond_model.{mi}.scale_embed"))?;
    let bw=s.tensor(&format!("audio_vae.decoder.sr_cond_model.{mi}.bias_embed"))?;
    let p=match cur{Buf::A=>&sc.scale_a,Buf::B=>&sc.scale_b};p.bind(cmd);p.push(cmd,&ScalePush{scale_offset:off(sw)?,bias_offset:off(bw)?,channels:c,length:len,bucket,buckets:DECODER_SR_BUCKETS,scale_dtype:dtype(sw)?,bias_dtype:dtype(bw)?});let gx=groups256(len);if gx>g.info.max_compute_work_group_count_x{bail!("AudioVAE scale/bias dispatch x={gx} exceeds device limit {}",g.info.max_compute_work_group_count_x);}unsafe{g.device.cmd_dispatch(cmd,gx,c,1)};g.compute_barrier(cmd);Ok(())
}
fn record_tanh(g:&VulkanContext,cmd:vk::CommandBuffer,sc:&Scratch,cur:Buf,n:u32)->Result<()> {let p=match cur{Buf::A=>&sc.tanh_a,Buf::B=>&sc.tanh_b};p.bind(cmd);p.push(cmd,&NPush{n});let gx=groups256(n);if gx>g.info.max_compute_work_group_count_x{bail!("AudioVAE tanh dispatch x={gx} exceeds device limit {}",g.info.max_compute_work_group_count_x);}unsafe{g.device.cmd_dispatch(cmd,gx,1,1)};g.compute_barrier(cmd);Ok(())}

fn encoder_max_elements(samples:usize,encoder_dim:usize)->usize{
    let mut max=samples; let mut len=samples; let mut c=encoder_dim; max=max.max(c*len);
    for &r in &ENCODER_RATES{len/=r as usize;c*=2;max=max.max(c*len);} max.max(64*len)
}
fn decoder_max_elements(frames:usize,decoder_dim:usize)->usize{
    let mut len=frames;let mut c=decoder_dim;let mut max=(64*frames).max(c*len);
    for &r in &DECODER_RATES{len*=r as usize;c/=2;max=max.max(c*len);}max.max(len)
}
fn channel_to_frame_major(x:&[f32],c:usize,t:usize)->Vec<f32>{let mut o=vec![0.0;c*t];for ti in 0..t{for ci in 0..c{o[ti*c+ci]=x[ci*t+ti];}}o}
fn frame_to_channel_major(x:&[f32],c:usize,t:usize)->Vec<f32>{let mut o=vec![0.0;c*t];for ti in 0..t{for ci in 0..c{o[ci*t+ti]=x[ti*c+ci];}}o}
fn align_samples(mut x:Vec<f32>,multiple:usize,side:AudioPadSide)->Vec<f32>{let rem=x.len()%multiple;if rem==0{return x;}let n=multiple-rem;match side{AudioPadSide::Right=>x.resize(x.len()+n,0.0),AudioPadSide::Left=>{let mut y=vec![0.0;n];y.extend_from_slice(&x);x=y;}}x}
fn stats(x:&[f32])->(f64,f64){let mut s=0.0;let mut q=0.0;for &v in x{let z=v as f64;s+=z;q+=z*z;}(s,q.sqrt())}

pub fn write_f32_file(path:&Path,values:&[f32])->Result<()> {let bytes=unsafe{std::slice::from_raw_parts(values.as_ptr() as *const u8,values.len()*4)};std::fs::write(path,bytes).with_context(||format!("write {}",path.display()))}
pub fn read_f32_file(path:&Path)->Result<Vec<f32>>{let b=std::fs::read(path).with_context(||format!("read {}",path.display()))?;if b.len()%4!=0{bail!("{} byte length is not divisible by 4",path.display());}Ok(b.chunks_exact(4).map(|z|f32::from_le_bytes(z.try_into().unwrap())).collect())}

pub fn write_wav_f32(path:&Path,samples:&[f32],sample_rate:u32)->Result<()> {
    let spec=hound::WavSpec{channels:1,sample_rate,bits_per_sample:32,sample_format:hound::SampleFormat::Float};
    let mut w=hound::WavWriter::create(path,spec).with_context(||format!("create WAV {}",path.display()))?;
    for &v in samples{w.write_sample(v.clamp(-1.0,1.0))?;}w.finalize()?;Ok(())
}

fn read_wav_mono(path:&Path)->Result<(Vec<f32>,u32,u16)> {
    let mut r=hound::WavReader::open(path).with_context(||format!("open WAV {}",path.display()))?;let spec=r.spec();let ch=spec.channels;
    if ch==0{bail!("WAV has zero channels");}
    if spec.sample_rate==0{bail!("WAV has zero sample rate");}
    let raw:Vec<f32>=match spec.sample_format{
        hound::SampleFormat::Float=>{if spec.bits_per_sample!=32{bail!("unsupported float WAV {}-bit",spec.bits_per_sample);}r.samples::<f32>().collect::<std::result::Result<Vec<_>,_>>()?},
        hound::SampleFormat::Int=>{
            let bits=spec.bits_per_sample;if bits==0||bits>32{bail!("unsupported PCM WAV bit depth {bits}");}
            let scale=(1u64<<(bits-1)) as f32;
            if bits<=8 { bail!("8-bit PCM WAV is not supported for VoxGen voice conditioning; convert it to 16/24/32-bit PCM"); }
            else if bits<=16 { r.samples::<i16>().map(|v|v.map(|z|z as f32/scale)).collect::<std::result::Result<Vec<_>,_>>()? }
            else { r.samples::<i32>().map(|v|v.map(|z|z as f32/scale)).collect::<std::result::Result<Vec<_>,_>>()? }
        }
    };
    if raw.len()%ch as usize!=0{bail!("WAV sample count is not divisible by channel count");}
    let mut mono=Vec::with_capacity(raw.len()/ch as usize);for frame in raw.chunks_exact(ch as usize){mono.push(frame.iter().copied().sum::<f32>()/ch as f32);}Ok((mono,spec.sample_rate,ch))
}

// Windowed-sinc host resampling is preprocessing only. All AudioVAE tensor inference stays on Vulkan.
fn resample_sinc(input:&[f32],src:u32,dst:u32)->Vec<f32>{
    if src==dst{return input.to_vec();}if input.is_empty(){return Vec::new();}
    let out_len=((input.len() as f64*dst as f64/src as f64).ceil() as usize).max(1);
    let ratio=src as f64/dst as f64;let cutoff=(dst as f64/src as f64).min(1.0)*0.94;let radius=24i32;
    let mut out=Vec::with_capacity(out_len);
    for j in 0..out_len{let pos=j as f64*ratio;let center=pos.floor() as i64;let mut sum=0.0f64;let mut norm=0.0f64;
        for k in -radius..=radius{let ix=center+k as i64;if ix<0||ix>=input.len() as i64{continue;}let d=pos-ix as f64;let ad=d.abs();if ad>radius as f64{continue;}let sinc=if d.abs()<1e-12{cutoff}else{(PI*cutoff*d).sin()/(PI*d)};let window=0.5+0.5*(PI*d/radius as f64).cos();let w=sinc*window;sum+=input[ix as usize] as f64*w;norm+=w;}
        out.push(if norm.abs()>1e-12{(sum/norm) as f32}else{0.0});
    }out
}
