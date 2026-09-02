use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde::Serialize;
use std::{collections::BTreeMap, fs::File, path::{Path, PathBuf}};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BaseFormat {
    Q8_0,
    F16,
}

impl BaseFormat {
    pub fn as_str(self) -> &'static str {
        match self { Self::Q8_0 => "q8_0", Self::F16 => "f16" }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GgmlType {
    F32,
    F16,
    Q8_0,
    Other(u32),
}

impl GgmlType {
    pub fn from_raw(v: u32) -> Self {
        match v { 0 => Self::F32, 1 => Self::F16, 8 => Self::Q8_0, other => Self::Other(other) }
    }
    pub fn name(self) -> String {
        match self { Self::F32 => "F32".into(), Self::F16 => "F16".into(), Self::Q8_0 => "Q8_0".into(), Self::Other(v) => format!("GGML_TYPE_{v}") }
    }
    pub fn storage_bytes(self, elements: u64) -> Result<u64> {
        match self {
            Self::F32 => elements.checked_mul(4).context("F32 tensor size overflow"),
            Self::F16 => elements.checked_mul(2).context("F16 tensor size overflow"),
            // ggml Q8_0: blocks of 32 values = fp16 scale (2 B) + 32 int8 quants.
            Self::Q8_0 => {
                if elements % 32 != 0 { bail!("Q8_0 tensor has {elements} elements, not divisible by 32"); }
                (elements / 32).checked_mul(34).context("Q8_0 tensor size overflow")
            }
            Self::Other(v) => bail!("unsupported GGML tensor type {v}"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
    pub elements: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GgufSummary {
    pub path: PathBuf,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub alignment: u64,
    pub data_offset: u64,
    pub file_bytes: u64,
    pub tensor_bytes: u64,
    pub dtype_counts: BTreeMap<String, u64>,
    pub metadata: BTreeMap<String, String>,
    #[serde(skip)]
    pub tensors: Vec<TensorInfo>,
    /// Token strings embedded by the VoxCPM2 GGUF converter. Kept out of diagnostics JSON.
    #[serde(skip)]
    pub tokenizer_tokens: Vec<String>,
    /// BPE merge rules embedded by the VoxCPM2 GGUF converter.
    #[serde(skip)]
    pub tokenizer_merges: Vec<String>,
}

impl GgufSummary {
    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
            .with_context(|| format!("required GGUF tensor {name:?} is missing from {}", self.path.display()))
    }

    pub fn metadata_str(&self, key: &str) -> Result<&str> {
        self.metadata.get(key).map(String::as_str)
            .with_context(|| format!("required GGUF metadata key {key:?} is missing from {}", self.path.display()))
    }

    pub fn metadata_u32(&self, key: &str) -> Result<u32> {
        self.metadata_str(key)?.parse::<u32>()
            .with_context(|| format!("GGUF metadata {key:?} is not a u32"))
    }

    pub fn metadata_f32(&self, key: &str) -> Result<f32> {
        self.metadata_str(key)?.parse::<f32>()
            .with_context(|| format!("GGUF metadata {key:?} is not an f32"))
    }

    pub fn metadata_bool(&self, key: &str) -> Result<bool> {
        match self.metadata_str(key)? {
            "true" | "1" => Ok(true),
            "false" | "0" => Ok(false),
            _ => bail!("GGUF metadata {key:?} is not a bool"),
        }
    }

    pub fn data_bytes(&self) -> Result<u64> {
        self.file_bytes.checked_sub(self.data_offset)
            .context("GGUF data offset exceeds file length")
    }

    pub fn primary_base_format(&self) -> Result<BaseFormat> {
        // Look only at sizeable multi-dimensional tensors. Norm/scalar tensors are often F32
        // even in quantized GGUFs and must not make a Q8 model look "mixed".
        let mut q8_bytes = 0u64;
        let mut f16_bytes = 0u64;
        for t in &self.tensors {
            if t.dims.len() < 2 || t.elements < 4096 { continue; }
            match t.ggml_type {
                GgmlType::Q8_0 => q8_bytes = q8_bytes.saturating_add(t.bytes),
                GgmlType::F16 => f16_bytes = f16_bytes.saturating_add(t.bytes),
                _ => {}
            }
        }
        if q8_bytes > 0 && q8_bytes >= f16_bytes { return Ok(BaseFormat::Q8_0); }
        if f16_bytes > 0 { return Ok(BaseFormat::F16); }
        bail!("could not identify BaseLM as Q8_0 or F16 from the tensor table")
    }

    pub fn validate_baselm(&self, requested: Option<BaseFormat>) -> Result<BaseFormat> {
        for t in &self.tensors {
            if !matches!(t.ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::Q8_0) {
                bail!("Unsupported BaseLM tensor type {} in {}", t.ggml_type.name(), t.name);
            }
        }
        let detected = self.primary_base_format()?;
        if let Some(req) = requested {
            if detected != req {
                bail!("BaseLM format mismatch: requested {}, GGUF tensor table looks like {}", req.as_str(), detected.as_str());
            }
        }
        Ok(detected)
    }

    pub fn validate_acoustic_f16(&self) -> Result<()> {
        for t in &self.tensors {
            if !matches!(t.ggml_type, GgmlType::F32 | GgmlType::F16) {
                bail!("Unsupported acoustic tensor type {} in {}. VoxGen currently requires the F16 acoustic GGUF.", t.ggml_type.name(), t.name);
            }
        }
        if !self.tensors.iter().any(|t| t.ggml_type == GgmlType::F16 && t.elements >= 4096) {
            bail!("acoustic GGUF does not contain expected F16 weight tensors");
        }
        Ok(())
    }
}

struct Reader<'a> { bytes: &'a [u8], pos: usize }
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, pos: 0 } }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("GGUF offset overflow")?;
        if end > self.bytes.len() { bail!("unexpected EOF while reading GGUF at byte {}", self.pos); }
        let s = &self.bytes[self.pos..end]; self.pos = end; Ok(s)
    }
    fn u8(&mut self) -> Result<u8> { Ok(self.take(1)?[0]) }
    fn i8(&mut self) -> Result<i8> { Ok(self.u8()? as i8) }
    fn u16(&mut self) -> Result<u16> { Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap())) }
    fn i16(&mut self) -> Result<i16> { Ok(i16::from_le_bytes(self.take(2)?.try_into().unwrap())) }
    fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn i32(&mut self) -> Result<i32> { Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn i64(&mut self) -> Result<i64> { Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn f32(&mut self) -> Result<f32> { Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn f64(&mut self) -> Result<f64> { Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn string(&mut self) -> Result<String> {
        let n = usize::try_from(self.u64()?).context("GGUF string too large")?;
        Ok(std::str::from_utf8(self.take(n)?).context("invalid UTF-8 in GGUF string")?.to_owned())
    }
}

#[derive(Debug)]
enum MetaValue { Text(String), U64(u64), StringArray(Vec<String>), Other }

fn read_meta_value(r: &mut Reader<'_>, ty: u32, depth: usize) -> Result<MetaValue> {
    if depth > 8 { bail!("GGUF metadata nesting too deep"); }
    Ok(match ty {
        0 => MetaValue::Text(r.u8()?.to_string()),
        1 => MetaValue::Text(r.i8()?.to_string()),
        2 => MetaValue::Text(r.u16()?.to_string()),
        3 => MetaValue::Text(r.i16()?.to_string()),
        4 => { let v = r.u32()? as u64; MetaValue::U64(v) },
        5 => MetaValue::Text(r.i32()?.to_string()),
        6 => MetaValue::Text(r.f32()?.to_string()),
        7 => MetaValue::Text((r.u8()? != 0).to_string()),
        8 => MetaValue::Text(r.string()?),
        9 => {
            let elem_ty = r.u32()?;
            let n = r.u64()?;
            if elem_ty == 8 {
                let cap = usize::try_from(n.min(1_000_000)).context("GGUF string array too large")?;
                let mut out = Vec::with_capacity(cap);
                for _ in 0..n { out.push(r.string()?); }
                MetaValue::StringArray(out)
            } else {
                for _ in 0..n { let _ = read_meta_value(r, elem_ty, depth + 1)?; }
                MetaValue::Other
            }
        }
        10 => { let v = r.u64()?; MetaValue::U64(v) },
        11 => MetaValue::Text(r.i64()?.to_string()),
        12 => MetaValue::Text(r.f64()?.to_string()),
        _ => bail!("unknown GGUF metadata value type {ty}"),
    })
}

pub fn load_summary(path: &Path) -> Result<GgufSummary> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file).with_context(|| format!("mmap {}", path.display()))? };
    let mut r = Reader::new(&mmap);
    if r.u32()? != GGUF_MAGIC { bail!("{} is not a GGUF file", path.display()); }
    let version = r.u32()?;
    if !(2..=3).contains(&version) { bail!("unsupported GGUF version {version}; expected v2/v3"); }
    let tensor_count = r.u64()?;
    let metadata_count = r.u64()?;
    let mut metadata = BTreeMap::new();
    let mut tokenizer_tokens = Vec::new();
    let mut tokenizer_merges = Vec::new();
    let mut alignment = DEFAULT_ALIGNMENT;
    for _ in 0..metadata_count {
        let key = r.string()?;
        let ty = r.u32()?;
        let value = read_meta_value(&mut r, ty, 0)?;
        match value {
            MetaValue::Text(v) => { if key.len() <= 128 && v.len() <= 512 { metadata.insert(key, v); } },
            MetaValue::U64(v) => {
                if key == "general.alignment" { alignment = v.max(1); }
                metadata.insert(key, v.to_string());
            }
            MetaValue::StringArray(v) => {
                match key.as_str() {
                    "tokenizer.ggml.tokens" => tokenizer_tokens = v,
                    "tokenizer.ggml.merges" => tokenizer_merges = v,
                    _ => {}
                }
            }
            MetaValue::Other => {}
        }
    }

    let mut tensors = Vec::with_capacity(usize::try_from(tensor_count.min(200_000)).unwrap_or(0));
    let mut dtype_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut tensor_bytes = 0u64;
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()?;
        if n_dims == 0 || n_dims > 8 { bail!("invalid tensor rank {n_dims} for {name}"); }
        let mut dims = Vec::with_capacity(n_dims as usize);
        let mut elements = 1u64;
        for _ in 0..n_dims {
            let d = r.u64()?;
            elements = elements.checked_mul(d).with_context(|| format!("element-count overflow for {name}"))?;
            dims.push(d);
        }
        let ggml_type = GgmlType::from_raw(r.u32()?);
        let offset = r.u64()?;
        let bytes = ggml_type.storage_bytes(elements).with_context(|| format!("tensor {name}"))?;
        *dtype_counts.entry(ggml_type.name()).or_default() += 1;
        tensor_bytes = tensor_bytes.saturating_add(bytes);
        tensors.push(TensorInfo { name, dims, ggml_type, offset, elements, bytes });
    }

    let pos = r.pos as u64;
    let data_offset = ((pos + alignment - 1) / alignment) * alignment;
    let file_bytes = mmap.len() as u64;
    for t in &tensors {
        let end = data_offset.checked_add(t.offset).and_then(|x| x.checked_add(t.bytes)).context("tensor absolute offset overflow")?;
        if end > file_bytes { bail!("tensor {} extends beyond end of {}", t.name, path.display()); }
    }
    Ok(GgufSummary { path: path.to_path_buf(), version, tensor_count, metadata_count, alignment, data_offset, file_bytes, tensor_bytes, dtype_counts, metadata, tensors, tokenizer_tokens, tokenizer_merges })
}
