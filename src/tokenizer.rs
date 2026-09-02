use crate::gguf::GgufSummary;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize)]
pub struct TokenizerInfo {
    pub vocab_size: usize,
    pub merge_count: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub unk_token_id: u32,
    pub add_bos: bool,
    pub add_eos: bool,
    pub cjk_split_entries: usize,
}

#[derive(Debug, Clone)]
pub struct VoxTokenizer {
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
    merge_rank: HashMap<(String, String), usize>,
    byte_ids: [Option<u32>; 256],
    bos_token_id: u32,
    eos_token_id: u32,
    unk_token_id: u32,
    add_bos: bool,
    add_eos: bool,
    cjk_expansion: HashMap<u32, Vec<u32>>,
}

impl VoxTokenizer {
    pub fn from_gguf(s: &GgufSummary) -> Result<Self> {
        if s.tokenizer_tokens.is_empty() {
            bail!("BaseLM GGUF does not contain tokenizer.ggml.tokens; VoxGen step 7 requires the tokenizer embedded by the VoxCPM2 converter");
        }
        let mut ids = HashMap::with_capacity(s.tokenizer_tokens.len());
        for (i, tok) in s.tokenizer_tokens.iter().enumerate() {
            ids.entry(tok.clone()).or_insert(i as u32);
        }
        let mut merge_rank = HashMap::with_capacity(s.tokenizer_merges.len());
        for (rank, m) in s.tokenizer_merges.iter().enumerate() {
            if let Some((a,b)) = m.split_once(' ') {
                merge_rank.entry((a.to_owned(), b.to_owned())).or_insert(rank);
            }
        }
        let mut byte_ids = [None; 256];
        for b in 0..=255u16 {
            let key = format!("<0x{:02X}>", b);
            byte_ids[b as usize] = ids.get(&key).copied();
        }
        let meta_u32 = |key: &str, default: u32| -> u32 {
            s.metadata.get(key).and_then(|v| v.parse::<u32>().ok()).unwrap_or(default)
        };
        let meta_bool = |key: &str, default: bool| -> bool {
            s.metadata.get(key).and_then(|v| match v.as_str(){"true"|"1"=>Some(true),"false"|"0"=>Some(false),_=>None}).unwrap_or(default)
        };
        let bos_token_id = meta_u32("tokenizer.ggml.bos_token_id", 1);
        let eos_token_id = meta_u32("tokenizer.ggml.eos_token_id", 2);
        let unk_token_id = meta_u32("tokenizer.ggml.unknown_token_id", 0);
        let add_bos = meta_bool("tokenizer.ggml.add_bos_token", true);
        let add_eos = meta_bool("tokenizer.ggml.add_eos_token", false);

        let mut cjk_expansion = HashMap::new();
        for (id, raw) in s.tokenizer_tokens.iter().enumerate() {
            let clean = raw.strip_prefix('▁').unwrap_or(raw);
            let chars: Vec<char> = clean.chars().collect();
            if chars.len() < 2 || !chars.iter().all(|&c| is_cjk(c)) { continue; }
            let mut expanded = Vec::with_capacity(chars.len());
            let mut ok = true;
            for c in chars {
                let one = c.to_string();
                if let Some(&cid) = ids.get(&one).or_else(|| ids.get(&format!("▁{one}"))) {
                    expanded.push(cid);
                } else { ok = false; break; }
            }
            if ok { cjk_expansion.insert(id as u32, expanded); }
        }

        Ok(Self { tokens:s.tokenizer_tokens.clone(), ids, merge_rank, byte_ids, bos_token_id, eos_token_id, unk_token_id, add_bos, add_eos, cjk_expansion })
    }

    pub fn info(&self) -> TokenizerInfo {
        TokenizerInfo { vocab_size:self.tokens.len(), merge_count:self.merge_rank.len(), bos_token_id:self.bos_token_id, eos_token_id:self.eos_token_id, unk_token_id:self.unk_token_id, add_bos:self.add_bos, add_eos:self.add_eos, cjk_split_entries:self.cjk_expansion.len() }
    }

    pub fn encode(&self, text:&str) -> Result<Vec<u32>> {
        if text.is_empty() { bail!("VoxGen cannot synthesize empty text"); }
        // VoxCPM2 tokenizer.json normalizer: Prepend ▁, then replace literal spaces with ▁.
        let normalized = format!("▁{}", text.replace(' ', "▁"));
        let mut symbols: Vec<String> = Vec::new();
        for ch in normalized.chars() {
            let s = ch.to_string();
            if self.ids.contains_key(&s) {
                symbols.push(s);
            } else {
                for b in s.as_bytes() {
                    let id = self.byte_ids[*b as usize].with_context(|| format!("tokenizer has no direct token for {ch:?} and no <0x{:02X}> byte fallback", b))?;
                    symbols.push(self.tokens[id as usize].clone());
                }
            }
        }
        while symbols.len() > 1 {
            let mut best: Option<(usize,usize)> = None;
            for i in 0..symbols.len()-1 {
                if let Some(&rank) = self.merge_rank.get(&(symbols[i].clone(), symbols[i+1].clone())) {
                    if best.map(|(_,r)|rank<r).unwrap_or(true) { best=Some((i,rank)); }
                }
            }
            let Some((i,_))=best else { break; };
            let merged = format!("{}{}",symbols[i],symbols[i+1]);
            symbols.splice(i..=i+1,[merged]);
        }
        let mut raw_ids=Vec::with_capacity(symbols.len()+2);
        if self.add_bos { raw_ids.push(self.bos_token_id); }
        for sym in symbols {
            let id=self.ids.get(&sym).copied().unwrap_or(self.unk_token_id);
            raw_ids.push(id);
        }
        if self.add_eos { raw_ids.push(self.eos_token_id); }
        // VoxCPM2 post-tokenization rule: split multi-character CJK vocabulary tokens.
        let mut out=Vec::with_capacity(raw_ids.len());
        for id in raw_ids {
            if let Some(expanded)=self.cjk_expansion.get(&id) { out.extend_from_slice(expanded); }
            else { out.push(id); }
        }
        Ok(out)
    }
}

fn is_cjk(c:char)->bool {
    matches!(c as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF |
        0x20000..=0x2FA1F | 0x3040..=0x30FF | 0xAC00..=0xD7AF)
}

#[cfg(test)]
mod tests { use super::*; #[test] fn cjk_ranges(){assert!(is_cjk('你'));assert!(is_cjk('語'));assert!(!is_cjk('A'));} }
