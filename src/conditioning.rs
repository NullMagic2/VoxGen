use anyhow::{bail, Result};
use serde::Serialize;

pub const AUDIO_START_TOKEN: u32 = 101;
pub const REF_AUDIO_START_TOKEN: u32 = 103;
pub const REF_AUDIO_END_TOKEN: u32 = 104;
pub const PATCH_FLOATS: usize = 4 * 64;

#[derive(Debug, Clone)]
pub enum PrefixPosition {
    Text(u32),
    Audio(Vec<f32>),
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditioningMode { ZeroShot, Reference, Continuation, ReferenceContinuation }

#[derive(Debug, Clone, Serialize)]
pub struct ConditioningPlanSummary {
    pub mode: ConditioningMode,
    pub total_positions: usize,
    pub text_positions: usize,
    pub audio_positions: usize,
    pub text_mask: Vec<u8>,
    pub feat_mask: Vec<u8>,
    pub prefix_condition_is_audio: bool,
    pub reference_patches: usize,
    pub prompt_patches: usize,
}

pub struct ConditioningPlan {
    pub mode: ConditioningMode,
    pub positions: Vec<PrefixPosition>,
    pub prefix_condition: Vec<f32>,
    pub summary: ConditioningPlanSummary,
}

pub fn split_patches(values: Vec<f32>, label: &str) -> Result<Vec<Vec<f32>>> {
    if values.len() % PATCH_FLOATS != 0 {
        bail!("{label} contains {} floats; latent conditioning files must contain an integer number of 4x64 (=256-float) patches", values.len());
    }
    Ok(values.chunks(PATCH_FLOATS).map(|x| x.to_vec()).collect())
}

pub fn build_plan(text_tokens: &[u32], reference: &[Vec<f32>], prompt: &[Vec<f32>]) -> Result<ConditioningPlan> {
    if text_tokens.is_empty() { bail!("conditioning text token list is empty"); }
    for (kind, xs) in [("reference", reference), ("prompt", prompt)] {
        for (i,p) in xs.iter().enumerate() {
            if p.len()!=PATCH_FLOATS { bail!("{kind} patch {i} has {} floats, expected {PATCH_FLOATS}",p.len()); }
            if let Some((j,x))=p.iter().copied().enumerate().find(|(_,x)|!x.is_finite()) { bail!("{kind} patch {i} contains non-finite value at index {j}: {x}"); }
        }
    }
    let mode = match (!reference.is_empty(), !prompt.is_empty()) {
        (false,false)=>ConditioningMode::ZeroShot,
        (true,false)=>ConditioningMode::Reference,
        (false,true)=>ConditioningMode::Continuation,
        (true,true)=>ConditioningMode::ReferenceContinuation,
    };
    let mut positions=Vec::new();
    if !reference.is_empty() {
        positions.push(PrefixPosition::Text(REF_AUDIO_START_TOKEN));
        positions.extend(reference.iter().cloned().map(PrefixPosition::Audio));
        positions.push(PrefixPosition::Text(REF_AUDIO_END_TOKEN));
    }
    positions.extend(text_tokens.iter().copied().map(PrefixPosition::Text));
    positions.push(PrefixPosition::Text(AUDIO_START_TOKEN));
    positions.extend(prompt.iter().cloned().map(PrefixPosition::Audio));

    let text_mask=positions.iter().map(|p|matches!(p,PrefixPosition::Text(_)) as u8).collect::<Vec<_>>();
    let feat_mask=positions.iter().map(|p|matches!(p,PrefixPosition::Audio(_)) as u8).collect::<Vec<_>>();
    // Official VoxCPM2 uses feat[:, -1]. Reference-only/zero-shot prefixes end in
    // AUDIO_START (zero feature); continuation prefixes end in the final prompt patch.
    let prefix_condition = match positions.last() {
        Some(PrefixPosition::Audio(x)) => x.clone(),
        _ => vec![0.0; PATCH_FLOATS],
    };
    let summary=ConditioningPlanSummary{
        mode,total_positions:positions.len(),text_positions:text_mask.iter().map(|&x|x as usize).sum(),audio_positions:feat_mask.iter().map(|&x|x as usize).sum(),
        text_mask,feat_mask,prefix_condition_is_audio:matches!(positions.last(),Some(PrefixPosition::Audio(_))),reference_patches:reference.len(),prompt_patches:prompt.len(),
    };
    Ok(ConditioningPlan{mode,positions,prefix_condition,summary})
}
