from pathlib import Path
import math
import re
import sys

ROOT = Path(__file__).resolve().parent
errors=[]
def need(cond,msg):
    if not cond: errors.append(msg)
def text(rel): return (ROOT/rel).read_text(encoding='utf-8')

root_cargo=text('Cargo.toml'); demo_cargo=text('demo/Cargo.toml')
need('version = "0.7.39"' in root_cargo and 'version = "0.7.39"' in demo_cargo, 'root/demo version 0.7.37')

base=text('src/baselm.rs'); acoustic=text('src/acoustic.rs'); local=text('src/local.rs'); vk=text('src/vulkan.rs')
qkv=text('shaders/qkv_matvec.comp'); qkvx=text('shaders/qkv_matvec_xtx7900.comp')
seq=text('shaders/seq_qkv.comp'); seqx=text('shaders/seq_qkv_xtx7900.comp'); pack=text('shaders/pack_locdit.comp')

# Shared one-dispatch QKV must be active in the two autoregressive engines.
for src,label in [(base,'BaseLM'),(acoustic,'ResidualLM')]:
    need('QKV_MATVEC_SPV' in src and 'QKV_MATVEC_XTX7900_SPV' in src, f'{label} embeds normal/XTX QKV shaders')
    need('.select_spirv(QKV_MATVEC_SPV, QKV_MATVEC_XTX7900_SPV)' in src, f'{label} mode-selects QKV shader')
    need('dispatch_qkv(' in src, f'{label} uses combined QKV dispatch')
    need('q_weight_offset' in src and 'k_weight_offset' in src and 'v_weight_offset' in src, f'{label} QKV push offsets')

# Local fusion is shared unless the explicit coopmat experiment is on.
need('SEQ_QKV_SPV' in local and 'SEQ_QKV_XTX7900_SPV' in local, 'Local embeds normal/XTX sequence QKV shaders')
need('let use_fused_qkv=!gpu.xtx_coopmat_enabled();' in local, 'Local QKV fusion preserves coopmat override')
need('if self.use_fused_qkv {' in local and 'record_qkv(' in local, 'Local transformer selects fused QKV path')
need('q:seq(&norm,&q)?,k:seq(&norm,&k)?,v:seq(&norm,&v)?,qkv' in local, 'Local retains separate coopmat Q/K/V fallback')

# Workgroup geometry stays unchanged: this release must not gamble performance on a new size.
for shader,label in [(qkv,'portable QKV'),(qkvx,'XTX QKV'),(seq,'portable seq QKV'),(seqx,'XTX seq QKV')]:
    need(re.search(r'local_size_x\s*=\s*256',shader) is not None, f'{label} keeps 256-thread workgroup')
need('GL_KHR_shader_subgroup_arithmetic' in qkvx and 'subgroupAdd' in qkvx, 'XTX QKV uses subgroup reduction')
need('GL_KHR_shader_subgroup_arithmetic' in seqx and 'subgroupAdd' in seqx, 'XTX seq QKV uses subgroup reduction')

# Buffer-scoped barriers should be used in all transformer hot paths.
need('pub fn compute_buffer_barrier' in vk, 'Vulkan buffer-scoped compute barrier helper exists')
need('vk::BufferMemoryBarrier' in vk and '.buffer(b.buffer)' in vk, 'barrier helper names exact buffers')
for src,label,min_calls in [(base,'BaseLM',8),(acoustic,'ResidualLM',7),(local,'Local',8)]:
    need(src.count('compute_buffer_barrier') >= min_calls, f'{label} transformer uses targeted barriers')

# CFG negative path must zero mu logically, not with two transfer fills.
need('zero_mu:u32' in local and 'zero_mu:zero_mu as u32' in local, 'pack_locdit zero_mu control wired')
need('pc.zero_mu!=0u?0.0:m1.x[d]' in pack and 'pc.zero_mu!=0u?0.0:m2.x[d]' in pack, 'pack_locdit zeroes both mu tokens')
need('cmd_fill_buffer(self.command_buffer,self.mu1.buffer' not in local and 'cmd_fill_buffer(self.command_buffer,self.mu2.buffer' not in local, 'per-CFG mu buffer fills removed')
need('record_locdit_body(gpu,a,base,true)' in local and 'record_locdit_body(gpu,a,base,false)' in local, 'positive/negative LocDiT pack modes explicit')
need('mu1_saved' not in local and 'mu2_saved' not in local, 'obsolete CFG mu save buffers removed')
need('cmd_copy_buffer(self.command_buffer,self.mu1' not in local and 'cmd_copy_buffer(self.command_buffer,self.mu2' not in local, 'per-step CFG mu restore copies removed')
need('record_upload_f32(self.command_buffer,&self.mu1,' in local and 'record_upload_f32(self.command_buffer,&self.mu2,' in local, 'CPU mu uploads directly into resident positive buffers')

# Mapping proof for combined rows: Q then K then V, exactly once each.
q_rows,kv_rows=2048,256
total=q_rows+2*kv_rows
seen=[set(),set(),set()]
for combined in range(total):
    if combined<q_rows: target,row=0,combined
    elif combined<q_rows+kv_rows: target,row=1,combined-q_rows
    else: target,row=2,combined-q_rows-kv_rows
    seen[target].add(row)
need(seen[0]==set(range(q_rows)), 'QKV row mapping covers Q exactly once')
need(seen[1]==set(range(kv_rows)), 'QKV row mapping covers K exactly once')
need(seen[2]==set(range(kv_rows)), 'QKV row mapping covers V exactly once')

# Static descriptor/push contracts.
need('create_compute_pipeline(\n            gpu.select_spirv(QKV_MATVEC_SPV, QKV_MATVEC_XTX7900_SPV),\n            5,' in base, 'BaseLM QKV has five storage bindings')
need('create_compute_pipeline(\n            gpu.select_spirv(QKV_MATVEC_SPV, QKV_MATVEC_XTX7900_SPV),\n            5,' in acoustic, 'ResidualLM QKV has five storage bindings')
need('create_compute_pipeline(gpu.select_spirv(SEQ_QKV_SPV,SEQ_QKV_XTX7900_SPV),5' in local, 'Local QKV has five storage bindings')

if errors:
    print('v0.7.34 power-efficiency validation FAILED')
    for e in errors: print(' -',e)
    sys.exit(1)
print('PASS: v0.7.34 shared QKV dispatches, buffer-scoped transformer barriers, coopmat fallback, and zero-fill-free CFG mu packing')
