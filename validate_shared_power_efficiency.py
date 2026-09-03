from pathlib import Path
import math
import random

root = Path(__file__).resolve().parent
errors = []

def need(cond, msg):
    if not cond:
        errors.append(msg)

root_cargo = (root/'Cargo.toml').read_text()
demo_cargo = (root/'demo'/'Cargo.toml').read_text()
base = (root/'src'/'baselm.rs').read_text()
acoustic = (root/'src'/'acoustic.rs').read_text()
local = (root/'src'/'local.rs').read_text()

need('version = "0.7.39"' in root_cargo and 'version = "0.7.39"' in demo_cargo, 'root/demo version 0.7.37')

for name in [
    'residual_rmsnorm.comp', 'residual_rmsnorm_xtx7900.comp',
    'matvec_swiglu.comp', 'matvec_swiglu_xtx7900.comp',
    'seq_residual_rmsnorm.comp', 'seq_residual_rmsnorm_xtx7900.comp',
    'seq_swiglu.comp', 'seq_swiglu_xtx7900.comp',
]:
    need((root/'shaders'/name).exists(), f'missing shader {name}')

for text, label in [(base,'BaseLM'), (acoustic,'ResidualLM')]:
    need('select_spirv(RESIDUAL_RMSNORM_SPV, RESIDUAL_RMSNORM_XTX7900_SPV)' in text, f'{label} does not select portable/XTX residual+RMS shader')
    need('select_spirv(SWIGLU_SPV, SWIGLU_XTX7900_SPV)' in text, f'{label} does not select portable/XTX SwiGLU shader')
    need('.dispatch_residual_rms' in text or 'dispatch_residual_rms(' in text, f'{label} fused residual+RMS dispatch missing')
    need('dispatch_swiglu(' in text, f'{label} fused Gate+Up+SwiGLU dispatch missing')
    need('silu_mul.bind' not in text, f'{label} still dispatches standalone SiLU multiply')
    need('pipelines.residual.bind' not in text, f'{label} still dispatches standalone residual add')

need('up: GpuBuffer' not in base and 'up: GpuBuffer' not in acoustic, 'BaseLM/ResidualLM still allocate the eliminated up intermediate')
need('use_fused_swiglu=!gpu.xtx_coopmat_enabled()' in local, 'LocalDiT does not preserve explicit coopmat fallback')
need('if self.use_fused_swiglu' in local and 'self.pipes.tr.silu_mul.bind' in local, 'LocalDiT fused/default and legacy coopmat branches are not both present')
need('select_spirv(SEQ_RESIDUAL_RMS_SPV,SEQ_RESIDUAL_RMS_XTX7900_SPV)' in local, 'LocalDiT residual+RMS mode selection missing')
need('select_spirv(SEQ_SWIGLU_SPV,SEQ_SWIGLU_XTX7900_SPV)' in local, 'LocalDiT SwiGLU mode selection missing')
need('let next_norm = if layer + 1 < self.config.block_count' in base, 'BaseLM does not chain final residual into next attention norm')
need('let next_norm = if layer + 1 < self.config.residual_block_count' in acoustic, 'ResidualLM does not chain final residual into next attention norm')
need('let next_norm=if layer+1<layers' in local, 'LocalDiT does not chain final residual into next attention norm')

# Numerical identity checks for the two fusions, independent of Vulkan.
rng = random.Random(733)
n = 37
hidden = [rng.uniform(-2, 2) for _ in range(n)]
branch = [rng.uniform(-2, 2) for _ in range(n)]
weights = [rng.uniform(0.5, 1.5) for _ in range(n)]
scale = 0.73
eps = 1e-5
old_hidden = [h + scale*b for h,b in zip(hidden, branch)]
old_inv = 1.0 / math.sqrt(sum(v*v for v in old_hidden)/n + eps)
old_norm = [v*old_inv*w for v,w in zip(old_hidden, weights)]
fused_hidden = [h + scale*b for h,b in zip(hidden, branch)]
fused_inv = 1.0 / math.sqrt(sum(v*v for v in fused_hidden)/n + eps)
fused_norm = [v*fused_inv*w for v,w in zip(fused_hidden, weights)]
need(max(abs(a-b) for a,b in zip(old_norm, fused_norm)) < 1e-12, 'residual+RMS mathematical identity failed')

rows, cols = 11, 17
x = [rng.uniform(-1,1) for _ in range(cols)]
gw = [[rng.uniform(-.5,.5) for _ in range(cols)] for _ in range(rows)]
uw = [[rng.uniform(-.5,.5) for _ in range(cols)] for _ in range(rows)]
def dot(w): return sum(a*b for a,b in zip(w,x))
def silu(v): return v/(1.0+math.exp(-v))
old = [silu(dot(g))*dot(u) for g,u in zip(gw,uw)]
fused = []
for g,u in zip(gw,uw):
    gs = us = 0.0
    for c in range(cols):
        xv=x[c]; gs += g[c]*xv; us += u[c]*xv
    fused.append(silu(gs)*us)
need(max(abs(a-b) for a,b in zip(old,fused)) < 1e-12, 'Gate+Up+SwiGLU mathematical identity failed')

if errors:
    print('shared power-efficiency validation FAILED')
    for e in errors:
        print(' -', e)
    raise SystemExit(1)
print('PASS: v0.7.34 shared Normal/XTX residual+RMS and Gate+Up+SwiGLU fusions; LocalDiT coopmat fallback preserved')
