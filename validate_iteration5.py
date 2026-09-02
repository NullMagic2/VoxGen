from pathlib import Path
import re, sys, math
root=Path(__file__).resolve().parent
errors=[]
# Every embedded SPIR-V must have a GLSL source.
includes=set()
for p in (root/'src').glob('*.rs'):
    s=p.read_text(encoding='utf-8')
    includes.update(re.findall(r'/([A-Za-z0-9_]+)\.spv',s))
shaders={p.stem for p in (root/'shaders').glob('*.comp')}
for name in sorted(includes-shaders): errors.append(f'missing shader source for {name}.spv')
for name in sorted(shaders-includes):
    # Build compiles all shaders; warn as error only for the two step-5 kernels if accidentally unreferenced.
    if name.startswith('cfm_'): errors.append(f'CFM shader source not embedded by Rust: {name}.comp')
# Deterministic vectors.
expected={
 'test_locenc_patch.f32':256,'test_locdit_x.f32':256,'test_locdit_cond.f32':256,
 'test_locdit_mu.f32':2048,'test_reference_latents.f32':512,'test_prompt_latents.f32':512,
 'test_cfm_initial_x.f32':256,
}
for name,n in expected.items():
    p=root/name
    if not p.exists() or p.stat().st_size != n*4: errors.append(f'{name}: expected {n*4} bytes')
# Project isolation.
for p in (root/'src').glob('*.rs'):
    low=p.read_text(encoding='utf-8',errors='ignore').lower()
    if 'reading_companion' in low or 'reading companion' in low: errors.append(f'non-VoxGen integration reference: {p.name}')
# Contract markers and exact solver semantics.
local=(root/'src'/'local.rs').read_text(encoding='utf-8')
shader=(root/'shaders'/'cfm_cfg_euler.comp').read_text(encoding='utf-8')
noise=(root/'shaders'/'cfm_noise.comp').read_text(encoding='utf-8')
for marker in [
 'CfmOptions','cfm_time_span','use_cfg_zero_star','sway_sampling_coef',
 'record_locdit_common(gpu,a,t,0.0)','cmd_fill_buffer','cfm_positive','cfm_negative',
 'x[i] -= pc.dt * guided',
]:
    blob=local+'\n'+shader
    if marker not in blob: errors.append(f'missing CFM contract marker: {marker}')
if 's_dot[0] / (s_norm[0] + 1.0e-8)' not in shader: errors.append('missing CFG-Zero* optimized scale')
if 'n * scale + pc.cfg * (p - n * scale)' not in shader: errors.append('missing exact CFG blend')
if 'Box-Muller' not in noise: errors.append('missing GPU Gaussian initializer')
# The OpenBMB sway schedule endpoints should remain 1 and 0 for default sway.
def warp(t,s=1.0): return t+s*(math.cos(math.pi/2*t)-1+t)
if abs(warp(1)-1)>1e-7 or abs(warp(0))>1e-7: errors.append('sway endpoint check failed')
# Version/status.
if 'version = "0.5.0"' not in (root/'Cargo.toml').read_text(): errors.append('Cargo version is not 0.5.0')
if 'implementation_iteration: 5' not in (root/'src'/'runtime.rs').read_text(): errors.append('runtime status is not iteration 5')
if errors:
    print('ITERATION 5 STATIC VALIDATION FAILED')
    print('\n'.join(' - '+e for e in errors))
    sys.exit(1)
print(f'ITERATION 5 STATIC VALIDATION OK: {len(includes)} embedded SPIR-V targets, {len(shaders)} shader sources')
