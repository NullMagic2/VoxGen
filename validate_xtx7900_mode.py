from pathlib import Path
import hashlib

root = Path(__file__).resolve().parent

def need(x, msg):
    if not x:
        raise AssertionError(msg)

cargo = (root / 'Cargo.toml').read_text()
demo_cargo = (root / 'demo/Cargo.toml').read_text()
main = (root / 'src/main.rs').read_text()
rt = (root / 'src/runtime.rs').read_text()
vk = (root / 'src/vulkan.rs').read_text()
http = (root / 'src/http.rs').read_text()
base = (root / 'src/baselm.rs').read_text()
ac = (root / 'src/acoustic.rs').read_text()
local = (root / 'src/local.rs').read_text()
vae = (root / 'src/audiovae.rs').read_text()
demo = (root / 'demo/src/main.rs').read_text()
coop = (root / 'shaders/seq_linear_bias_coopmat_xtx7900.comp').read_text()

need('version = "0.7.60"' in cargo, 'root version')
need('version = "0.7.60"' in demo_cargo, 'demo version')
need('enum ModeArg' in main and 'default_value = "normal"' in main, '--mode default normal')
need('#[value(name = "xtx7900")]' in main, 'xtx7900 CLI value')
need('ExecutionMode::Xtx7900' in main, 'CLI->runtime mode mapping')
need('pub enum ExecutionMode { Normal, Xtx7900 }' in vk, 'execution mode enum')
need('info.vendor_id != 0x1002' in vk and 'name.contains("7900 xtx")' in vk, 'strict RX 7900 XTX validation')
need('mode == ExecutionMode::Xtx7900' in vk and 'find(|(_, x)|' in vk, 'xtx mode auto-selects the XTX when --gpu is omitted')
need('subgroup_arithmetic' in vk and 'matches!(info.subgroup_size, 32 | 64)' in vk, 'subgroup capability validation')
need('subgroup_size_control' in vk and 'required_subgroup_size_compute' in vk, 'subgroup-size-control capability discovery')
need('PipelineShaderStageRequiredSubgroupSizeCreateInfo' in vk and '.required_subgroup_size(32)' in vk, 'forced wave32 pipeline stage')
need('select_spirv' in vk, 'mode-specific shader selector')
need('execution_mode: self.gpu.mode.as_str()' in rt, 'runtime reports selected mode')
need('default_mode: ExecutionMode' in http and 'state.default_mode' in http, 'lifecycle server preserves startup mode')
need('"mode": st.execution_mode' in http and 'server execution mode' in http, 'HTTP/status mode reporting')
need('Generic Vulkan kernels enabled' in main and 'RX 7900 XTX stream-safe kernels enabled' in main, 'pass-3 startup mode diagnostics')

# Pass 2 cooperative-matrix discovery and fallback.
need('PhysicalDeviceCooperativeMatrixFeaturesKHR' in vk, 'cooperative-matrix feature query')
need('get_physical_device_cooperative_matrix_properties' in vk, 'cooperative-matrix shape query')
need('cooperative_matrix_16x16x16_f16_f32' in vk, 'required cooperative-matrix shape flag')
need('pub fn xtx_coopmat_enabled' in vk, 'cooperative-matrix runtime selector')
need('SEQ_LINEAR_COOPMAT_XTX7900_SPV' in local, 'embedded cooperative-matrix shader')
need('gpu.xtx_coopmat_enabled()' in local, 'local path selects cooperative-matrix kernel when available')
need('coopMatMulAdd' in coop and 'coopMatLoad' in coop and 'coopMatStore' in coop, 'cooperative-matrix shader operations')
need('16, 16, gl_MatrixUseA' in coop and '16, 16, gl_MatrixUseB' in coop, '16x16 cooperative-matrix tiles')

# Pass 2 Vulkan GPU timestamp profiler.
need('GpuProfileSnapshot' in vk and 'gpu_profile_begin' in vk and 'gpu_profile_end' in vk, 'GPU profiler API')
need('create_query_pool' in vk and 'vk::QueryType::TIMESTAMP' in vk, 'timestamp query pool')
need('collect_gpu_profile' in vk and 'timestamp_period_ns' in vk, 'timestamp collection')
need('"gpu_profile":gpu_profile' in main.replace(' ', ''), 'direct TTS JSON includes GPU profile')
for label in ['baselm.matvec', 'baselm.rmsnorm']:
    need(label in base, f'profile label {label}')
for label in ['residual.matvec', 'residual.rmsnorm']:
    need(label in ac, f'profile label {label}')
for label in ['local.seq_linear', 'local.seq_rmsnorm', 'local.attn_scores', 'local.softmax', 'local.attn_values']:
    need(label in local, f'profile label {label}')
for label in ['audiovae.conv1d', 'audiovae.convtranspose1d', 'audiovae.snake']:
    need(label in vae, f'profile label {label}')

# Normal mode remains the v0.7.20 correctness reference. These hashes prove the
# optimized release did not silently alter the generic hot shader sources.
normal_hashes = {
    'matvec': '8987379147eb2a71c717cd2c6e6b0842ba97dbf7ddd2ba696ff8fdb81229fe68',
    'linear_bias': '1b44c3dd3eabfc53621d33840bc94b9c0d7b3fe07ebc8a8a58dac87a6bd93d3d',
    'fusion_linear': 'ab34e1948b11c74dac17d4c7898b9b8577a343f248fbd203085dea1948c875b9',
    'seq_linear_bias': '82e839e55db021d4e497042673a332be36175e93fad1e6f2382d15dd262fb21f',
    'rmsnorm': 'dde2e6dd68005677101063ecbaaf50eebacfab7bd8203a1fcf7d9ca28095a748',
    'seq_rmsnorm': 'a71ba75632b6186b45f0829ca9d843c42df91172f7154365b9a603ed040768f9',
    'attn_scores': 'b80781ca6736cd05cd5094b856c8b955d8ca3deb619a6774058bb47b3f841e91',
    'dense_attn_scores': 'b7eb4400decdba0db275b0257219c9e5823ce7058d90cc0e9ae6a4fa2bb3206d',
    'softmax': 'bf05960bf46adfb49fa1c39b58121ad33672dedab65cce7a060e5befe7916caa',
}
for stem, expected in normal_hashes.items():
    actual = hashlib.sha256((root / 'shaders' / f'{stem}.comp').read_bytes()).hexdigest()
    need(actual == expected, f'{stem}: normal shader changed; normal mode must remain v0.7.20 reference')

# Pass-1 subgroup shaders remain, plus the pass-2 cooperative-matrix sequence linear.
subgroup_stems = ['matvec', 'linear_bias', 'fusion_linear', 'seq_linear_bias', 'rmsnorm', 'seq_rmsnorm', 'attn_scores', 'dense_attn_scores', 'softmax']
for stem in subgroup_stems:
    path = root / 'shaders' / f'{stem}_xtx7900.comp'
    need(path.exists(), f'missing XTX subgroup shader {stem}')
    text = path.read_text()
    need('GL_KHR_shader_subgroup_arithmetic' in text, f'{path.name}: subgroup arithmetic extension')
    need('subgroupAdd' in text or 'subgroupMax' in text, f'{path.name}: no subgroup reduction')
need((root / 'shaders/seq_linear_bias_coopmat_xtx7900.comp').exists(), 'missing pass-2 cooperative-matrix shader')

for src, names in [
    (base, ['MATVEC', 'RMSNORM', 'ATTN_SCORES', 'SOFTMAX']),
    (ac, ['MATVEC', 'RMSNORM', 'ATTN_SCORES', 'SOFTMAX', 'LINEAR_BIAS', 'FUSION_LINEAR']),
    (local, ['SEQ_RMS', 'DENSE_SCORES']),
]:
    for n in names:
        need(f'{n}_XTX7900_SPV' in src, f'missing {n} XTX embedded shader')
        need(f'select_spirv({n}_SPV, {n}_XTX7900_SPV)' in src, f'{n} not mode-selected')
need('SEQ_LINEAR_XTX7900_SPV' in local and 'SEQ_LINEAR_COOPMAT_XTX7900_SPV' in local, 'both sequence-linear XTX variants embedded')

# Pass 3 widens hot XTX linear loops to four values per lane iteration.
for stem in ['matvec', 'linear_bias', 'fusion_linear', 'seq_linear_bias']:
    text = (root / 'shaders' / f'{stem}_xtx7900.comp').read_text()
    need('unpackHalf2x16' in text and 'lane*4u' in text and '1024u' in text, f'{stem}: missing x4 F16 path')
    need('qoff' in text and 'q0' in text and 'q3' in text, f'{stem}: missing x4 Q8 path')

# Demo exposes and persists the engine mode, so XTX mode is testable without manual server launch.
need('("normal", "Normal")' in demo and '("xtx7900", "XTX 7900")' in demo, 'demo engine mode choices')
need('with_label("Engine mode:")' in demo, 'demo Engine mode control')
need('mode={}' in demo and 'cfg.engine_mode' in demo, 'demo persists engine mode')
need('"--mode"' in demo and 'engine_mode' in demo, 'demo passes --mode to engine')
need('Click Load VoxCPM2 to switch mode and reload the models.' in demo, 'mode change points to unified load action')
need('with_label("Apply mode + reload")' not in demo and 'apply_mode_button' not in demo, 'obsolete second mode button removed')
need('load_models_button.on_click' in demo and 'if mode_mismatch' in demo, 'single load button conditionally switches mode')

need('if self.gpu.gpu_profiling_enabled() { 16 } else { 32 }' in rt, 'profile16/live32 XTX prefill batches')
need('prefill.cross_engine_batch' in rt, 'cross-engine prefill GPU profile span')
need('record_token_gpu_only_in' in base and 'prefill_command_buffer' in base, 'BaseLM external prefill recording')
need('record_text_prefix_from_gpu_base_in' in ac, 'ResidualLM external prefill recording')
need('self.gpu.mode == ExecutionMode::Xtx7900' in rt, 'prefill batching isolated to xtx7900 mode')
need('gpu_profile: bool' in vk and 'cooperative_matrix: bool' in vk, 'XTX tuning flags')
need('self.xtx_tuning.gpu_profile' in vk, 'GPU profiling opt-in gate')
need('self.xtx_tuning.cooperative_matrix' in vk, 'cooperative-matrix opt-in gate')
need('default_value = "off"' in main and 'long = "gpu-profile"' in main and 'long = "xtx-coopmat"' in main, 'stream-safe CLI defaults')
need('.arg("--gpu-profile")' in demo and '.arg("off")' in demo and '.arg("--xtx-coopmat")' in demo, 'demo uses stream-safe XTX options')
print('xtx7900 validation OK: stream-safe wave32/subgroup/x4/prefill defaults with opt-in coopmat/profiling')
