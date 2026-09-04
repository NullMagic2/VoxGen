from pathlib import Path
root=Path(__file__).resolve().parent

def need(ok,msg):
    if not ok: raise AssertionError(msg)

cargo=(root/'Cargo.toml').read_text()
main=(root/'src/main.rs').read_text()
vk=(root/'src/vulkan.rs').read_text()
http=(root/'src/http.rs').read_text()
demo=(root/'demo/src/main.rs').read_text()
need('version = "0.7.60"' in cargo, 'root version')
need('gpu_profile: bool' in vk and 'cooperative_matrix: bool' in vk, 'XtxTuning fields')
need('default_value = "off"' in main and 'long = "gpu-profile"' in main, '--gpu-profile default off')
need('default_value = "off"' in main and 'long = "xtx-coopmat"' in main, '--xtx-coopmat default off')
need('&& self.xtx_tuning.gpu_profile' in vk, 'timestamp profiler must be opt-in')
need('&& self.xtx_tuning.cooperative_matrix' in vk, 'coopmat path must be opt-in')
need('&& xtx_tuning.cooperative_matrix' in vk, 'coopmat device extension must not be enabled by default')
need('.arg("--gpu-profile")' in demo and '.arg("off")' in demo, 'demo explicitly disables GPU profiling')
need('.arg("--xtx-coopmat")' in demo and '.arg("off")' in demo, 'demo explicitly disables coopmat')
need('"gpu_profile": state.default_xtx_tuning.gpu_profile' in http, 'health reports profiling mode')
need('"xtx_coopmat": state.default_xtx_tuning.cooperative_matrix' in http, 'health reports coopmat mode')
need('stream-safe kernels enabled' in main, 'startup diagnostics')
need('VulkanContext::new(gpu_index, mode, xtx_tuning)' in (root/'src/runtime.rs').read_text(), 'Runtime must forward XTX tuning to Vulkan')
need('state.default_xtx_tuning' in http and 'default_xtx_tuning: XtxTuning' in http, 'lifecycle model reload must preserve XTX tuning')
need('server_xtx_stream_safe()' in demo and 'stop_existing_voxgen_server(state)?;' in demo, 'demo must replace older non-stream-safe XTX server')
print('xtx7900 stream-safe validation OK: profiling/coopmat opt-in; demo defaults both off')
