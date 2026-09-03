from pathlib import Path
root=Path(__file__).resolve().parent
demo=(root/'demo/src/main.rs').read_text()
main=(root/'src/main.rs').read_text()
http=(root/'src/http.rs').read_text()
runtime=(root/'src/runtime.rs').read_text()
vk=(root/'src/vulkan.rs').read_text()
root_cargo=(root/'Cargo.toml').read_text()
demo_cargo=(root/'demo/Cargo.toml').read_text()

def need(cond,msg):
    if not cond: raise SystemExit(f'FAIL: {msg}')

need('version = "0.7.39"' in root_cargo, 'root version')
need('version = "0.7.39"' in demo_cargo, 'demo version')
# Adaptive buffering
need('STREAM_PREBUFFER_MIN_PATCHES: usize = 1' in demo, 'one-patch low-latency minimum buffer')
need('if initial_speed > 100.0' in demo and 'prebuffer_target = prebuffer_target.max(2)' in demo, 'faster-speed reserve')
need('STREAM_PREBUFFER_MAX_PATCHES: usize = 4' in demo, 'four-patch maximum buffer')
need('STREAM_PREBUFFER_RISK_RATIO: f64 = 0.85' in demo, 'adaptive risk threshold')
need('prebuffer_samples.extend_from_slice(&rendered)' in demo, 'stream samples are prebuffered')
need('generated_patches >= prebuffer_target' in demo, 'buffer target gate')
need('Adaptive startup buffer:' in demo, 'buffer benchmark line')
need('First PCM ready:' in demo and 'Time to playback start:' in demo, 'separate first-ready and audible-start metrics')
# Controlled same-seed A/B
need('Benchmark Normal vs XTX' in demo, 'A/B button')
need('run_offline_benchmark_case(' in demo, 'offline A/B helper')
need('"normal",\n                            &text, sample.as_deref(), &expressive, seed' in demo, 'Normal receives shared seed')
need('"xtx7900",\n                            &text, sample.as_deref(), &expressive, seed' in demo, 'XTX receives shared seed')
need('Normal vs XTX 7900 A/B benchmark' in demo, 'comparison block')
need('Text hash: fnv1a64:' in demo, 'stable text fingerprint')
need('restore_selected_server(' in demo, 'selected live mode restoration')
# Offline profiling
need('long = "benchmark-profile"' in main, '--benchmark-profile CLI')
need('--benchmark-profile is offline-only; use --stream off' in main, 'offline-only guard')
need('--benchmark-profile requires --mode xtx7900' in main, 'XTX-only guard')
need('gpu_profile: args.gpu_profile.enabled() || args.benchmark_profile' in main, 'benchmark profile enables timestamps')
need('fn reset_gpu_profile(&self)' in vk and 'state.totals.clear()' in vk, 'GPU totals reset')
need('GET", "/v1/profile/gpu"' in http and 'POST", "/v1/profile/gpu/reset"' in http, 'profile HTTP API')
need('Profile XTX' in demo and 'ensure_offline_profile_server' in demo, 'demo offline profile action')
need('Hot kernels (descending total GPU time):' in demo, 'hot-kernel output')
# End-to-end timing begins before prefill.
need(runtime.index('let started=Instant::now();') < runtime.index('let _prefill=self.prefill_latent_conditioning'), 'synthesis timer must include prefill')
need('X-VoxGen-Elapsed-Ms' in http and 'X-VoxGen-First-PCM-Ms' in http, 'offline benchmark headers')
# Structured semantic request errors before streaming headers.
need('fn parse_speech_request' in http, 'single-pass speech parse/prevalidation')
need('let req: SpeechRequest = serde_json::from_slice(body)' in http, 'speech request is actually parsed in prevalidation helper')
need('fn speech(mut s: TcpStream, state: Arc<ServerState>, req: SpeechRequest, streaming: bool)' in http, 'speech handler consumes parsed request')
speech_start=http.index('fn speech(mut s: TcpStream')
speech_end=http.index('fn handle(', speech_start) if 'fn handle(' in http[speech_start:] else http.index('pub fn serve', speech_start)
need('serde_json::from_slice(body)' not in http[speech_start:speech_end], 'speech handler must not reparse an undefined/raw body')
need('clone_mode=reference requires reference audio' in http, 'reference error remains explicit')
need('400 Bad Request' in http, 'speech semantic errors become HTTP 400')
print('PASS: adaptive buffering + controlled A/B + offline XTX profiling + benchmark timing')
