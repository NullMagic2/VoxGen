from pathlib import Path
import re, sys
root=Path(__file__).resolve().parent
rt=(root/'src/runtime.rs').read_text()
http=(root/'src/http.rs').read_text()
demo=(root/'demo/src/main.rs').read_text()
root_cargo=(root/'Cargo.toml').read_text()
demo_cargo=(root/'demo/Cargo.toml').read_text()
errors=[]
def need(ok,msg):
    if not ok: errors.append(msg)
need('version = "0.7.37"' in root_cargo and 'version = "0.7.37"' in demo_cargo,'root/demo version 0.7.37')
need('conditioning_audio_cache' in rt and 'audiovae_encode_wav_patches_cached' in rt,'AudioVAE conditioning cache')
need('warm_reference_wav' in rt and '/v1/audio/conditioning/warm' in http,'conditioning warm endpoint')
need('reference_audio_path' in demo and 'prompt_audio_path' in demo,'demo stable reference/prompt path')
need('reference_audio"] = json!' not in demo and 'BASE64.encode' not in demo,'demo does not base64 local reference per Speak')
need('const STREAM_PREBUFFER_MIN_PATCHES: usize = 1;' in demo,'one-patch normal-speed startup')
need('if initial_speed > 100.0' in demo and 'prebuffer_target = prebuffer_target.max(2)' in demo,'faster playback retains reserve')
# First PCM block must precede stop prediction in the generation loop.
pcm=rt.find('if first_pcm_ms.is_none()')
stop=rt.find('let stop=self.predict_stop()?', pcm)
need(pcm>=0 and stop>pcm,'PCM published before stop predictor')
need('if self.gpu.gpu_profiling_enabled() { 16 } else { 32 }' in rt,'profile16/live32 XTX prefill batching')
need('prefill32-live' in demo,'benchmark reports live prefill32 tuning')
need('base64 = "0.22"' not in demo_cargo,'unused demo base64 dependency removed')
need('"version": env!("CARGO_PKG_VERSION")' in http and 'fn server_version() -> Option<String>' in demo,'health exposes server version')
need('server_version().as_deref() != Some(env!("CARGO_PKG_VERSION"))' in demo,'demo restarts stale VoxGen listener')
if errors:
    print('v0.7.37 streaming-startup validation FAILED')
    for e in errors: print(' -',e)
    sys.exit(1)
print('PASS: v0.7.37 stable-path conditioning cache/prewarm, first-patch playback, PCM-before-stop, and live XTX prefill32')
