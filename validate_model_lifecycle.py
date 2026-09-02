from pathlib import Path
import re

root = Path(__file__).resolve().parent

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

main = (root / 'src/main.rs').read_text()
http = (root / 'src/http.rs').read_text()
demo = (root / 'demo/src/main.rs').read_text()
readme = (root / 'README.md').read_text()

need('server: bool' in main, 'missing --server CLI flag')
need('args.server && args.base_lm.is_none()' in main, 'missing empty-server startup path')
need('Option<Arc<Runtime>>' in http, 'HTTP server must permit unloaded runtime')
need('struct ModelLoadRequest' in http, 'missing model-load request type')
for field in ['base_lm: PathBuf', 'acoustic: Option<PathBuf>', 'base_format: Option<String>', 'gpu: Option<usize>', 'max_context: Option<u32>']:
    need(field in http, f'missing model load field {field}')
for route in ['/v1/models/load', '/v1/models/current', '/v1/models/unload']:
    need(route in http, f'missing route {route}')
need('inference_gate: Mutex<()>' in http, 'model reload must serialize against inference')
need('let old =' in http and 'drop(old);' in http, 'old runtime must be dropped before replacement load')
need('Runtime::load(' in http, 'model load endpoint does not construct runtime')
need('speech_inference_ready' in http, 'lifecycle readiness missing')
# Model paths belong to the model lifecycle request, not the speech request.
speech = http[http.index('struct SpeechRequest'):http.index('struct ModelLoadRequest')]
need('base_lm' not in speech and 'acoustic' not in speech, 'speech request should not trigger model reloads')
for text in ['Select BaseLM component...', 'Select Acoustic component...', 'Load VoxCPM2', 'current_model_paths', 'load_models(base: &Path, acoustic: &Path)']:
    need(text in demo, f'demo model selection missing {text}')
need('.arg("--server")' in demo, 'demo must start lifecycle server rather than hard-code model CLI paths')
need('"/v1/models/load"' in demo and '"/v1/models/current"' in demo, 'demo must use public model API')
need('POST /v1/models/load' in readme and 'server-side filesystem paths' in readme, 'README lifecycle documentation missing')
print('model lifecycle API validation OK')
