from pathlib import Path
import re, sys

root = Path(__file__).resolve().parent
vulkan = (root/'src/vulkan.rs').read_text()
local = (root/'src/local.rs').read_text()
audio = (root/'src/audiovae.rs').read_text()
acoustic = (root/'src/acoustic.rs').read_text()
http = (root/'src/http.rs').read_text()
cargo = (root/'Cargo.toml').read_text()

def body(src, fn_name):
    m = re.search(rf'\bfn\s+{re.escape(fn_name)}\b[^{{]*\{{', src)
    if not m:
        raise AssertionError(f'missing function {fn_name}')
    i = m.end()-1
    depth = 0
    for j in range(i, len(src)):
        if src[j] == '{': depth += 1
        elif src[j] == '}':
            depth -= 1
            if depth == 0:
                return src[i:j+1]
    raise AssertionError(f'unclosed function {fn_name}')

errors=[]
def need(cond,msg):
    if not cond: errors.append(msg)

need('version = "0.7.55"' in cargo, 'Cargo version is not 0.7.13')
need('submit_fence: Mutex<vk::Fence>' in vulkan, 'persistent submit fence missing')
sub=body(vulkan,'submit_and_wait')
need('create_fence' not in sub and 'destroy_fence' not in sub, 'submit_and_wait still allocates/destroys fences')
need('reset_fences' in sub and 'wait_for_fences' in sub, 'persistent fence reset/wait missing')
need('record_upload_f32' in vulkan and 'submit_and_read_f32' in vulkan, 'recorded transfer helpers missing')

cfm=body(local,'cfm_solve')
need('submit_and_wait' not in cfm, 'CFM still waits inside solve')
need(cfm.count('submit_and_read_f32') == 1, 'CFM should have exactly one fused submit/readback')
need('record_upload_f32' in cfm, 'CFM condition upload is not recorded into batch')
loop = cfm[cfm.find('for step in 1..span.len()'):]
need('begin_one_time' not in loop, 'CFM restarts command buffer inside Euler loop')

enc=body(audio,'encode_samples')
dec=body(audio,'decode_latents')
for name,b in [('AudioVAE encode',enc),('AudioVAE decode',dec)]:
    need('record_upload_f32' in b, f'{name} does not fuse upload')
    need('submit_and_read_f32' in b, f'{name} does not fuse readback')
    need('gpu.upload_f32' not in b, f'{name} still has standalone upload submission')
    need('gpu.read_f32' not in b, f'{name} still has standalone readback submission')

for name in ['predict_stop_from_gpu_base','predict_stop_from_current_lm']:
    b=body(acoustic,name)
    need('submit_and_read_f32' in b, f'{name} does not fuse logit readback')
    need('gpu.read_f32' not in b, f'{name} still has standalone readback')

loc=body(local,'encode_patch_gpu_only')
need('record_upload_f32' in loc and loc.count('submit_and_wait') == 1, 'LocEnc GPU-only upload/compute is not fused')

speech=body(http,'speech')
need('sync_channel::<Vec<u8>>(4)' in speech, 'bounded streaming writer queue missing')
need('thread::spawn' in speech and 'streaming audio writer' in speech, 'streaming writer thread missing')

if errors:
    print('SUBMISSION BATCHING VALIDATION FAILED')
    for e in errors: print(' -',e)
    sys.exit(1)
print('submission batching validation OK')
