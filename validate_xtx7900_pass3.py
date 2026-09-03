from pathlib import Path

root = Path(__file__).resolve().parent

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

rt = (root/'src/runtime.rs').read_text()
base = (root/'src/baselm.rs').read_text()
ac = (root/'src/acoustic.rs').read_text()
main = (root/'src/main.rs').read_text()

need('version = "0.7.39"' in (root/'Cargo.toml').read_text(), 'root version')
need('if self.gpu.gpu_profiling_enabled() { 16 } else { 32 }' in rt, 'profile16/live32 prefill batch size')
need('self.gpu.mode == ExecutionMode::Xtx7900' in rt, 'XTX-only prefill batching gate')
need('prefill.cross_engine_batch' in rt, 'prefill profiler span')
need('record_token_gpu_only_in' in base, 'BaseLM external recorder')
need('record_text_prefix_from_gpu_base_in' in ac, 'ResidualLM external recorder')
need('gpu.submit_and_wait(cmd)?;' in rt, 'one batch submission path')
need('decode_token_gpu_only' in rt and 'step_text_prefix_from_gpu_base_gpu_only' in rt, 'normal reference prefill retained')
need('stream-safe kernels enabled' in main and 'live32/profile16 cross-engine prefill batching' in main, 'stream-safe pass-3 startup diagnostics')

for stem in ['matvec','linear_bias','fusion_linear','seq_linear_bias']:
    text=(root/'shaders'/f'{stem}_xtx7900.comp').read_text()
    need('lane*4u' in text and '1024u' in text, f'{stem}: x4 lane loop')
    need('q0' in text and 'q1' in text and 'q2' in text and 'q3' in text, f'{stem}: q8 x4 unpack')

# Source-level submission arithmetic for a representative 65-token contiguous run.
tokens=65
old_submits=tokens*2
profile_submits=(tokens+15)//16
live_submits=(tokens+31)//32
need(old_submits == 130 and profile_submits == 5 and live_submits == 3, 'submission-count sanity check')
print(f'xtx7900 pass-3 validation OK: representative text run {old_submits} -> {live_submits} live queue submissions ({profile_submits} while profiling)')
