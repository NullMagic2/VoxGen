from pathlib import Path
root=Path(__file__).resolve().parent
src=(root/'demo/src/main.rs').read_text()
root_cargo=(root/'Cargo.toml').read_text()
demo_cargo=(root/'demo/Cargo.toml').read_text()

def need(cond,msg):
    if not cond: raise SystemExit(f'FAIL: {msg}')

need('version = "0.7.55"' in root_cargo, 'root version')
need('version = "0.7.55"' in demo_cargo, 'demo version')
need('fn append_benchmark_results(' in src, 'benchmark helper')
need('--- Benchmark results ---' in src, 'benchmark heading')
need('Mode: {mode_label}' in src, 'mode line')
need('Generation wall time:' in src, 'wall time line')
need('First PCM ready:' in src, 'first PCM line')
need('Time to playback start:' in src, 'buffered playback start line')
need('Adaptive startup buffer:' in src, 'adaptive buffer line')
need('End-to-end throughput headroom:' in src, 'end-to-end headroom line')
need('Streaming cadence headroom:' in src, 'streaming cadence headroom line')
need('Patch delivery: avg' in src, 'patch cadence line')
need('late_patch_intervals' in src and 'patch_deadline_ms' in src, 'stream jitter counters')
need('append_benchmark_results(log, &engine_mode, true, &text, variations, &summaries);' in src, 'stream benchmark append')
need('append_benchmark_results(log, &engine_mode, false, &text, variations, &summaries);' in src, 'file benchmark append')
need('XTX tuning: shared QKV + targeted barriers + residual-rms/swiglu + wave32 + subgroup reductions + x4 linear + prefill32-live; GPU profile off; coopmat off' in src, 'XTX tuning line')
print('demo benchmark validation passed')
