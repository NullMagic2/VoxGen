from pathlib import Path

root = Path(__file__).resolve().parent

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

ac = (root / 'src/acoustic.rs').read_text()
rt = (root / 'src/runtime.rs').read_text()
http = (root / 'src/http.rs').read_text()
demo = (root / 'demo/src/main.rs').read_text()

need('current_lm: GpuBuffer' in ac, 'missing canonical LM-state buffer')
need('pub fn current_lm_buffer(&self)' in ac, 'missing canonical LM-state accessor')
need('self.record_current_lm_copy(gpu,&self.fsq_output)' in ac or 'self.record_current_lm_copy(gpu, &self.fsq_output)' in ac,
     'post-FSQ state is not promoted to canonical LM state')
need('self.record_current_lm_copy(gpu,src)' in ac or 'self.record_current_lm_copy(gpu, src)' in ac,
     'raw text-prefix state is not promoted to canonical LM state')
need('ae.current_lm_buffer(), ae.output_buffer()' in rt,
     'LocDiT is not bound to canonical LM state + ResidualLM state')
need('predict_stop_from_current_lm' in rt,
     'stop predictor is not consuming canonical LM state')
# Guard against regressing to the exact v0.7.7 bug in the synthesis path.
predict = rt[rt.index('pub fn predict_stop'):rt.index('pub fn advance_generated_patch')]
need('base.output_buffer()' not in predict, 'synthesis stop predictor still uses raw BaseLM output')
# Current upstream default is 4.
need('streaming_prefix_len: r.streaming_prefix_len.unwrap_or(4)' in http,
     'HTTP streaming prefix default must be 4')
# Diagnostics to distinguish early stop from decoder problems on hardware.
for h in ['X-VoxGen-Generated-Patches','X-VoxGen-Stopped-By-Predictor','X-VoxGen-Audio-Seconds','X-VoxGen-RTF']:
    need(h in http, f'missing generation response header {h}')
need('acoustic patches (~{duration:.2}s)' in demo and 'Variation {}/{}' in demo, 'demo does not report generation patch count')
need('Select BaseLM component...' in demo and 'Select Acoustic component...' in demo and 'Load VoxCPM2' in demo,
     'demo still presents split GGUF components as alternative models')
print('generation-state validation OK: canonical raw/FSQ LM state + diagnostics')
