import math, struct
from pathlib import Path

def write(name, values):
    Path(name).write_bytes(b''.join(struct.pack('<f', float(x)) for x in values))
    print(name, len(values), 'f32')

def patch(phase, scale=0.2):
    return [scale*math.sin(phase+i*0.073)+0.03*math.cos(i*0.031) for i in range(256)]

write('test_locenc_patch.f32', patch(0.1))
write('test_locdit_x.f32', patch(0.7, 0.35))
write('test_locdit_cond.f32', patch(1.3, 0.25))
write('test_locdit_mu.f32', [0.12*math.sin(i*0.017)+0.05*math.cos(i*0.009) for i in range(2048)])
# Two reference patches and two continuation/prompt patches.
write('test_reference_latents.f32', patch(0.2)+patch(0.5))
write('test_prompt_latents.f32', patch(1.1)+patch(1.4))

write('test_cfm_initial_x.f32', [0.45*math.sin(i*0.113+0.4)+0.17*math.cos(i*0.071-0.2) for i in range(256)])
