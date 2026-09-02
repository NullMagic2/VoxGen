import math, struct, wave
from pathlib import Path
root=Path(__file__).resolve().parent
sr=16000
n=3000  # deliberately not patch-aligned: exercises right/left padding to 5120 samples
samples=[0.16*math.sin(2*math.pi*220*i/sr)+0.04*math.sin(2*math.pi*440*i/sr) for i in range(n)]
with wave.open(str(root/'test_vae_input.wav'),'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(b''.join(struct.pack('<h',max(-32768,min(32767,round(x*32767)))) for x in samples))
# One deterministic 4x64 latent patch. Small amplitude keeps the raw decoder smoke path well behaved.
lat=[]
for t in range(4):
    for d in range(64):
        lat.append(0.05*math.sin((t*64+d+1)*0.071)+0.015*math.cos((d+1)*0.113))
(root/'test_vae_latents.f32').write_bytes(b''.join(struct.pack('<f',x) for x in lat))
(root/'test_vae_pcm16k.f32').write_bytes(b''.join(struct.pack('<f',x) for x in samples))
print('wrote',root/'test_vae_input.wav',root/'test_vae_pcm16k.f32',root/'test_vae_latents.f32')
