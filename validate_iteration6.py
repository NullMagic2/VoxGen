from pathlib import Path
import math, random, re, struct, sys, wave

root=Path(__file__).resolve().parent
errors=[]

def need(cond,msg):
    if not cond: errors.append(msg)

# Package/version/dependency.
cargo=(root/'Cargo.toml').read_text(encoding='utf-8')
need('version = "0.6.0"' in cargo,'Cargo version is not 0.6.0')
need('hound = "3.5"' in cargo,'hound WAV dependency missing')
need((root/'src'/'audiovae.rs').exists(),'src/audiovae.rs missing')

# Every unique embedded SPIR-V target must map exactly to one source stem, with no stale source.
includes=set()
for p in (root/'src').glob('*.rs'):
    text=p.read_text(encoding='utf-8')
    includes.update(re.findall(r'/([A-Za-z0-9_]+)\.spv',text))
shaders={p.stem for p in (root/'shaders').glob('*.comp')}
for name in sorted(includes-shaders): errors.append(f'missing shader source for embedded {name}.spv')
for name in sorted(shaders-includes): errors.append(f'shader source is not embedded by Rust: {name}.comp')
need(len(shaders)==34,f'expected 34 shader sources after AudioVAE addition, found {len(shaders)}')
for name in ['vae_conv1d','vae_convtranspose1d','vae_snake','vae_scale_bias','vae_tanh','vae_add']:
    need(name in includes,f'AudioVAE shader {name}.spv is not embedded')

# Basic source delimiter scan (comments/strings are not parsed; catches packaging truncation).
for p in (root/'src').glob('*.rs'):
    s=p.read_text(encoding='utf-8')
    for op,cl in [('(',')'),('[',']'),('{','}')]:
        need(s.count(op)==s.count(cl),f'{p.name}: unbalanced {op}{cl} counts')

vae=(root/'src'/'audiovae.rs').read_text(encoding='utf-8')
runtime=(root/'src'/'runtime.rs').read_text(encoding='utf-8')
main=(root/'src'/'main.rs').read_text(encoding='utf-8')

# Exact architecture/conditioning markers.
markers=[
    'const ENCODER_RATES: [u32; 4] = [2, 5, 8, 8]',
    'const DECODER_RATES: [u32; 6] = [8, 6, 5, 2, 2, 2]',
    'const SR_BINS: [u32; 3] = [20_000, 30_000, 40_000]',
    '!= (128, 64, 2048, 16_000, 48_000, 4)',
    'cond_type != "scale_bias"',
    'audio_vae.encoder.fc_mu',
    'audio_vae.encoder.fc_logvar',
    'audio_vae.decoder.sr_cond_model',
    'audio_vae.decoder.model.9',
    'AudioPadSide::Right=>x.resize',
    'AudioPadSide::Left=>',
    '.ceil() as usize',
]
for m in markers: need(m in vae,f'missing AudioVAE contract marker: {m}')
need('implementation_iteration: 6' in runtime,'runtime status is not iteration 6')
need('speech_inference_ready: false' in runtime,'step-7 speech boundary is not explicit')
need('audiovae_encode_wav' in runtime and 'audiovae_encode_pcm16k' in runtime and 'audiovae_decode_latents' in runtime,'runtime AudioVAE bridges missing')
need('--reference-wav' in main and '--prompt-wav' in main,'WAV-conditioning CLI missing')

# New shader interfaces contain expected operations/binding contracts.
conv=(root/'shaders'/'vae_conv1d.comp').read_text()
convt=(root/'shaders'/'vae_convtranspose1d.comp').read_text()
snake=(root/'shaders'/'vae_snake.comp').read_text()
scale=(root/'shaders'/'vae_scale_bias.comp').read_text()
tanh=(root/'shaders'/'vae_tanh.comp').read_text()
for blob,name in [(conv,'conv1d'),(convt,'convtranspose1d')]:
    need('binding = 0' in blob and 'binding = 1' in blob and 'binding = 2' in blob,f'{name}: expected three storage bindings')
    need('weight_offset' in blob and 'bias_offset' in blob and 'groups' in blob,f'{name}: missing push parameters')
need('int(t * pc.stride + k * pc.dilation) - int(pc.left_pad)' in conv,'causal Conv1d indexing marker missing')
need('uint first_k=t%pc.stride' in convt and 'uint ti=(t-k)/pc.stride' in convt,'optimized ConvTranspose gather indexing marker missing')
need('x+(s*s)/(a+1e-9)' in snake,'Snake formula marker missing')
need('values[i]=values[i]*scale+bias' in scale,'sample-rate scale/bias formula missing')
need('values[i]=tanh(values[i])' in tanh,'final tanh marker missing')

# Derived temporal contracts.
enc_rates=[2,5,8,8]; dec_rates=[8,6,5,2,2,2]
enc_hop=math.prod(enc_rates); dec_hop=math.prod(dec_rates)
need(enc_hop==640,'encoder hop != 640')
need(dec_hop==1920,'decoder hop != 1920')
need(enc_hop*4==2560,'input samples/patch != 2560')
need(dec_hop*4==7680,'output samples/patch != 7680')
need(abs((enc_hop*4/16000)-(dec_hop*4/48000))<1e-12,'encoder/decoder patch durations differ')

# Verify causal Conv1d shader indexing against independently padded standard convolution.
def conv_shader(x,w,b,stride=1,dilation=1,groups=1,left_pad=0):
    # x [IC][T], w [OC][IC/groups][K]
    ic=len(x); ilen=len(x[0]); oc=len(w); klen=len(w[0][0]);
    out_per_group=oc//groups; in_per_group=ic//groups
    # same integer shape supplied by caller, derive from causal standard convolution
    olen=(ilen+left_pad-dilation*(klen-1)-1)//stride+1
    y=[[0.0]*olen for _ in range(oc)]
    for o in range(oc):
        grp=o//out_per_group; ib=grp*in_per_group
        for t in range(olen):
            acc=b[o]
            for ig in range(in_per_group):
                for k in range(klen):
                    ti=t*stride+k*dilation-left_pad
                    if 0<=ti<ilen: acc+=x[ib+ig][ti]*w[o][ig][k]
            y[o][t]=acc
    return y

def conv_padded_ref(x,w,b,stride=1,dilation=1,groups=1,left_pad=0):
    padded=[[0.0]*left_pad+row[:] for row in x]
    ic=len(x); plen=len(padded[0]); oc=len(w); klen=len(w[0][0]);
    out_per_group=oc//groups; in_per_group=ic//groups
    olen=(plen-dilation*(klen-1)-1)//stride+1
    y=[[0.0]*olen for _ in range(oc)]
    for o in range(oc):
        grp=o//out_per_group; ib=grp*in_per_group
        for t in range(olen):
            acc=b[o]
            for ig in range(in_per_group):
                for k in range(klen):
                    acc+=padded[ib+ig][t*stride+k*dilation]*w[o][ig][k]
            y[o][t]=acc
    return y

rng=random.Random(1234)
def rr(): return rng.uniform(-0.5,0.5)
for stride,dilation,groups,ic,oc,k,left in [(1,1,1,3,4,7,6),(1,3,4,4,4,7,18),(5,1,1,4,8,10,5)]:
    x=[[rr() for _ in range(30)] for _ in range(ic)]
    w=[[[rr() for _ in range(k)] for _ in range(ic//groups)] for _ in range(oc)]
    b=[rr() for _ in range(oc)]
    a=conv_shader(x,w,b,stride,dilation,groups,left); z=conv_padded_ref(x,w,b,stride,dilation,groups,left)
    need(len(a)==len(z) and all(abs(a[o][t]-z[o][t])<1e-10 for o in range(oc) for t in range(len(a[o]))),f'Conv1d indexing parity failed for stride={stride}, dilation={dilation}, groups={groups}')

# Verify retained ConvTranspose samples: shader gather vs scatter-then-right-crop.
def convt_gather(x,w,b,stride=2,groups=1):
    # PyTorch raw w [IC][OC/groups][K], retained output length = input_len*stride
    ic=len(x); ilen=len(x[0]); oper=len(w[0]); oc=oper*groups; klen=len(w[0][0]); olen=ilen*stride
    ipg=ic//groups; y=[[0.0]*olen for _ in range(oc)]
    for o in range(oc):
        grp=o//oper; og=o-grp*oper; ib=grp*ipg
        for t in range(olen):
            acc=b[o]
            for ig in range(ipg):
                ii=ib+ig
                for k in range(klen):
                    if t<k: continue
                    d=t-k
                    if d%stride: continue
                    ti=d//stride
                    if ti<ilen: acc+=x[ii][ti]*w[ii][og][k]
            y[o][t]=acc
    return y

def convt_scatter_crop(x,w,b,stride=2,groups=1):
    ic=len(x); ilen=len(x[0]); oper=len(w[0]); oc=oper*groups; klen=len(w[0][0]); full=(ilen-1)*stride+klen
    ipg=ic//groups; y=[[b[o] for _ in range(full)] for o in range(oc)]
    for ii in range(ic):
        grp=ii//ipg
        for og in range(oper):
            o=grp*oper+og
            for ti in range(ilen):
                for k in range(klen): y[o][ti*stride+k]+=x[ii][ti]*w[ii][og][k]
    # AudioVAE k=2*stride and causal crop removes exactly stride tail -> ilen*stride retained.
    return [row[:ilen*stride] for row in y]

for stride,groups,ic,oc in [(2,1,4,2),(5,1,4,2),(2,2,4,4)]:
    k=2*stride; oper=oc//groups
    x=[[rr() for _ in range(7)] for _ in range(ic)]
    w=[[[rr() for _ in range(k)] for _ in range(oper)] for _ in range(ic)]
    b=[rr() for _ in range(oc)]
    a=convt_gather(x,w,b,stride,groups); z=convt_scatter_crop(x,w,b,stride,groups)
    need(all(abs(a[o][t]-z[o][t])<1e-10 for o in range(oc) for t in range(len(a[o]))),f'ConvTranspose indexing parity failed for stride={stride}, groups={groups}')

# Alignment side semantics.
def align(xs,m,left):
    rem=len(xs)%m
    if not rem: return xs[:]
    n=m-rem
    return ([0.0]*n+xs) if left else (xs+[0.0]*n)
x=[1.0,2.0,3.0]
need(align(x,5,False)==[1,2,3,0,0],'right padding semantics failed')
need(align(x,5,True)==[0,0,1,2,3],'left padding semantics failed')

# Included deterministic inputs.
wp=root/'test_vae_input.wav'
need(wp.exists(),'test_vae_input.wav missing')
if wp.exists():
    with wave.open(str(wp),'rb') as w:
        need(w.getnchannels()==1,'test VAE WAV is not mono')
        need(w.getframerate()==16000,'test VAE WAV is not 16 kHz')
        need(w.getnframes()==3000,'test VAE WAV should be deliberately non-aligned at 3000 samples')
pcm=root/'test_vae_pcm16k.f32'
need(pcm.exists() and pcm.stat().st_size==3000*4,'test_vae_pcm16k.f32 must contain exactly 3000 f32 samples')
latent=root/'test_vae_latents.f32'
need(latent.exists() and latent.stat().st_size==256*4,'test_vae_latents.f32 must be exactly 256 f32 values')
need(len(align([0.0]*3000,2560,False))==5120,'3000-sample right alignment should yield 5120 samples')
need(len(align([0.0]*3000,2560,True))==5120,'3000-sample left alignment should yield 5120 samples')

# Project isolation.
for p in list((root/'src').glob('*.rs'))+[root/'README.md',root/'ITERATION6_VALIDATION.md']:
    low=p.read_text(encoding='utf-8',errors='ignore').lower()
    # README explicitly says no integration; that phrase itself is allowed. Source must contain none.
    if p.suffix=='.rs' and ('reading_companion' in low or 'reading companion' in low):
        errors.append(f'non-VoxGen application integration reference in {p.name}')

# No hidden speech-ready claim.
need('complete autoregressive TTS/stop-token/streaming loop is step 7' in runtime,'runtime does not preserve step-7 boundary message')

if errors:
    print('ITERATION 6 STATIC VALIDATION FAILED')
    print('\n'.join(' - '+e for e in errors))
    sys.exit(1)
print(f'ITERATION 6 STATIC VALIDATION OK: {len(includes)} unique embedded SPIR-V targets, {len(shaders)} shader sources')
print('AudioVAE CPU indexing contracts: causal Conv1d and retained ConvTranspose1d parity OK')
print('Temporal contract: 2560 input samples/patch @16k == 7680 output samples/patch @48k == 160 ms')
