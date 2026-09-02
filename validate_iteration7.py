from pathlib import Path
import re, hashlib, sys
root=Path(__file__).resolve().parent

def need(x,msg):
    if not x: raise AssertionError(msg)

cargo=(root/'Cargo.toml').read_text()
need('version = "0.7.37"' in cargo,'Cargo version')
need('base64 = "0.22"' in cargo,'HTTP base64 dependency')
main=(root/'src/main.rs').read_text(); rt=(root/'src/runtime.rs').read_text(); ac=(root/'src/acoustic.rs').read_text(); gg=(root/'src/gguf.rs').read_text(); tok=(root/'src/tokenizer.rs').read_text(); http=(root/'src/http.rs').read_text()
need('implementation_iteration: 7' in rt,'runtime iteration')
need('speech_inference_ready: audio_ready && local_ready && acoustic_ready' in rt,'speech ready status')
for x in ['stop_predictor.linear1.weight','stop_predictor.linear1.bias','stop_predictor.linear2.weight']:
    need(x in ac,f'missing {x}')
need('predict_stop_from_gpu_base' in ac,'stop predictor execution')
need('current_lm: GpuBuffer' in ac and 'predict_stop_from_current_lm' in ac,'canonical post-FSQ LM state')
need('record_current_lm_copy' in ac,'LM-state transition copy')
need('ae.current_lm_buffer()' in rt,'LocDiT bound to canonical LM state')
need('step_record_submit_gpu_only' in ac,'ResidualLM GPU-only fast path')
need('tokenizer.ggml.tokens' in gg and 'tokenizer.ggml.merges' in gg,'GGUF tokenizer arrays')
need('byte fallback' in tok.lower() and 'cjk_expansion' in tok,'specialized tokenizer')
need('pub fn synthesize' in rt and 'advance_generated_patch_gpu_only' in rt,'autoregressive loop')
need('streaming_prefix_len' in rt and '7680' in rt,'rolling 160ms streaming path')
need('POST' not in '', 'noop')
for route in ['/v1/audio/speech','/v1/audio/speech/stream','/v1/health','/v1/models/load','/v1/models/current','/v1/models/unload']:
    need(route in http,f'missing route {route}')
need('Transfer-Encoding: chunked' in http,'chunked streaming')
need('inference_gate' in http and 'loading: AtomicBool' in http,'model lifecycle serialization')
need('args.server && args.base_lm.is_none()' in main,'empty lifecycle server mode')
need('reference_audio' in http and 'prompt_audio' in http,'HTTP voice cloning')
need('--text' in (root/'README.md').read_text(),'README TTS')
for f in ['smoke_tts.bat','smoke_tts_stream.bat','smoke_voice_clone_reference.bat','smoke_voice_clone_continuation.bat']:
    need((root/'build_windows'/f).exists(),f'missing Windows {f}')
for f in ['smoke_tts.sh','smoke_tts_stream.sh','smoke_voice_clone_reference.sh','smoke_voice_clone_continuation.sh']:
    need((root/'build_linux'/f).exists(),f'missing Linux {f}')
need((root/'build_voxgen.bat').exists(),'missing root Windows master launcher')
need((root/'build_voxgen.sh').exists(),'missing root Linux master launcher')
root_scripts=[p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in {'.bat','.sh'}]
need(sorted(root_scripts)==['build_voxgen.bat','build_voxgen.sh'],f'unexpected root scripts: {root_scripts}')
win_scripts={p.stem for p in (root/'build_windows').glob('*.bat') if p.stem not in {'_common','build_voxgen'}}
lin_scripts={p.stem for p in (root/'build_linux').glob('*.sh') if p.stem not in {'_common','build_voxgen'}}
need(win_scripts==lin_scripts,f'Windows/Linux script parity mismatch win-only={sorted(win_scripts-lin_scripts)} linux-only={sorted(lin_scripts-win_scripts)}')
# Every Rust embedded SPIR-V target has a matching GLSL source, and vice versa.
spv=set()
for p in (root/'src').glob('*.rs'):
    spv.update(re.findall(r'/([A-Za-z0-9_]+)\.spv',p.read_text()))
sh={p.stem for p in (root/'shaders').glob('*.comp')}
need(spv==sh,f'SPIR-V/source mismatch embedded-only={sorted(spv-sh)} source-only={sorted(sh-spv)}')
need(len(sh)==53,f'expected 53 shaders, got {len(sh)}')
# GLSL reserves several C/C++-looking identifiers (notably `half`). Catch
# accidental use before glslc gets as far as Cargo's build script.
glsl_reserved=set('common partition active asm class union enum typedef template this packed goto inline noinline volatile public static extern external interface long short half fixed unsigned superp input output hvec2 hvec3 hvec4 fvec2 fvec3 fvec4 sampler3DRect filter sizeof cast namespace using row_major'.split())
for p in (root/'shaders').glob('*.comp'):
    text=p.read_text()
    text=re.sub(r'/\\*.*?\\*/',' ',text,flags=re.S)
    text=re.sub(r'//.*',' ',text)
    hits=sorted(set(re.findall(r'\\b[A-Za-z_]\\w*\\b',text)) & glsl_reserved)
    need(not hits,f'{p.name}: GLSL reserved identifier(s): {hits}')
# GLSL built-in function names can also be rejected as declared identifiers by
# glslc (for example a push-constant field named `length`). Restrict this check
# to declaration sites so ordinary calls such as max(...) remain valid.
glsl_builtin_names=set('radians degrees sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh pow exp log exp2 log2 sqrt inversesqrt abs sign floor trunc round roundEven ceil fract mod modf min max clamp mix step smoothstep isnan isinf floatBitsToInt floatBitsToUint intBitsToFloat uintBitsToFloat fma frexp ldexp packUnorm2x16 packSnorm2x16 packUnorm4x8 packSnorm4x8 unpackUnorm2x16 unpackSnorm2x16 unpackUnorm4x8 unpackSnorm4x8 packDouble2x32 unpackDouble2x32 length distance dot cross normalize faceforward reflect refract matrixCompMult outerProduct transpose determinant inverse lessThan lessThanEqual greaterThan greaterThanEqual equal notEqual any all not uaddCarry usubBorrow umulExtended imulExtended bitfieldExtract bitfieldInsert bitfieldReverse bitCount findLSB findMSB texture textureProj textureLod textureOffset texelFetch texelFetchOffset textureProjOffset textureLodOffset textureProjLod textureProjLodOffset textureGrad textureGradOffset textureProjGrad textureProjGradOffset textureSize textureQueryLod textureQueryLevels textureSamples interpolateAtCentroid interpolateAtSample interpolateAtOffset barrier memoryBarrier memoryBarrierAtomicCounter memoryBarrierBuffer memoryBarrierShared memoryBarrierImage groupMemoryBarrier atomicAdd atomicMin atomicMax atomicAnd atomicOr atomicXor atomicExchange atomicCompSwap imageSize imageLoad imageStore imageAtomicAdd imageAtomicMin imageAtomicMax imageAtomicAnd imageAtomicOr imageAtomicXor imageAtomicExchange imageAtomicCompSwap dFdx dFdy fwidth dFdxFine dFdyFine fwidthFine dFdxCoarse dFdyCoarse fwidthCoarse'.split())
glsl_decl_type=r'(?:uint|int|float|double|bool|vec[234]|ivec[234]|uvec[234]|bvec[234]|mat[234](?:x[234])?)'
for p in (root/'shaders').glob('*.comp'):
    text=p.read_text()
    text=re.sub(r'/\*.*?\*/',' ',text,flags=re.S)
    text=re.sub(r'//.*',' ',text)
    hits=[]
    for name in glsl_builtin_names:
        if re.search(r'\b'+glsl_decl_type+r'\s+'+re.escape(name)+r'\b', text):
            hits.append(name)
    need(not hits,f'{p.name}: GLSL built-in name used as identifier: {sorted(hits)}')
# Very small lexical delimiter audit for Rust source. It ignores strings/chars/comments.
def delimiters(text,name):
    stack=[]; pairs={')':'(',']':'[','}':'{'}; opens=set(pairs.values());i=0;state='code';raw_hash=0
    while i<len(text):
        c=text[i];n=text[i+1] if i+1<len(text) else ''
        if state=='line':
            if c=='\n':state='code'
        elif state=='block':
            if c=='*' and n=='/':state='code';i+=1
        elif state=='str':
            if c=='\\':i+=1
            elif c=='"':state='code'
        elif state=='char':
            if c=='\\':i+=1
            elif c=="'":state='code'
        else:
            if c=='/' and n=='/':state='line';i+=1
            elif c=='/' and n=='*':state='block';i+=1
            elif c=='"':state='str'
            elif c=="'":
                # lifetimes such as 'a are not character literals
                if i+2<len(text) and (text[i+1].isalpha() or text[i+1]=='_') and text[i+2] != "'": pass
                else: state='char'
            elif c in opens:stack.append((c,i))
            elif c in pairs:
                need(stack and stack[-1][0]==pairs[c],f'{name}: bad delimiter {c} at {i}, stack tail={stack[-3:]}');stack.pop()
        i+=1
    need(not stack,f'{name}: unclosed delimiters {stack[-5:]}')
for p in (root/'src').glob('*.rs'): delimiters(p.read_text(),p.name)
# Isolation: no app/server wrapper from Reading Companion lives in this standalone tree.
for p in root.rglob('*'):
    if p.is_file() and p.suffix.lower() in {'.rs','.toml','.bat','.sh'}:
        need('reading_companion' not in p.name.lower(),'application integration file leaked into VoxGen')
print(f'iteration7 static validation OK: {len(list((root/"src").glob("*.rs")))} Rust modules, {len(sh)} shaders')
