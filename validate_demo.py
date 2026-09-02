from pathlib import Path
import os, re, subprocess

root = Path(__file__).resolve().parent
demo = root / "demo"

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

need(demo.is_dir(), "missing demo directory")
for rel in ["Cargo.toml", "README.md", "src/main.rs", "build_demo.bat", "run_demo.bat", "build_demo.sh", "run_demo.sh"]:
    need((demo / rel).is_file(), f"missing demo/{rel}")

cargo = (demo / "Cargo.toml").read_text()
need('name = "voxgen-demo"' in cargo, "demo package name")
need('wxdragon = "=0.9.20"' in cargo, "wxDragon pin")
need('rust-version = "1.87"' in cargo, "WSOLA-compatible Rust MSRV declaration")

src = (demo / "src/main.rs").read_text()
for token in [
    "TextCtrlStyle::MultiLine",
    "update_emotion_sample_button",
    "Select {emotion} sample...",
    "Select BaseLM component...",
    "Select Acoustic component...",
    "Load VoxCPM2",
    'Button::builder(&panel).with_label("Speak")',
    '"/v1/audio/speech"',
    '"/v1/models/load"',
    '"/v1/models/current"',
    '"reference_audio_path"',
    '"seed": seed',
    "Sound::play_file",
    "SoundFlags::Async",
    "pcm16_wav_from_voxgen",
    'POST /v1/audio/speech/stream HTTP/1.1',
    "WaveOutPlayer",
    "waveOutOpen",
    "DEFAULT_GAIN_PERCENT",
    "DEMO_ONSET_FADE_SAMPLES",
    "VOXGEN_DEMO_SEED",
    "WordSpacingProcessor",
    "SpinCtrl::builder",
    'with_label("Word spacing (ms):")',
    "DEFAULT_WORD_SPACING_MS",
    "WORD_GAP_MIN_MS",
    "WORD_GAP_MAX_MS",
    "ensure_server",
    "HealthState::EngineOnly",
    "VOXGEN_MODEL_DIR",
    "VOXGEN_BASE_MODEL",
    "VOXGEN_ACOUSTIC",
    "validate_voice_wav",
    "input.enable(false)",
]:
    need(token in src, f"demo contract missing {token}")
need("SizerFlag::None" not in src, "invalid SizerFlag::None")
need("event.event.skip" not in src, "outdated WindowEventData field access")
need("event.skip(true)" in src, "close event propagation")
need("input.clear()" not in src, "successful synthesis must preserve the user's input text")
need("const DEFAULT_GAIN_PERCENT: u32 = 140" in src, "configurable playback gain default")
need("const DEMO_ONSET_FADE_SAMPLES: usize = 1_440" in src, "30-ms onset fade")
need("const DEFAULT_WORD_SPACING_MS: u32 = 30" in src, "default word spacing")
need(".with_range(0, 100)" in src, "word spacing range")
need("word_spacing_control_copy" in src and ".value()" in src, "word spacing value capture")
need("self.quiet_run_samples" in src and "out.resize(out.len() + self.extra_samples, 0.0)" in src, "streaming gap expansion")
need("POST /v1/audio/speech/stream HTTP/1.1" in src, "Windows low-latency streaming request")
need("playback_file: Option<PathBuf>" in src, "playback temp-file lifetime tracking")
need("AtomicU64" in src, "unique playback file generation")
need("base_model: Option<PathBuf>" in src and "acoustic_model: Option<PathBuf>" in src, "explicit model selections in demo state")
need("fn is_voxgen_root" in src, "structural VoxGen root predicate")
need("fn search_up_for_voxgen_root" in src, "upward VoxGen root search")
need("env::current_exe()" in src, "binary-location fallback for project root")
need("Portable deployment: if voxgen lives beside voxgen-demo" in src, "adjacent engine discovery")
need("exe.parent().map(|dir| dir.join(name))" in src, "adjacent engine PathBuf construction")
need("if let Some(p) = adjacent.as_ref()" in src, "adjacent engine candidate check")
need("copy it next to the demo" in src, "portable deployment diagnostic")
# Discovery precedence must remain: explicit override -> adjacent portable binary -> project build outputs.
idx_override = src.index('env::var_os("VOXGEN_BIN")')
idx_adjacent = src.index('let adjacent = env::current_exe()')
idx_release = src.index('root.join("target").join("release").join(name)', idx_adjacent)
need(idx_override < idx_adjacent < idx_release, "engine discovery precedence")
need('root.join("target").join("release").join(name)' in src, "PathBuf engine binary construction")
need('root.join("target").join("debug").join(name)' in src, "PathBuf debug engine construction")
need('{}/target' not in src, "mixed-separator engine error construction must not return")
need('demo/target/release/voxgen-demo.exe -> .../VoxGen' in src, "nested demo target regression comment")

# Cheap lexical delimiter audit for the standalone demo source.
def delimiters(text, name):
    stack=[]; pairs={')':'(',']':'[','}':'{'}; opens=set(pairs.values()); i=0; state='code'
    while i < len(text):
        c=text[i]; n=text[i+1] if i+1<len(text) else ''
        if state=='line':
            if c=='\n': state='code'
        elif state=='block':
            if c=='*' and n=='/': state='code'; i+=1
        elif state=='str':
            if c=='\\': i+=1
            elif c=='"': state='code'
        elif state=='char':
            if c=='\\': i+=1
            elif c=="'": state='code'
        else:
            if c=='/' and n=='/': state='line'; i+=1
            elif c=='/' and n=='*': state='block'; i+=1
            elif c=='"': state='str'
            elif c=="'":
                if i+2<len(text) and (text[i+1].isalpha() or text[i+1]=='_') and text[i+2] != "'": pass
                else: state='char'
            elif c in opens: stack.append((c,i))
            elif c in pairs:
                need(stack and stack[-1][0]==pairs[c], f"{name}: bad delimiter {c} at {i}")
                stack.pop()
        i+=1
    need(not stack, f"{name}: unclosed delimiters {stack[-5:]}")

delimiters(src, "demo/src/main.rs")

for sh in [demo / "build_demo.sh", demo / "run_demo.sh"]:
    need(os.access(sh, os.X_OK), f"{sh.name} not executable")
    proc = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    need(proc.returncode == 0, f"{sh.name}: {proc.stderr.strip()}")

# Root stays clean: platform-specific VoxGen scripts remain in their folders;
# demo launchers are intentionally scoped to demo/.
root_scripts=sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in {'.bat','.sh'})
need(root_scripts == ['build_voxgen.bat','build_voxgen.sh'], f"unexpected root scripts {root_scripts}")
print("wxDragon demo static validation OK")
