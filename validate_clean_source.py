from pathlib import Path
import os
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parent


def need(condition, message):
    if not condition:
        raise AssertionError(message)

sh = root / "clean_source.sh"
bat = root / "clean_source.bat"
ps1 = root / "clean_source.ps1"
demo_sh = root / "demo" / "clean_source.sh"
demo_bat = root / "demo" / "clean_source.bat"

for p in (sh, bat, ps1, demo_sh, demo_bat):
    need(p.is_file(), f"missing {p.relative_to(root)}")

need(os.access(sh, os.X_OK), "clean_source.sh must be executable")
need(os.access(demo_sh, os.X_OK), "demo/clean_source.sh must be executable")
for p in (sh, demo_sh):
    proc = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    need(proc.returncode == 0, f"{p.name}: bash syntax error: {proc.stderr.strip()}")

shell = sh.read_text()
for token in (
    'release/voxgen', 'debug/voxgen', 'release/voxgen.exe', 'debug/voxgen.exe',
    'release/voxgen-demo', 'debug/voxgen-demo', 'release/voxgen-demo.exe', 'debug/voxgen-demo.exe',
    '"$ROOT/models"', '"$ROOT/demo/target"', 'Cargo.lock', 'test_tts.wav',
):
    need(token in shell, f"Linux cleaner missing contract token: {token}")

powershell = ps1.read_text()
for token in (
    'release\\voxgen.exe', 'debug\\voxgen.exe', 'release\\voxgen-demo.exe', 'debug\\voxgen-demo.exe',
    "'models'", "'demo\\target'", 'Cargo.lock', 'test_tts.wav', 'ReparsePoint',
):
    need(token in powershell, f"Windows cleaner missing contract token: {token}")
need('clean_source.ps1' in bat.read_text(), "Windows batch entry point must invoke clean_source.ps1")
need('../clean_source.sh' in demo_sh.read_text(), "demo Linux wrapper must invoke project cleaner")
need('..\\clean_source.bat' in demo_bat.read_text(), "demo Windows wrapper must invoke project cleaner")

# Functional Linux test in an isolated synthetic tree. This proves that final
# engine/demo binaries survive while build intermediates, local downloads and
# generated smoke outputs are removed.
with tempfile.TemporaryDirectory(prefix="voxgen-clean-") as tmp:
    fixture = Path(tmp) / "VoxGen"
    (fixture / "demo").mkdir(parents=True)
    shutil.copy2(sh, fixture / "clean_source.sh")
    os.chmod(fixture / "clean_source.sh", 0o755)

    for rel, payload in {
        "target/release/voxgen": b"engine-release",
        "target/debug/voxgen.exe": b"engine-debug-win",
        "target/release/deps/junk.rlib": b"junk",
        "demo/target/release/voxgen-demo": b"demo-release",
        "demo/target/debug/voxgen-demo.exe": b"demo-debug-win",
        "demo/target/release/deps/junk.rlib": b"junk",
        "models/model.gguf": b"model",
        "downloads/archive.bin": b"download",
        "demo/.cache/cache.bin": b"cache",
        "Cargo.lock": b"lock",
        "demo/Cargo.lock": b"lock",
        "test_tts.wav": b"generated",
        "test_base_hidden.f32": b"generated-fixture",
        "test_vae_pcm16k.f32": b"generated-fixture",
        "test_vae_input.wav": b"fixture-must-survive",
    }.items():
        p = fixture / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)

    proc = subprocess.run([str(fixture / "clean_source.sh")], cwd=tmp, capture_output=True, text=True)
    need(proc.returncode == 0, f"clean_source.sh fixture run failed: {proc.stderr}\n{proc.stdout}")

    for rel in (
        "target/release/voxgen",
        "target/debug/voxgen.exe",
        "demo/target/release/voxgen-demo",
        "demo/target/debug/voxgen-demo.exe",
        "test_vae_input.wav",
    ):
        need((fixture / rel).is_file(), f"cleaner removed preserved file: {rel}")

    for rel in (
        "target/release/deps/junk.rlib",
        "demo/target/release/deps/junk.rlib",
        "models",
        "downloads",
        "demo/.cache",
        "Cargo.lock",
        "demo/Cargo.lock",
        "test_tts.wav",
        "test_base_hidden.f32",
        "test_vae_pcm16k.f32",
    ):
        need(not (fixture / rel).exists(), f"cleaner left disposable artifact: {rel}")

print("clean-source validation passed")
