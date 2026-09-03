from pathlib import Path
import os, subprocess, sys

root = Path(__file__).resolve().parent
win = root / 'build_windows'
lin = root / 'build_linux'

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

root_scripts = sorted(p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in {'.bat', '.sh'})
need(root_scripts == ['build_voxgen.bat', 'build_voxgen.sh', 'clean_source.bat', 'clean_source.sh'], f'root scripts must be master build/clean entry points only: {root_scripts}')
need((win / 'build_voxgen.bat').exists(), 'missing Windows platform build script')
need((lin / 'build_voxgen.sh').exists(), 'missing Linux platform build script')
need((win / '_common.bat').exists(), 'missing Windows common script')
need((lin / '_common.sh').exists(), 'missing Linux common script')

w = {p.stem for p in win.glob('*.bat') if p.stem not in {'_common', 'build_voxgen'}}
l = {p.stem for p in lin.glob('*.sh') if p.stem not in {'_common', 'build_voxgen'}}
need(w == l, f'platform script parity mismatch: Windows-only={sorted(w-l)}, Linux-only={sorted(l-w)}')

for p in win.glob('*.bat'):
    text = p.read_text(errors='replace').lower()
    need('%~dp0target' not in text, f'{p.name}: stale root-relative target assumption')
    if p.name != '_common.bat':
        need('c:\\software\\voxcpm-q8\\models' not in text, f'{p.name}: model path should come from _common.bat')

bash = os.environ.get('BASH', 'bash')
for p in [root / 'build_voxgen.sh', root / 'clean_source.sh', root / 'demo' / 'clean_source.sh', *sorted(lin.glob('*.sh'))]:
    need(os.access(p, os.X_OK), f'{p}: not executable')
    need(p.read_text().startswith('#!/usr/bin/env bash'), f'{p}: missing portable bash shebang')
    proc = subprocess.run([bash, '-n', str(p)], capture_output=True, text=True)
    need(proc.returncode == 0, f'{p}: bash syntax error: {proc.stderr.strip()}')

print(f'build layout validation OK: {len(w)} paired smoke/bench scripts + platform build helpers')
