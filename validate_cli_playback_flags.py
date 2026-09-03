from pathlib import Path

root = Path(__file__).resolve().parent
main = (root / "src/main.rs").read_text()
readme = (root / "README.md").read_text()

def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

need('#[arg(long = "speed", default_value_t = 100.0)]' in main, '--speed CLI flag')
need('#[arg(long = "pitch", default_value_t = 0.0)]' in main, '--pitch CLI flag')
need('--speed-percent' not in main, 'no --speed-percent alias in CLI source')
need('--pitch-semitones' not in main, 'no --pitch-semitones alias in CLI source')
need('`--speed`' in readme and '`--pitch`' in readme, 'README documents concise flags')
print('v0.7.55 CLI playback flag validation passed')
