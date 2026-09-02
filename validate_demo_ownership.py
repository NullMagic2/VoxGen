from pathlib import Path

root = Path(__file__).resolve().parent
src = (root / 'demo/src/main.rs').read_text()
cargo = (root / 'demo/Cargo.toml').read_text()


def need(cond, msg):
    if not cond:
        raise AssertionError(msg)

need('version = "0.7.37"' in cargo, 'demo version')
needle = 'let speak_live_playback_controls = live_playback_controls.clone();'
need(needle in src, 'Speak callback does not get a pre-move controls clone')

speak = src.index('speak_button.on_click(move |_| {')
close = src.index('frame.on_close(move |event| {', speak)
block = src[speak:close]
need('speak_live_playback_controls.clone()' in block,
     'Speak callback does not use its dedicated controls clone')
need('                    live_playback_controls.clone(),' not in block,
     'Speak move closure still captures the outer live_playback_controls')
need('let controls = live_playback_controls.clone();' in src[close-1200:close],
     'outer controls handle is not preserved for close/settings callback')

print('demo ownership validation OK')
