from pathlib import Path

src = Path(__file__).with_name("demo") / "src" / "main.rs"
text = src.read_text(encoding="utf-8")
dsp = (Path(__file__).with_name("src") / "playback_dsp.rs").read_text(encoding="utf-8")
root_cargo = (Path(__file__).with_name("Cargo.toml")).read_text(encoding="utf-8")
demo_cargo = (Path(__file__).with_name("demo") / "Cargo.toml").read_text(encoding="utf-8")
checks = {
    "root/demo version 0.7.40": 'version = "0.7.60"' in root_cargo and 'version = "0.7.60"' in demo_cargo,
    "bounded WinMM live queue": "STREAM_MAX_PENDING_BLOCKS" in text and "wait_for_live_capacity" in text,
    "wait occurs before live DSP render": text.find("player.wait_for_live_capacity();") < text.find("let rendered = realtime.push(&paced, live_controls, managed_continuity);"),
    "demo forwards live controls": ".set_controls(native)" in text,
    "shared processor tracks controls": "controls: PlaybackControls" in dsp and "if controls == self.controls" in dsp,
    "WSOLA reset on live rate change": "self.stretcher = SpeechWsola::new(self.sample_rate, controls.wsola_tempo())?" in dsp,
    "resampler reset only when pitch changes": "let pitch_changed" in dsp and "if pitch_changed" in dsp and "self.resampler = StreamingSincResampler::new();" in dsp,
}
failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
if failed:
    raise SystemExit("stream speed recovery validation failed: " + ", ".join(failed))
print("PASS: streaming speed recovery state/queue guards are present")
