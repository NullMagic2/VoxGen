from pathlib import Path

src = Path(__file__).with_name("demo") / "src" / "main.rs"
text = src.read_text(encoding="utf-8")
root_cargo = (Path(__file__).with_name("Cargo.toml")).read_text(encoding="utf-8")
demo_cargo = (Path(__file__).with_name("demo") / "Cargo.toml").read_text(encoding="utf-8")
checks = {
    "root/demo version 0.7.37": 'version = "0.7.37"' in root_cargo and 'version = "0.7.37"' in demo_cargo,
    "bounded WinMM live queue": "STREAM_MAX_PENDING_BLOCKS" in text and "wait_for_live_capacity" in text,
    "wait occurs before live DSP render": text.find("player.wait_for_live_capacity();") < text.find("let rendered = realtime.push(&paced, live_controls);"),
    "processor tracks prior speed": "last_speed_percent" in text,
    "processor tracks prior pitch": "last_pitch_semitones" in text,
    "WSOLA reset on live rate change": "reset_for_live_rate_change" in text and "self.stretcher = TimeStretch::new" in text,
    "resampler reset only when pitch changes": "if pitch_semitones != self.last_pitch_semitones" in text and "self.resampler = StreamingSincResampler::new();" in text,
}
failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}: {name}")
if failed:
    raise SystemExit("stream speed recovery validation failed: " + ", ".join(failed))
print("PASS: streaming speed recovery state/queue guards are present")
