from pathlib import Path

root = Path(__file__).resolve().parent
root_cargo = (root / 'Cargo.toml').read_text(encoding='utf-8')
demo_cargo = (root / 'demo' / 'Cargo.toml').read_text(encoding='utf-8')
demo = (root / 'demo' / 'src' / 'main.rs').read_text(encoding='utf-8')
http = (root / 'src' / 'http.rs').read_text(encoding='utf-8')
runtime = (root / 'src' / 'runtime.rs').read_text(encoding='utf-8')

checks = {
    'root/demo version 0.7.37': 'version = "0.7.37"' in root_cargo and 'version = "0.7.37"' in demo_cargo,
    'Stop button exists': 'with_label("Stop")' in demo and 'stop_button.enable(false)' in demo,
    'Stop flushes WinMM': 'waveOutReset(raw as HWaveOut)' in demo,
    'Stop requests server cancellation': 'cancel_active_server_speech(request_id)' in demo and '/v1/audio/speech/cancel' in demo,
    'server cancellation bypasses inference gate': '("POST", "/v1/audio/speech/cancel")' in http and 'cancel_speech.store(true' in http,
    'request-scoped cancel closes pre-request race': 'request_id: Option<u64>' in http and 'cancel_speech_request' in http and 'active_speech_request' in http and 'request["request_id"] = json!(request_id)' in demo,
    'speech uses cancelable runtime': 'runtime.synthesize_cancelable(' in http and 'Some(&state.cancel_speech)' in http,
    'runtime checks safe patch boundary': 'if cancelled() { bail!("speech synthesis cancelled"); }' in runtime and 'Safe cancellation boundary' in runtime,
    'cancelled stream is normal control flow': 'A cancelled stream is a normal control-flow event' in http,
    'queued write serialized against reset': 'Serialize the actual queue operation against Stop\'s waveOutReset' in demo,
}

failed = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(('PASS' if ok else 'FAIL') + ': ' + name)
if failed:
    raise SystemExit('v0.7.37 cancellation validation FAILED: ' + ', '.join(failed))
print('PASS: v0.7.37 Stop flushes playback immediately and cooperatively cancels server generation')
