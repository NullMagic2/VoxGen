# VoxGen v0.7.37 mid-generation cancellation validation

v0.7.37 adds a dedicated **Stop** button to the wxDragon demo.

The local cancellation path first sets a shared atomic flag and, on Windows, calls `waveOutReset` on the active WinMM device to flush already queued PCM immediately. The client then posts `/v1/audio/speech/cancel` with the active request ID. The server cancellation endpoint does not acquire `inference_gate`, so it remains callable while speech synthesis owns that lifecycle lock. Request-scoped cancellation also closes the race where Stop is clicked just before the speech POST reaches the server.

The runtime checks cancellation only between completed GPU operations / acoustic patches. It never attempts to interrupt an in-flight Vulkan submission. The next synthesis request resets the cancellation flag after obtaining the inference gate.

Run:

```text
python validate_mid_generation_cancel.py
```
