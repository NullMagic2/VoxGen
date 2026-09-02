@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
if not exist test_vae_input.wav "%VOXGEN_PYTHON%" make_test_vae_inputs.py || exit /b 1
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --acoustic "%VOXGEN_ACOUSTIC%" --base-format q8_0 --max-context 256 --vae-roundtrip-wav test_vae_input.wav --vae-output-latents-f32 test_vae_roundtrip.f32 --vae-output-wav test_vae_roundtrip.wav
