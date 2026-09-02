@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
if not exist test_reference_latents.f32 "%VOXGEN_PYTHON%" make_test_local_inputs.py || exit /b 1
rem Token IDs 1,2,3 are deterministic plumbing inputs, not meaningful speech text.
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --acoustic "%VOXGEN_ACOUSTIC%" --base-format q8_0 --max-context 256 --conditioning-text-tokens 1,2,3 --reference-latents-f32 test_reference_latents.f32
