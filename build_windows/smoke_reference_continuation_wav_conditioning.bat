@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
if not exist test_vae_input.wav "%VOXGEN_PYTHON%" make_test_vae_inputs.py || exit /b 1
rem The same deterministic WAV is used twice only to exercise both encoding roles in one prefix.
rem Reference is right-padded; prompt/continuation is left-padded.
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --acoustic "%VOXGEN_ACOUSTIC%" --base-format q8_0 --max-context 256 --conditioning-text-tokens 1,2,3 --reference-wav test_vae_input.wav --prompt-wav test_vae_input.wav
