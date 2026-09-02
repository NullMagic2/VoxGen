@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
if not exist test_locenc_patch.f32 "%VOXGEN_PYTHON%" make_test_local_inputs.py || exit /b 1
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --acoustic "%VOXGEN_ACOUSTIC%" --base-format q8_0 --max-context 256 --locenc-patch-f32 test_locenc_patch.f32
