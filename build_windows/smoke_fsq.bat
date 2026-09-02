@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
if not exist test_base_hidden.f32 "%VOXGEN_PYTHON%" make_test_embeddings.py || exit /b 1
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --acoustic "%VOXGEN_ACOUSTIC%" --base-format q8_0 --max-context 256 --fsq-input-f32 test_base_hidden.f32
