@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
if not exist test_locdit_x.f32 "%VOXGEN_PYTHON%" make_test_local_inputs.py || exit /b 1
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --acoustic "%VOXGEN_ACOUSTIC%" --base-format q8_0 --max-context 256 --locdit-x-f32 test_locdit_x.f32 --locdit-cond-f32 test_locdit_cond.f32 --locdit-mu-f32 test_locdit_mu.f32 --locdit-t 0.5 --locdit-dt 0.0
