@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
if not exist test_cfm_initial_x.f32 "%VOXGEN_PYTHON%" make_test_local_inputs.py || exit /b 1
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --acoustic "%VOXGEN_ACOUSTIC%" --base-format q8_0 --max-context 256 --cfm-mu-f32 test_locdit_mu.f32 --cfm-cond-f32 test_locdit_cond.f32 --cfm-initial-x-f32 test_cfm_initial_x.f32 --cfm-steps 10 --cfm-cfg 2.0 --cfm-bench --bench-warmup 2 --bench-iters 10
