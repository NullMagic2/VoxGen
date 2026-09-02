@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --base-format q8_0 --max-context 4096 --baselm-token 1 --top-k 8
