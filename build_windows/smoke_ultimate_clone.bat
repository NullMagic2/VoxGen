@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
if "%~2"=="" (echo Usage: %~nx0 expressive-reference.wav "exact reference transcript" & exit /b 2)
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --acoustic "%VOXGEN_ACOUSTIC%" --clone-mode ultimate --reference-wav "%~1" --prompt-text "%~2" --text "VoxGen Ultimate cloning test." --max-steps 12 --output-wav test_ultimate.wav
