@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
if "%~1"=="" (echo Usage: %~nx0 speaker.wav & exit /b 2)
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --acoustic "%VOXGEN_ACOUSTIC%" --clone-mode reference --reference-wav "%~1" --control "warm and cheerful, conversational, with natural changes in emphasis" --text "VoxGen expressive speech control test." --max-steps 12 --output-wav test_expressive.wav
