@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
if "%~2"=="" (echo Usage: %~nx0 prompt.wav "Exact prompt transcript. " & exit /b 2)
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --acoustic "%VOXGEN_ACOUSTIC%" --prompt-wav "%~1" --prompt-text "%~2" --text "This is the continuation." --max-steps 12 --output-wav test_continuation.wav
