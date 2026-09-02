@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --acoustic "%VOXGEN_ACOUSTIC%" --text "VoxGen end to end speech test." --max-steps 12 --output-wav test_tts.wav
