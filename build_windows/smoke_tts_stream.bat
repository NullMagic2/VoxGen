@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --acoustic "%VOXGEN_ACOUSTIC%" --text "VoxGen rolling streaming decoder test." --max-steps 12 --stream on --output-wav test_tts_stream.wav
