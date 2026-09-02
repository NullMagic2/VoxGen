@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
if not exist test_current_embed.f32 "%VOXGEN_PYTHON%" make_test_embeddings.py || exit /b 1
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_F16%" --acoustic "%VOXGEN_ACOUSTIC%" --base-format f16 --max-context 256 --base-residual-embedding-f32 test_current_embed.f32
