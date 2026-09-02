@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1
if not exist "%VOXGEN_EXE%" (
  echo Missing %VOXGEN_EXE%. Run build_voxgen.bat from the project root first.
  exit /b 1
)
"%VOXGEN_EXE%" --base-lm "%VOXGEN_BASE_Q8%" --acoustic "%VOXGEN_ACOUSTIC%" --max-context 256 --base-residual-text-prefill 1,2,3,4
