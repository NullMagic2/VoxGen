@echo off
rem Shared path/model setup for VoxGen Windows scripts.
for %%I in ("%~dp0..") do set "VOXGEN_ROOT=%%~fI"
cd /d "%VOXGEN_ROOT%" || exit /b 1

if not defined VOXGEN_MODEL_DIR (
  if exist "%VOXGEN_ROOT%\models\VoxCPM2-Acoustic-F16.gguf" (
    set "VOXGEN_MODEL_DIR=%VOXGEN_ROOT%\models"
  ) else (
    set "VOXGEN_MODEL_DIR=C:\Software\VoxCPM-Q8\models"
  )
)
if not defined VOXGEN_BASE_Q8 set "VOXGEN_BASE_Q8=%VOXGEN_MODEL_DIR%\VoxCPM2-BaseLM-Q8_0.gguf"
if not defined VOXGEN_BASE_F16 set "VOXGEN_BASE_F16=%VOXGEN_MODEL_DIR%\VoxCPM2-BaseLM-F16.gguf"
if not defined VOXGEN_ACOUSTIC set "VOXGEN_ACOUSTIC=%VOXGEN_MODEL_DIR%\VoxCPM2-Acoustic-F16.gguf"
if not defined VOXGEN_EXE set "VOXGEN_EXE=%VOXGEN_ROOT%\target\release\voxgen.exe"
if not defined VOXGEN_PYTHON set "VOXGEN_PYTHON=python"
exit /b 0
