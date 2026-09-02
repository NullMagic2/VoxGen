@echo off
setlocal
call "%~dp0_common.bat" || exit /b 1

set "MODE=%~1"
if "%MODE%"=="" set "MODE=release"

where cargo >nul 2>nul || (
  echo ERROR: cargo not found. Install Rust 1.78+ from https://rustup.rs/
  exit /b 1
)

if defined VOXGEN_GLSLC (
  if not exist "%VOXGEN_GLSLC%" (
    echo ERROR: VOXGEN_GLSLC points to a missing file: %VOXGEN_GLSLC%
    exit /b 1
  )
) else if defined VULKAN_SDK (
  if exist "%VULKAN_SDK%\Bin\glslc.exe" set "VOXGEN_GLSLC=%VULKAN_SDK%\Bin\glslc.exe"
)
if not defined VOXGEN_GLSLC (
  where glslc >nul 2>nul || (
    echo ERROR: glslc not found. Install the Vulkan SDK or set VOXGEN_GLSLC.
    exit /b 1
  )
)

if /I "%MODE%"=="clean" (
  echo [VoxGen] cargo clean
  cargo clean
  exit /b %errorlevel%
)
if /I "%MODE%"=="check" (
  echo [VoxGen] cargo check + Vulkan shader compilation
  cargo check
  exit /b %errorlevel%
)
if /I "%MODE%"=="debug" (
  echo [VoxGen] Building debug binary and Vulkan shaders...
  cargo build
  if errorlevel 1 exit /b %errorlevel%
  set "BUILT_EXE=%VOXGEN_ROOT%\target\debug\voxgen.exe"
) else if /I "%MODE%"=="release" (
  echo [VoxGen] Building release binary and Vulkan shaders...
  cargo build --release
  if errorlevel 1 exit /b %errorlevel%
  set "BUILT_EXE=%VOXGEN_ROOT%\target\release\voxgen.exe"
) else (
  echo Usage: build_voxgen.bat [release^|debug^|check^|clean] [--no-probe]
  exit /b 2
)

echo.
echo [VoxGen] Build succeeded: %BUILT_EXE%
if /I "%~2"=="--no-probe" exit /b 0
echo [VoxGen] Probing Vulkan devices...
"%BUILT_EXE%" --list-devices
exit /b %errorlevel%
