@echo off
setlocal
set "DEMO_DIR=%~dp0"
echo [VoxGen Demo] Building wxDragon demo...
cargo build --manifest-path "%DEMO_DIR%Cargo.toml" --release
if errorlevel 1 exit /b %errorlevel%
echo [VoxGen Demo] Built: %DEMO_DIR%target\release\voxgen-demo.exe
