@echo off
setlocal
set "DEMO_DIR=%~dp0"
for %%I in ("%DEMO_DIR%..") do set "VOXGEN_ROOT=%%~fI"
if not exist "%DEMO_DIR%target\release\voxgen-demo.exe" call "%DEMO_DIR%build_demo.bat"
if errorlevel 1 exit /b %errorlevel%
cd /d "%VOXGEN_ROOT%"
"%DEMO_DIR%target\release\voxgen-demo.exe"
