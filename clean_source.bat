@echo off
setlocal EnableExtensions
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0clean_source.ps1"
if errorlevel 1 (
  echo [VoxGen clean] ERROR: cleanup failed.
  exit /b 1
)
exit /b 0
