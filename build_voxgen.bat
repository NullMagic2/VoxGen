@echo off
setlocal
call "%~dp0build_windows\build_voxgen.bat" %*
exit /b %errorlevel%
