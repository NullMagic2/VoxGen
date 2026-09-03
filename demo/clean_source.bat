@echo off
setlocal
call "%~dp0..\clean_source.bat"
exit /b %errorlevel%
