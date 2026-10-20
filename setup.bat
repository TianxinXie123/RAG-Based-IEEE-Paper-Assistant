@echo off
cd /d "%~dp0"
echo Setting up the RAG PDF reader environment. Please wait...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
echo.
echo Full log saved to setup_log.txt
pause
