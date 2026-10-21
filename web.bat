@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Environment not set up yet. Double-click setup.bat first.
  pause
  exit /b 1
)
rem The page keeps looking for the backend, so it can open before the models have loaded
start "" http://localhost:8000
".venv\Scripts\python.exe" server.py %*
