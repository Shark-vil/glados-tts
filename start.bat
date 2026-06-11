@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -W ignore glados_api.py --port 8016
pause