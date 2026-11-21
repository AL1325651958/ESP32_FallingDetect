@echo off
cd /d "%~dp0"
python project.py train %*
if errorlevel 1 pause
