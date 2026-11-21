@echo off
cd /d "%~dp0"
python project.py build %*
if errorlevel 1 pause
