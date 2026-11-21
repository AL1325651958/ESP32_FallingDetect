@echo off
cd /d "%~dp0..\.."
python project.py review %*
if errorlevel 1 pause
