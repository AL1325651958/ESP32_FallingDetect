@echo off
cd /d "%~dp0"
python project.py verify %*
if errorlevel 1 pause
