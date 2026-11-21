@echo off
cd /d "%~dp0"
set /p FLASH_PORT=Serial port (COM3): 
if not defined FLASH_PORT exit /b 1
python flash.py --port "%FLASH_PORT%"
pause
