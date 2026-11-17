@echo off
rem Starts KernalSounder without a console window. Use run_console.bat to see logs.
cd /d "%~dp0"
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0kernal_sounder.py" %*
