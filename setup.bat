@echo off
rem KernalSounder - Windows setup
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.10+ is required: https://www.python.org/downloads/ & exit /b 1)
if not exist venv python -m venv venv
call "%~dp0venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt || exit /b 1
python download_models.py || exit /b 1
echo.
echo Setup complete.
echo   Start:              run.bat            (say "Hi Sounder")
echo   Start with console: run_console.bat --dashboard
echo   Start at login:     venv\Scripts\python kernal_sounder.py --install-autostart
