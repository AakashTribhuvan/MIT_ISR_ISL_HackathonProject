@echo off
setlocal
cd /d "%~dp0"

python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

python extract.py %*

pause
