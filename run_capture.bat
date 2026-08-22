@echo off
setlocal
cd /d "%~dp0"
set PYTHON="C:\Users\Shamita\anaconda3\python.exe"

%PYTHON% -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

%PYTHON% capture_pose.py %*

pause
