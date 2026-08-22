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

echo === Extracting any new/unprocessed clips in TrainingData ===
%PYTHON% build_dataset.py
if errorlevel 1 (
    echo build_dataset.py failed.
    pause
    exit /b 1
)

echo.
echo === Training dynamic (word) model ===
%PYTHON% train_model.py
if errorlevel 1 (
    echo train_model.py failed.
    pause
    exit /b 1
)

echo.
echo === Training static (pose/alphabet) model ===
%PYTHON% train_static_model.py

echo.
echo Done. Run run_recognize.bat to test live.
pause
