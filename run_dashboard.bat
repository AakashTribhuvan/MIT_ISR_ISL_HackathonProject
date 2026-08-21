@echo off
setlocal
cd /d "%~dp0"

python list_training_files.py

cd /d "%~dp0dashboard"
start "" http://localhost:8000
python -m http.server 8000
