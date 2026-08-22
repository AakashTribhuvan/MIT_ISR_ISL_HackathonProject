@echo off
setlocal
cd /d "%~dp0"
set PYTHON="C:\Users\Shamita\anaconda3\python.exe"

%PYTHON% list_training_files.py
start "" http://localhost:8000
%PYTHON% dashboard_server.py
