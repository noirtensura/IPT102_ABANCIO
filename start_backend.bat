@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Create it with:
    echo   python -m venv .venv
    echo   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo Starting Skin Disease Detection Python backend on http://127.0.0.1:5000 ...
echo Frontend URL (XAMPP): http://localhost/IPT_ABANCIO/scan.php
".venv\Scripts\python.exe" -B backend\app.py
