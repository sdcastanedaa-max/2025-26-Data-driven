@echo off
REM ---- Start FastAPI backend (uvicorn) ----

REM Folder where this .bat lives (repo root)
set "REPO_ROOT=%~dp0"

REM Go to Backend folder
cd /d "%REPO_ROOT%Dashboard\Backend"

REM Use your conda env Python (adjust if yours is different)
"C:\Users\luas1\anaconda3\envs\2025-26-Data-driven\python.exe" -m uvicorn main:app --reload --port 8000

pause
