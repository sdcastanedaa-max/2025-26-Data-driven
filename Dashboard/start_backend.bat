@echo off
cd /d C:\Users\luas1\OneDrive\Documentos\SENSE\DDCEE\2025-26-Data-driven\Dashboard\Backend

REM
"C:\Users\luas1\anaconda3\envs\2025-26-Data-driven\python.exe" -m uvicorn main:app --reload --port 8000

pause
