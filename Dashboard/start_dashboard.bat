@echo off

echo Starting backend...
start cmd /k "cd /d C:\Users\luas1\OneDrive\Documentos\SENSE\DDCEE\2025-26-Data-driven\Dashboard\Backend && C:\Users\luas1\anaconda3\envs\2025-26-Data-driven\python.exe -m uvicorn main:app --reload --port 8000"

echo Starting frontend...
start cmd /k "cd /d C:\Users\luas1\OneDrive\Documentos\SENSE\DDCEE\2025-26-Data-driven\Dashboard\Frontend && npm run dev -- --host 127.0.0.1 --port 4173"

echo Both servers started.
pause
