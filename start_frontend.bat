@echo off
REM ---- Start React/Vite frontend ----

set "REPO_ROOT=%~dp0"

cd /d "%REPO_ROOT%Dashboard\Frontend"

npm run dev -- --host 127.0.0.1 --port 4173

pause
