@echo off
REM ---- Start backend + frontend in two windows ----

set "REPO_ROOT=%~dp0"

echo Repo root: %REPO_ROOT%

REM Backend window
start "Backend" cmd /k "%REPO_ROOT%start_backend.bat"

REM Frontend window
start "Frontend" cmd /k "%REPO_ROOT%start_frontend.bat"

echo Both servers launched (check the two new terminals).
pause

