@echo off
chcp 65001
:menu
cls
echo ========================================
echo       SpotSync Server Manager
echo ========================================
echo 1. Start All Servers
echo 2. Stop All Servers
echo 3. Exit
echo ========================================
set /p choice="Select an option (1-3): "

if "%choice%"=="1" goto start_servers
if "%choice%"=="2" goto stop_servers
if "%choice%"=="3" goto exit

goto menu

:start_servers
echo Starting Local AI Search (v9) Server...
start "AI Search v9 (Port 8000)" cmd /c "cd /d %~dp0 && call .venv\Scripts\activate && uvicorn ai_search_v9:app --host 0.0.0.0 --port 8000"

echo Starting Vertex AI Search Server...
start "Vertex AI Search (Port 8001)" cmd /c "cd /d %~dp0 && call .venv\Scripts\activate && uvicorn ai_search_vertex:app --host 0.0.0.0 --port 8001"

echo Starting Legacy Local Search Server...
start "Legacy Local Search (Port 8002)" cmd /c "cd /d %~dp0 && call .venv\Scripts\activate && uvicorn ai_search_legacy:app --host 0.0.0.0 --port 8002"

echo Starting Frontend Server...
start "Frontend (Port 5173)" cmd /c "cd /d %~dp0\frontend && npm run dev"

echo All servers have been started in new windows!
pause
goto menu

:stop_servers
echo Stopping Servers...
taskkill /FI "WINDOWTITLE eq AI Search v9*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Vertex AI Search*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Legacy Local Search*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Frontend*" /T /F >nul 2>&1
echo Servers have been stopped!
pause
goto menu

:exit
exit
