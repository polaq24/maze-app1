@echo off
cd /d "%~dp0"
echo [1] Streamlit Terminal (old)
echo [2] WebSocket Server + Dashboard (new, real-time)
choice /c 12 /n /m "Scegli [1/2]: "
if errorlevel 2 goto server
python -m streamlit run terminal.py
goto end
:server
python server.py
:end
pause
