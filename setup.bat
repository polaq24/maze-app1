@echo off
chcp 65001 >nul
title Maze Capital Terminal - Installer
cd /d "%~dp0"
echo Maze Capital Terminal - Installer
echo ====================================
echo.
echo Cerco Python...
where python >nul 2>nul
if errorlevel 1 (
    echo Python non trovato.
    echo Scarica da: https://www.python.org/downloads/
    echo SPUNTA "Add Python to PATH" durante l'installazione.
    pause
    start https://www.python.org/downloads/
    exit /b
)
python --version
echo.
echo Installo dipendenze...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Errore installazione. Riprova con:
    echo   python -m pip install -r requirements.txt
    pause
    exit /b
)
echo.
echo Fatto! Avvio terminale...
python -m streamlit run terminal.py
pause
