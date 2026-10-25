@echo off
cd /d "%~dp0"

if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run app.py
) else (
    streamlit run app.py
)

if errorlevel 1 pause
