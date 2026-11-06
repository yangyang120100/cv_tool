@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHON_CMD=python"
if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_CMD=%~dp0.venv\Scripts\python.exe"
"%PYTHON_CMD%" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Streamlit is not installed in the selected Python environment.
  echo Install dependencies with:
  echo   python -m pip install -r requirements-web.txt
  echo Advanced tools may also need requirements-optional.txt.
  pause
  exit /b 1
)
"%PYTHON_CMD%" -m streamlit run web_app.py --server.address 127.0.0.1 --server.port 8000
if errorlevel 1 (
  echo [ERROR] Web app stopped with an error. See the messages above.
  pause
)
