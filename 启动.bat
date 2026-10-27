@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Matte Studio
set PYTHONIOENCODING=utf-8

echo.
echo   Matte Studio  -  Video to Transparent Frames
echo.

python -c "import matte_studio" 2>nul
if errorlevel 1 (
    echo   [ERROR] matte_studio is not installed for this Python.
    echo   Run this first:
    echo       pip install -e ".[onnx]"
    echo.
    pause
    exit /b 1
)

echo   Starting local service, browser will open shortly...
echo.
python -m matte_studio serve

echo.
echo   Service stopped.
pause
