@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Matte Studio
set PYTHONIOENCODING=utf-8

echo.
echo   Matte Studio  -  Video to Transparent Frames
echo   Starting local service, browser will open shortly...
echo.

python -m matte_studio serve

echo.
echo   Service stopped.
pause
