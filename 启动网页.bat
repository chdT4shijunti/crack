@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 无人机裂缝识别系统 - 网页服务

REM --- 自动定位本机 conda 安装路径 ---
set "CONDA_BASE=%USERPROFILE%\miniconda3"
if not exist "%CONDA_BASE%\Scripts\activate.bat" set "CONDA_BASE=%USERPROFILE%\anaconda3"
if not exist "%CONDA_BASE%\Scripts\activate.bat" set "CONDA_BASE=C:\ProgramData\miniconda3"
if not exist "%CONDA_BASE%\Scripts\activate.bat" set "CONDA_BASE=C:\ProgramData\anaconda3"

if not exist "%CONDA_BASE%\Scripts\activate.bat" (
    echo [错误] 未找到 Anaconda / Miniconda，请先运行【环境初始化.bat】
    pause
    exit /b 1
)

call "%CONDA_BASE%\Scripts\activate.bat" uav_crack
if errorlevel 1 (
    echo [错误] 未找到 uav_crack 环境，请先双击运行【环境初始化.bat】
    pause
    exit /b 1
)

echo ============================================================
echo   正在启动无人机裂缝识别网页，请稍候约 20 秒...
echo   看到下面的网址后，用浏览器打开即可。
echo   本窗口不要关闭；关闭则网页停止。
echo ============================================================
echo.

python scripts\web_demo.py

echo.
echo 服务已退出。按任意键关闭窗口。
pause >nul
