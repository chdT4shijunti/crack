@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 无人机裂缝识别系统 - 环境初始化（仅首次运行）

echo ============================================================
echo   无人机裂缝识别系统 - 环境初始化（只需运行一次）
echo ============================================================
echo.

REM --- 自动定位本机 conda 安装路径 ---
set "CONDA_BASE=%USERPROFILE%\miniconda3"
if not exist "%CONDA_BASE%\Scripts\activate.bat" set "CONDA_BASE=%USERPROFILE%\anaconda3"
if not exist "%CONDA_BASE%\Scripts\activate.bat" set "CONDA_BASE=C:\ProgramData\miniconda3"
if not exist "%CONDA_BASE%\Scripts\activate.bat" set "CONDA_BASE=C:\ProgramData\anaconda3"

if not exist "%CONDA_BASE%\Scripts\activate.bat" (
    echo [错误] 未找到 Anaconda / Miniconda。
    echo 请先安装 Miniconda: https://docs.conda.io/en/latest/miniconda.html
    echo 安装时默认选项即可，装完再双击本脚本。
    pause
    exit /b 1
)

echo 使用 conda: %CONDA_BASE%
echo.
echo [1/3] 创建 uav_crack 环境（Python 3.11）...
call "%CONDA_BASE%\Scripts\activate.bat"
conda create -n uav_crack python=3.11 -y
if errorlevel 1 ( echo [错误] 环境创建失败 & pause & exit /b 1 )

call "%CONDA_BASE%\Scripts\activate.bat" uav_crack
echo.
echo [2/3] 安装 PyTorch（CUDA 12.8，无 N 卡可改用 CPU 版）...
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
echo.
echo [3/3] 安装其余依赖...
pip install -r requirements.txt

echo.
echo ============================================================
echo   环境就绪！以后每次使用，只需双击【启动网页.bat】
echo ============================================================
pause
