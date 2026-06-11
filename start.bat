@echo off
chcp 65001 >nul
title 多轮对话系统 - Qwen3.5-0.8B
setlocal enabledelayedexpansion

:: ============================================
:: 多轮对话系统 — 一键部署启动
:: 用法：双击 start.bat 或在 cmd 中运行
:: ============================================

set PROJECT_DIR=%~dp0
set MODEL_DIR=%PROJECT_DIR%Qwen3.5-0.8B
set LORA_DIR=%PROJECT_DIR%outputs_multi_turn\lora_adapter

echo ╔══════════════════════════════════════════╗
echo ║     多轮对话系统 — 一键启动             ║
echo ║     基于 Qwen3.5-0.8B + LoRA            ║
echo ╚══════════════════════════════════════════╝
echo.

:: ========== 1. 检查 Python ==========
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未检测到 Python，请安装 Python 3.10+
    echo    下载: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
echo [1/5] ✅ Python %PY_VER%

:: ========== 2. 安装依赖 ==========
echo [2/5] 安装 Python 依赖...
pip install -q torch transformers datasets peft accelerate fastapi uvicorn jinja2 2>nul
if %errorlevel% neq 0 (
    echo   ⏳ 首次安装可能较慢，请耐心等待...
    pip install torch transformers datasets peft accelerate fastapi uvicorn jinja2
)
pip show transformers >nul 2>&1
if %errorlevel% equ 0 (echo   ✅ 依赖就绪) else (echo   ⚠️ 部分依赖可能缺失)

:: ========== 3. 下载模型 ==========
echo [3/5] 检查模型文件...
if not exist "%MODEL_DIR%\config.json" (
    echo   ⏳ 模型不存在，开始下载（约 1.7GB，首次需等待）...
    echo   下载源: HF 镜像 (hf-mirror.com)
    echo.

    :: 优先用 hf 命令，没有则用 pip + python 下载
    where hf >nul 2>&1
    if !errorlevel! equ 0 (
        set HF_ENDPOINT=https://hf-mirror.com
        hf download Qwen/Qwen3.5-0.8B --local-dir "%MODEL_DIR%"
    ) else (
        echo   正在通过 huggingface_hub 下载...
        pip install -q huggingface_hub
        python -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-0.8B', local_dir=r'%MODEL_DIR%', local_dir_use_symlinks=False)
"
    )

    if not exist "%MODEL_DIR%\config.json" (
        echo ❌ 模型下载失败，请手动下载：
        echo    set HF_ENDPOINT=https://hf-mirror.com
        echo    hf download Qwen/Qwen3.5-0.8B --local-dir "%MODEL_DIR%"
        pause
        exit /b 1
    )
    echo   ✅ 模型下载完成
) else (
    echo   ✅ 模型已就绪
)

:: ========== 4. 检查 LoRA 适配器 ==========
echo [4/5] 检查 LoRA 适配器...
if not exist "%LORA_DIR%\adapter_config.json" (
    echo.
    echo   ────────────────────────────────────────────
    echo   首次使用，可以用示例数据快速训练一个模型：
    echo      python multi_turn_finetune_qwen_cpu.py
    echo   （约 3-5 分钟，不训练也能启动，但回答质量一般）
    echo   ────────────────────────────────────────────
    echo.
) else (
    echo   ✅ LoRA 适配器已就绪
)

:: ========== 5. 启动服务 ==========
echo [5/5] 启动 Web 服务...
echo.
echo ╔══════════════════════════════════════════╗
echo ║  服务启动中（加载模型约 30-60 秒）...   ║
echo ║                                         ║
echo ║  🌐 聊天界面: http://localhost:8000      ║
echo ║  📚 API 文档: http://localhost:8000/docs ║
echo ║                                         ║
echo ║  按 Ctrl+C 停止服务                      ║
echo ╚══════════════════════════════════════════╝
echo.

python -X utf8 "%PROJECT_DIR%multi_turn_app.py"

echo.
echo 服务已停止。
pause
