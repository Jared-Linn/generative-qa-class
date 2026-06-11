@echo off
chcp 65001 >nul
title 多轮对话系统 - Qwen3.5-0.8B
setlocal enabledelayedexpansion

:: ============================================
:: 多轮对话系统 — 一键部署启动脚本
:: ============================================

set PROJECT_DIR=%~dp0
set MODEL_DIR=%PROJECT_DIR%Qwen3.5-0.8B
set LORA_DIR=%PROJECT_DIR%outputs_multi_turn\lora_adapter
set DATA_FILE=%PROJECT_DIR%data\multi_turn_qa.json

echo ╔══════════════════════════════════════════╗
echo ║     多轮对话系统 — 一键启动             ║
echo ║     基于 Qwen3.5-0.8B + LoRA            ║
echo ╚══════════════════════════════════════════╝
echo.

:: ========== 1. 检查 Python ==========
echo [1/5] 检查 Python 环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ❌ Python 未安装或不在 PATH 中，请安装 Python 3.10+
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
echo   ✅ Python %PY_VER%

:: ========== 2. 检查/安装依赖 ==========
echo [2/5] 检查 Python 依赖...
pip show transformers >nul 2>&1
if %errorlevel% neq 0 (
    echo   ⏳ 安装依赖中，请稍候...
    pip install torch transformers datasets peft accelerate fastapi uvicorn jinja2 -q
    if !errorlevel! neq 0 (
        echo   ❌ 依赖安装失败，请手动执行: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo   ✅ 依赖安装完成
) else (
    echo   ✅ 依赖已就绪
)

:: ========== 3. 检查模型文件 ==========
echo [3/5] 检查模型文件...
if not exist "%MODEL_DIR%\config.json" (
    echo   ❌ 模型文件缺失，请确认 Qwen3.5-0.8B 目录完整
    echo      预期路径: %MODEL_DIR%
    echo.
    echo   如需下载：hf download Qwen/Qwen3.5-0.8B --local-dir "%MODEL_DIR%"
    pause
    exit /b 1
)
echo   ✅ 模型已就绪 (Qwen3.5-0.8B)

:: ========== 4. 检查 LoRA 适配器 ==========
echo [4/5] 检查 LoRA 适配器...
if not exist "%LORA_DIR%\adapter_config.json" (
    echo   ⚠️ 未检测到 LoRA 适配器，将使用原始模型推理
    echo   (仅当您已训练过才会有效果，训练命令：python multi_turn_finetune_qwen_cpu.py)
    echo.
    set USE_RAW_MODEL=1
) else (
    echo   ✅ LoRA 适配器已就绪
    set USE_RAW_MODEL=0
)

:: ========== 5. 启动服务 ==========
echo [5/5] 启动 Web 服务...
echo.
echo ╔══════════════════════════════════════════╗
echo ║  服务启动中，请稍候（约 30-60秒）...    ║
echo ║                                         ║
echo ║  聊天界面: http://localhost:8000         ║
echo ║  API 文档: http://localhost:8000/docs    ║
echo ║                                         ║
echo ║  Ctrl+C 停止服务                         ║
echo ╚══════════════════════════════════════════╝
echo.

python -X utf8 "%PROJECT_DIR%multi_turn_app.py"

echo.
echo 服务已停止。
pause
