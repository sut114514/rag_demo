@echo off
chcp 65001 >nul
setlocal
title Terraria RAG

rem Always use this project's directory and virtual environment.
pushd "%~dp0"
if errorlevel 1 (
    echo 无法进入项目目录。
    pause
    exit /b 1
)

if not exist "ragdemo\Scripts\python.exe" (
    echo 未找到 ragdemo 虚拟环境，请先创建环境并安装依赖。
    goto :failed
)

if not exist "app.py" (
    echo 未找到 app.py，请将此脚本放在项目根目录。
    goto :failed
)

if not exist ".env" (
    echo 未找到 .env，请参考 .env.example 配置 DeepSeek API 密钥。
    goto :failed
)

echo 正在使用 ragdemo 虚拟环境启动问答服务……
echo 服务就绪后会自动使用 Microsoft Edge 打开问答页面。
echo 默认地址 http://127.0.0.1:8000 ，端口以 .env 中的 PORT 为准。
echo 请保持此窗口运行，按 Ctrl+C 停止服务。
echo.

"ragdemo\Scripts\python.exe" "app.py" --open-edge
set "ragExitCode=%errorlevel%"
echo.
if not "%ragExitCode%"=="0" echo 启动失败或服务异常退出，请查看上方错误信息。若端口已被占用，请勿重复启动。
pause
popd
exit /b %ragExitCode%

:failed
pause
popd
exit /b 1
