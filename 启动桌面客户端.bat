@echo off
chcp 65001 >nul
echo ==============================================
echo 正在为您启动 Lovart 原生桌面客户端...
echo ==============================================
echo (请勿关闭此黑色窗口，客户端界面即将弹出)
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    uv run python app.py
) else (
    python app.py
)
pause
