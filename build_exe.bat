@echo off
chcp 65001 >nul
echo Start building...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
uv run pyinstaller --noconfirm --onedir --windowed --name "Lovart自动化助手" --collect-data gradio --collect-data gradio_client --collect-data groovy --collect-data playwright --collect-data safehttpx webui.py
pause
