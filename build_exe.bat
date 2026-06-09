@echo off
chcp 65001 >nul
echo Start building...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
uv run pyinstaller --noconfirm --onedir --windowed --name LovartAuto --collect-data gradio --collect-data playwright app.py
pause
