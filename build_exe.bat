@echo off
chcp 65001 >nul
echo Start building...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
uv run pyinstaller --noconfirm --onedir --windowed --name "Lovart自动化助手" --collect-all gradio --collect-all gradio_client --collect-data playwright --collect-data safehttpx --collect-data groovy --hidden-import uvicorn.loops.auto --hidden-import uvicorn.protocols.http.auto --hidden-import uvicorn.protocols.websockets.auto --hidden-import uvicorn.lifespan.on --hidden-import websockets.legacy.server --collect-data uvicorn app.py
pause
