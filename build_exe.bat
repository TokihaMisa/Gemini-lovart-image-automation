@echo off
chcp 65001 >nul
echo Start building...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
uv run --no-sync pyinstaller --noconfirm --onedir --windowed --name "Lovart_Auto" --add-data "preamble.txt;." --add-data "config.example.yaml;." --add-data ".env.example;." --collect-all gradio --collect-all gradio_client --collect-data playwright --collect-data safehttpx --collect-data groovy --hidden-import uvicorn.loops.auto --hidden-import uvicorn.protocols.http.auto --hidden-import uvicorn.protocols.websockets.auto --hidden-import uvicorn.lifespan.on --hidden-import websockets.legacy.server --collect-data uvicorn app.py

echo.
echo ----------------------------------------
echo 打包已全部完成！内置文件已打包到 _internal 目录中。
echo 你可以将 dist\Lovart_Auto 文件夹发送给同事使用了。
echo ----------------------------------------
pause
