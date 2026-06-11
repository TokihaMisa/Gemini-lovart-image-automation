@echo off
chcp 65001 >nul
echo Start building...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
uv run pyinstaller --noconfirm --onedir --windowed --name "Lovart_Auto" --collect-all gradio --collect-all gradio_client --collect-data playwright --collect-data safehttpx --collect-data groovy --hidden-import uvicorn.loops.auto --hidden-import uvicorn.protocols.http.auto --hidden-import uvicorn.protocols.websockets.auto --hidden-import uvicorn.lifespan.on --hidden-import websockets.legacy.server --collect-data uvicorn app.py

echo.
echo ----------------------------------------
echo 正在复制必要的外部配置文件到打包目录...
echo ----------------------------------------
copy /Y preamble.txt dist\Lovart_Auto\
copy /Y config.example.yaml dist\Lovart_Auto\
copy /Y .env.example dist\Lovart_Auto\
echo.
echo 打包已全部完成！你可以将 dist\Lovart_Auto 文件夹发送给同事使用了。
pause
