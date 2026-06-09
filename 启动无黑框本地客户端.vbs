Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd.exe /c chcp 65001 >nul & where uv >nul 2>nul & if errorlevel 1 (python app.py) else (uv run --no-sync python app.py)", 0, False
