Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd.exe /c chcp 65001 >nul & where uv >nul 2>nul & if errorlevel 1 (python webui.py) else (uv run python webui.py)", 0, False
