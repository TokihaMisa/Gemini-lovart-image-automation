Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd.exe /c chcp 65001 >nul & uv run python app.py", 0, False
