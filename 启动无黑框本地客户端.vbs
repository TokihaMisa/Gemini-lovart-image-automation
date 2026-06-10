Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd.exe /c chcp 65001 >nul & ""C:\Users\Soul-\AppData\Local\Python\pythoncore-3.14-64\python.exe"" app.py", 0, False
