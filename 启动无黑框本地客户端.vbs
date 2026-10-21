Option Explicit

Dim WshShell, FileSystem, ProjectDir, PythonPath, AppPath
Set WshShell = CreateObject("WScript.Shell")
Set FileSystem = CreateObject("Scripting.FileSystemObject")
ProjectDir = FileSystem.GetParentFolderName(WScript.ScriptFullName)
WshShell.CurrentDirectory = ProjectDir
WshShell.Environment("Process")("PYTHONUTF8") = "1"
WshShell.Environment("Process")("PYTHONIOENCODING") = "utf-8"

PythonPath = FileSystem.BuildPath(ProjectDir, ".venv\Scripts\pythonw.exe")
If Not FileSystem.FileExists(PythonPath) Then
    PythonPath = FileSystem.BuildPath(ProjectDir, ".venv\Scripts\python.exe")
End If
If Not FileSystem.FileExists(PythonPath) Then
    PythonPath = WshShell.ExpandEnvironmentStrings("%LOCALAPPDATA%\Python\pythoncore-3.14-64\pythonw.exe")
End If
If Not FileSystem.FileExists(PythonPath) Then
    MsgBox "Python was not found. Set up the project's .venv first.", 16, "Lovart Auto"
    WScript.Quit 1
End If

AppPath = FileSystem.BuildPath(ProjectDir, "app.py")
WshShell.Run """" & PythonPath & """ """ & AppPath & """", 0, False
