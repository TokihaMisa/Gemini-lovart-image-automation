import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from version import VERSION, UPDATE_INFO_URL


def check_for_updates():
    """Check for updates and return (has_update, version, url, changelog)"""
    try:
        req = urllib.request.Request(UPDATE_INFO_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        latest_version = data.get("version")
        url = data.get("url")
        changelog = data.get("changelog", "")
        
        if latest_version and latest_version != VERSION:
            return True, latest_version, url, changelog
            
    except Exception as e:
        print(f"Failed to check for updates: {e}")
        
    return False, None, None, None


def download_and_install_update(url: str, output_queue=None):
    def log(msg):
        if output_queue:
            output_queue.put(msg)
        else:
            print(msg)
            
    try:
        log("开始下载更新包...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        temp_zip_path = Path("update.zip")
        
        with urllib.request.urlopen(req) as response, open(temp_zip_path, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', -1))
            downloaded = 0
            block_size = 1024 * 1024 # 1MB
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total_size > 0:
                    percent = int((downloaded / total_size) * 100)
                    log(f"下载中... {percent}%")
                    
        log("下载完成，正在解压...")
        
        temp_update_dir = Path("temp_update")
        if temp_update_dir.exists():
            try:
                shutil.rmtree(temp_update_dir)
            except:
                pass
        temp_update_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            # 如果 zip 内部套了一层文件夹，我们最好把内容提出来
            # 但简单起见，我们假定更新包是直接把根目录文件打包的
            zip_ref.extractall(temp_update_dir)
            
        log("解压完成。准备重启并应用更新...")
        
        # 创建一个独立的后台更新脚本
        exe_path = sys.executable
        if not getattr(sys, 'frozen', False):
            # 如果没有被打包成 exe，则使用 app.py 启动
            exe_path = f"uv run python app.py"
        else:
            exe_path = f'"{exe_path}"'
            
        bat_content = f"""@echo off
chcp 65001 >nul
echo 正在应用更新，请勿关闭窗口...
timeout /t 3 /nobreak >nul
xcopy /E /Y /C "{temp_update_dir.absolute()}\\*" "{Path.cwd().absolute()}"
rmdir /s /q "{temp_update_dir.absolute()}"
del /f /q "{temp_zip_path.absolute()}"
echo 更新完成！正在重新启动软件...
start "" {exe_path}
del "%~f0"
"""
        bat_path = Path("install_update.bat")
        bat_path.write_text(bat_content, encoding="utf-8")
        
        # 启动更新脚本并脱离当前进程
        CREATE_NEW_CONSOLE = 0x00000010
        subprocess.Popen(["cmd.exe", "/c", str(bat_path.absolute())], creationflags=CREATE_NEW_CONSOLE)
        
        log("更新脚本已启动，当前程序即将退出...")
        
        # 强制退出，解除文件占用
        os._exit(0)
        
    except Exception as e:
        log(f"更新失败: {e}")
        if output_queue:
            output_queue.put(None)
