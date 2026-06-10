import os
import sys
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if "--run-main" in sys.argv:
        sys.argv.remove("--run-main")
        try:
            from main import main as run_main
            run_main()
        except Exception as e:
            import traceback
            print("\n" + "="*50)
            print("❌ 核心程序发生致命错误 (Fatal Error)")
            print("可能原因: 网络连接被重置 (代理/VPN冲突) 或 API 配置错误。")
            print("="*50)
            traceback.print_exc()
        sys.exit(0)

    if "--run-tkinter-dir" in sys.argv:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        res = filedialog.askdirectory()
        if res:
            print(res)
        sys.exit(0)

    from webui import build_ui
    import subprocess
    import time
    import webbrowser

    def open_as_native_app(url):
        # 优先尝试使用 Edge 的 App 模式 (Windows 10/11 必定自带)
        edge_paths = [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
        ]
        for path in edge_paths:
            if os.path.exists(path):
                subprocess.Popen([path, f"--app={url}"])
                return
        
        # 其次尝试 Chrome 的 App 模式
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe")
        ]
        for path in chrome_paths:
            if os.path.exists(path):
                subprocess.Popen([path, f"--app={url}"])
                return
                
        # 如果都没找到，兜底使用普通浏览器打开
        webbrowser.open(url)

    demo = build_ui()
    # 启动 Gradio 服务器，不阻塞主线程。允许系统自动分配可用端口，避免 7860 端口占用冲突。
    _, local_url, _ = demo.launch(server_name="127.0.0.1", prevent_thread_lock=True)

    theme_url = local_url.rstrip('/') + '/?__theme=dark'
    
    print(f"\n✅ 服务已启动！内部运行地址: {theme_url}")
    print("正在通过 Edge/Chrome 内核为您唤醒原生客户端窗口...")
    open_as_native_app(theme_url)
    
    # 保持主进程存活，直到用户手动关闭黑框
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    # 强制结束所有残留的 Gradio 后台线程，防止产生幽灵进程
    os._exit(0)
