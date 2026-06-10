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

    if "--run-tkinter-splash" in sys.argv:
        import tkinter as tk
        from tkinter import ttk
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes('-topmost', True)
        w, h = 320, 110
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f'{w}x{h}+{int(sw/2-w/2)}+{int(sh/2-h/2)}')
        f = tk.Frame(root, highlightbackground='#6366f1', highlightthickness=2, bg='white')
        f.pack(fill='both', expand=True)
        tk.Label(f, text='🚀 Lovart AI 引擎启动中...', font=('Microsoft YaHei', 12, 'bold'), bg='white', fg='#333333').pack(pady=(20, 5))
        p = ttk.Progressbar(f, orient='horizontal', length=260, mode='indeterminate')
        p.pack(pady=10)
        p.start(15)
        root.mainloop()
        sys.exit(0)

    # 启动进度条动画 (Splash Screen)
    import subprocess
    import sys
    import os
    splash_proc = None
    try:
        if getattr(sys, 'frozen', False):
            cmd = [sys.executable, "--run-tkinter-splash"]
        else:
            cmd = [sys.executable, __file__, "--run-tkinter-splash"]
        
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        splash_proc = subprocess.Popen(cmd, creationflags=creation_flags)
    except Exception:
        pass

    try:
        from webui import build_ui
        import time
        import webbrowser

        def open_as_native_app(url):
            edge_paths = [
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
            ]
            for path in edge_paths:
                if os.path.exists(path):
                    subprocess.Popen([path, f"--app={url}"])
                    return
            
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe")
            ]
            for path in chrome_paths:
                if os.path.exists(path):
                    subprocess.Popen([path, f"--app={url}"])
                    return
                    
            webbrowser.open(url)

        demo = build_ui()
        _, local_url, _ = demo.launch(server_name="127.0.0.1", prevent_thread_lock=True)

        theme_url = local_url.rstrip('/') + '/?__theme=dark'
        
        # 启动完成后立即关闭动画
        if splash_proc:
            try:
                splash_proc.terminate()
                splash_proc = None
            except Exception:
                pass
                
        print(f"\n✅ 服务已启动！内部运行地址: {theme_url}")
        print("正在通过 Edge/Chrome 内核为您唤醒原生客户端窗口...")
        open_as_native_app(theme_url)
        
        # 保持主进程存活
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    except Exception as e:
        import traceback
        with open("crash.log", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("致命错误", f"程序启动失败，请将程序目录下的 crash.log 发给开发者。\n\n错误信息: {e}")
    finally:
        # 确保哪怕发生任何错误，动画窗口一定会被关闭！
        if splash_proc:
            try:
                splash_proc.terminate()
            except Exception:
                pass
        
        os._exit(0)
