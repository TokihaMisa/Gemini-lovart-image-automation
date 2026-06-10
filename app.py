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
    import webview
    
    # 强制将 WebView2 的用户数据目录设为独立文件夹，避免因权限混淆或幽灵进程锁定导致的 0x8007139F 错误
    os.environ["WEBVIEW2_USER_DATA_FOLDER"] = os.path.abspath("data/webview_cache")

    demo = build_ui()
    # 启动 Gradio 服务器，不阻塞主线程。允许系统自动分配可用端口，避免 7860 端口占用冲突。
    _, local_url, _ = demo.launch(server_name="127.0.0.1", prevent_thread_lock=True)

    theme_url = local_url.rstrip('/') + '/?__theme=dark'
    # 启动原生窗口并加载该 URL
    webview.create_window('Lovart自动化助手', theme_url, width=1024, height=768)
    webview.start()
    
    # 强制结束所有残留的 Gradio 后台线程，防止产生幽灵进程
    os._exit(0)
