import os
import sys

def install_playwright():
    print("正在检查并下载必要的浏览器内核 (首次启动可能需要几分钟)...")
    try:
        from playwright.__main__ import main as playwright_main
        old_argv = sys.argv
        sys.argv = ["playwright", "install", "chromium"]
        playwright_main()
        sys.argv = old_argv
    except SystemExit:
        pass
    except Exception as e:
        print(f"浏览器内核下载失败: {e}")

if __name__ == "__main__":
    if "--run-main" in sys.argv:
        sys.argv.remove("--run-main")
        from main import main as run_main
        run_main()
        sys.exit(0)

    install_playwright()

    from webui import build_ui
    import webview

    demo = build_ui()
    # 启动 Gradio 服务器，不阻塞主线程
    demo.launch(server_name="127.0.0.1", server_port=7860, prevent_thread_lock=True)

    # 启动原生窗口并加载该 URL
    webview.create_window('Lovart自动化助手', 'http://127.0.0.1:7860', width=1024, height=768)
    webview.start()
    
    # 强制结束所有残留的 Gradio 后台线程，防止产生幽灵进程
    os._exit(0)
