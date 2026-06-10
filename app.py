import os
import sys
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if "--run-main" in sys.argv:
        sys.argv.remove("--run-main")
        from main import main as run_main
        run_main()
        sys.exit(0)

    from webui import build_ui
    import webview

    demo = build_ui()
    # 启动 Gradio 服务器，不阻塞主线程
    demo.launch(server_name="127.0.0.1", server_port=7860, prevent_thread_lock=True)

    # 启动原生窗口并加载该 URL
    webview.create_window('Lovart自动化助手', 'http://127.0.0.1:7860/?__theme=dark', width=1024, height=768)
    webview.start()
    
    # 强制结束所有残留的 Gradio 后台线程，防止产生幽灵进程
    os._exit(0)
