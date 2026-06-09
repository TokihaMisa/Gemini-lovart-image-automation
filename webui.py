import os
import subprocess
import threading
import time
from pathlib import Path

import gradio as gr
import yaml


def load_config() -> dict:
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_config(config_data: dict):
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, allow_unicode=True, sort_keys=False)


def save_env(gemini_key: str, nvidia_key: str, lovart_access: str, lovart_secret: str):
    lines = []
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f.readlines():
                if any(line.startswith(k) for k in ["GEMINI_API_KEY=", "NVIDIA_API_KEY=", "LOVART_ACCESS_KEY=", "LOVART_SECRET_KEY="]):
                    continue
                lines.append(line)
    
    lines.append(f"GEMINI_API_KEY={gemini_key}\n")
    lines.append(f"NVIDIA_API_KEY={nvidia_key}\n")
    lines.append(f"LOVART_ACCESS_KEY={lovart_access}\n")
    lines.append(f"LOVART_SECRET_KEY={lovart_secret}\n")
    
    with open(".env", "w", encoding="utf-8") as f:
        f.writelines(lines)


def get_env(key: str) -> str:
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip()
    return ""


def run_process(excel_file, prompt_source, lovart_mode, lovart_image_model, gemini_key, nvidia_key, lovart_access, lovart_secret):
    # Save env and configs
    save_env(gemini_key, nvidia_key, lovart_access, lovart_secret)
    
    config = load_config()
    if "lovart" not in config:
        config["lovart"] = {}
    config["lovart"]["image_model"] = lovart_image_model
    save_config(config)

    # Save Excel
    if excel_file is not None:
        target_excel = Path("data/products.xlsx")
        target_excel.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(excel_file, target_excel)

    yield "Starting the automation process...\n"
    
    import sys
    if getattr(sys, 'frozen', False):
        # We are running as a PyInstaller executable
        cmd = [sys.executable, "--run-main", "--prompt-source", prompt_source, "--lovart", lovart_mode]
    else:
        # We are running from source
        cmd = [sys.executable, "main.py", "--prompt-source", prompt_source, "--lovart", lovart_mode]
    
    # Run the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )
    
    output = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            output.append(line)
            # keep only the last 100 lines to avoid UI freezing
            if len(output) > 100:
                output = output[-100:]
            yield "".join(output)
            
    rc = process.poll()
    output.append(f"\n[Process finished with exit code {rc}]\n")
    yield "".join(output)


def ui_check_update():
    from updater import check_for_updates, download_and_install_update
    import queue
    import threading
    
    yield "正在检查更新，请稍候..."
    has_update, new_version, url, changelog = check_for_updates()
    if not has_update:
        yield "当前已是最新版本，无需更新。"
        return
        
    yield f"发现新版本: v{new_version}\n更新内容: {changelog}\n\n准备下载..."
    
    q = queue.Queue()
    threading.Thread(target=download_and_install_update, args=(url, q), daemon=True).start()
    
    output = []
    while True:
        msg = q.get()
        if msg is None:
            break
        output.append(msg)
        if len(output) > 20:
            output = output[-20:]
        yield "\n".join(output)


def build_ui():
    with gr.Blocks(title="Lovart Image Automation WebUI") as demo:
        gr.Markdown("# 🎨 Lovart Product Image Automation")
        gr.Markdown("Upload your Excel file, configure the models, and start the batch generation.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. File & Models")
                excel_file = gr.File(label="Upload Excel File (products.xlsx)", file_types=[".xlsx"])
                
                prompt_source = gr.Dropdown(
                    choices=["gemini_api", "gemini_browser", "nvidia"], 
                    value="gemini_browser", 
                    label="Prompt Generation Source"
                )
                
                lovart_mode = gr.Dropdown(
                    choices=["unlimited", "fast"], 
                    value="unlimited", 
                    label="Lovart Mode (Unlimited=Free/Queue, Fast=Credits/No Queue)"
                )
                
                lovart_image_model = gr.Dropdown(
                    choices=["auto", "gpt_image_2", "nano_banana", "seedream_v4_5", "midjourney"], 
                    value="auto", 
                    label="Lovart Image Model",
                    multiselect=False
                )

            with gr.Column(scale=1):
                gr.Markdown("### 2. API Keys (Saved to .env)")
                gemini_key = gr.Textbox(label="GEMINI_API_KEY", value=get_env("GEMINI_API_KEY"), type="password")
                nvidia_key = gr.Textbox(label="NVIDIA_API_KEY (Kimi)", value=get_env("NVIDIA_API_KEY"), type="password")
                lovart_access = gr.Textbox(label="LOVART_ACCESS_KEY", value=get_env("LOVART_ACCESS_KEY"), type="password")
                lovart_secret = gr.Textbox(label="LOVART_SECRET_KEY", value=get_env("LOVART_SECRET_KEY"), type="password")

        with gr.Row():
            with gr.Column():
                from version import VERSION
                gr.Markdown(f"### ⚙️ 系统设置与更新 (当前版本: v{VERSION})")
                check_update_btn = gr.Button("🔄 检查更新并升级", variant="secondary")
                update_log = gr.Textbox(label="更新状态", lines=4, autoscroll=True)

        with gr.Row():
            start_btn = gr.Button("🚀 Start Process", variant="primary")
            
        with gr.Row():
            console_output = gr.Textbox(
                label="Terminal Output", 
                lines=20, 
                max_lines=30,
                autoscroll=True
            )

        start_btn.click(
            fn=run_process,
            inputs=[
                excel_file, prompt_source, lovart_mode, lovart_image_model,
                gemini_key, nvidia_key, lovart_access, lovart_secret
            ],
            outputs=console_output
        )
        
        check_update_btn.click(
            fn=ui_check_update,
            inputs=[],
            outputs=update_log
        )
        
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
