import os
import subprocess
import threading
import time
from pathlib import Path

import gradio as gr
import yaml


def load_config() -> dict:
    if not os.path.exists("config.yaml"):
        default_config = """excel:
  path: data/products.xlsx
  sheet: 0
  columns:
    id: A
    name_cn: B
    image_size: C
    language: D
    selling_points: E
    reference_images_are_product: I
  image_columns:
    start: F
    max_columns: 20
    empty_streak: 20
browser:
  chrome_exe: ""
  user_data_dir: browser_profile
gemini:
  preamble_file: preamble.txt
  base_url: https://gemini.google.com
  thinking_mode: true
  reply_timeout: 300
  upload_timeout: 120
  upload_attempts: 3
gemini_api:
  model: gemini-2.5-flash-lite
nvidia_api:
  base_url: https://integrate.api.nvidia.com/v1
  model_choice: kimi
  send_images: true
  models:
    kimi: moonshotai/kimi-k2.5
lovart:
  base_url: https://lgw.lovart.ai
  image_model: auto
  model_selection: prefer
  reasoning_mode: fast
  wait_forever_on_credit_prompt: true
  max_confirmation_rounds: 5
  max_auto_confirm_credits: 10
  wait_timeout: 10800
  poll_interval: 5
  timeout: 600
  upload_attempts: 3
  upload_retry_delay: 2
output_dir: output
"""
        with open("config.yaml", "w", encoding="utf-8") as f:
            f.write(default_config)
            
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def run_process(excel_file, custom_output_dir, prompt_source, lovart_mode, lovart_image_model, gemini_key, nvidia_key, lovart_access, lovart_secret):
    # Save env and configs
    save_env(gemini_key, nvidia_key, lovart_access, lovart_secret)
    
    config = load_config()
    if "lovart" not in config:
        config["lovart"] = {}
    config["lovart"]["image_model"] = lovart_image_model
    config["output_dir"] = custom_output_dir.strip() if custom_output_dir else ""
    save_config(config)

    # Save Excel
    if excel_file is not None:
        target_excel = Path("data/products.xlsx")
        target_excel.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(excel_file, target_excel)

    yield "Starting the automation process...\n"
    
    import sys
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    cmd_args = [
        "--prompt-source", prompt_source, 
        "--lovart", lovart_mode,
        "--lovart-model-selection", "prefer",
        "--lovart-reasoning", "fast"
    ]
    if lovart_image_model and lovart_image_model != "auto":
        cmd_args.extend(["--lovart-image-model", lovart_image_model])
    else:
        cmd_args.extend(["--lovart-image-model", "auto"])
        
    if getattr(sys, 'frozen', False):
        cmd = [sys.executable, "--run-main"] + cmd_args
    else:
        cmd = [sys.executable, "main.py"] + cmd_args
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["UI_MODE"] = "1"
    if custom_output_dir and custom_output_dir.strip():
        env["LOVART_OUTPUT_DIR"] = custom_output_dir.strip()

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )
    
    logs = []
    current_product = "初始化中..."
    current_status = "准备启动环境"
    current_model = lovart_image_model if lovart_image_model else "auto"
    status_color = "#64748b" # slate
    
    products_dict = {}
    
    import html
    def render_board():
        cards_html = ""
        if products_dict:
            cards_html += "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 15px; margin-top: 20px;'>"
            for pid, pdata in products_dict.items():
                img_tag = ""
                if pdata.get("image"):
                    try:
                        import base64
                        import os
                        img_path = pdata["image"]
                        if os.path.exists(img_path):
                            with open(img_path, "rb") as f:
                                encoded = base64.b64encode(f.read()).decode('utf-8')
                            ext = os.path.splitext(img_path)[1].lower().replace('.', '')
                            if ext == 'jpg': ext = 'jpeg'
                            b64_src = f"data:image/{ext};base64,{encoded}"
                            img_tag = f"<img src='{b64_src}' style='width: 60px; height: 60px; object-fit: cover; border-radius: 8px; flex-shrink: 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>"
                    except Exception:
                        pass
                
                link_tag = ""
                if pdata.get("url"):
                    link_tag = f"<a href='{pdata['url']}' target='_blank' style='display: block; margin-top: 10px; background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; text-align: center; padding: 8px; border-radius: 8px; text-decoration: none; font-weight: bold; font-size: 0.9em; box-shadow: 0 2px 10px rgba(59,130,246,0.3);'>🔗 前往 Lovart 查看</a>"
                
                logs_list = pdata.get("logs", [])
                logs_html = ""
                if logs_list:
                    logs_content = "<br>".join(logs_list)
                    # Clean dark background for logs
                    logs_html = f"<div style='margin-top: 10px; background: #0f172a; color: #94a3b8; padding: 12px; border-radius: 6px; border: 1px solid #334155; font-family: \"Cascadia Code\", monospace; font-size: 0.75em; max-height: 150px; overflow-y: auto;'>{logs_content}</div>"

                is_active = pid == current_pid
                animation_css = "animation: pulse-glow 2s infinite;" if is_active else ""
                active_border = "border-color: #8b5cf6;" if is_active else ""
                
                models_attempted = pdata.get("models_attempted", [])
                model_display = " ➔ ".join(models_attempted) if models_attempted else pdata.get("used_model", "")
                
                cards_html += f"""
                <div class='status-card' style='border-left: 4px solid {pdata["color"]}; transition: all 0.3s ease; {animation_css} {active_border}' onmouseover="this.style.transform='translateY(-2px)';" onmouseout="this.style.transform='none';">
                    <div style='display: flex; gap: 15px; align-items: flex-start;'>
                        {img_tag}
                        <div style='flex-grow: 1; min-width: 0;'>
                            <div style='font-size: 0.85em; color: #94a3b8; font-weight: 600; margin-bottom: 6px; display: flex; justify-content: space-between;'>
                                <span>ID: {pid}</span>
                                {f"<span style='color: #d8b4fe; background: rgba(168, 85, 247, 0.2); padding: 2px 8px; border-radius: 9999px; font-size: 0.85em; border: 1px solid rgba(168, 85, 247, 0.3);'>{model_display}</span>" if model_display else ""}
                            </div>
                            <div style='font-size: 1.05em; color: #f8fafc; font-weight: 700; margin-bottom: 10px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;' title="{pdata['name']}">{pdata['name']}</div>
                            <div style='display: inline-block; background: {pdata["color"]}15; color: {pdata["color"]}; padding: 4px 12px; border-radius: 9999px; font-size: 0.8em; font-weight: 600; border: 1px solid {pdata["color"]}30;'>
                                {pdata["status"]}
                            </div>
                        </div>
                    </div>
                    {logs_html}
                    {link_tag}
                </div>
                """
            cards_html += "</div>"
            
        safe_logs = [html.escape(l) for l in logs]
        return f"""
        <div style='margin-top: 15px;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid #334155; padding-bottom: 16px;'>
                <h3 style='margin: 0; display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 1.25em; color: #f8fafc;'>
                    <span>⚡</span> AI 策略指挥中心
                </h3>
                
                <div style='display: flex; gap: 12px; align-items: center;'>
                    <div style='background: {status_color}; color: white; padding: 6px 16px; border-radius: 6px; font-weight: 600; font-size: 0.9em; box-shadow: 0 2px 4px {status_color}40;'>
                        {current_status}
                    </div>
                    <div style='padding: 6px 16px; font-weight: 500; font-size: 0.9em; color: #94a3b8; background: #0f172a; border: 1px solid #334155; border-radius: 6px;'>
                        目标: {current_product}
                    </div>
                    <div style='padding: 6px 16px; font-weight: 600; font-size: 0.9em; color: #d8b4fe; background: rgba(168, 85, 247, 0.2); border-radius: 6px; border: 1px solid rgba(168, 85, 247, 0.3);'>
                        模型: {current_model}
                    </div>
                </div>
            </div>

            <div style="margin-bottom: 24px; background: #0f172a; color: #cbd5e1; padding: 12px 16px; border-radius: 8px; border: 1px solid #334155; font-family: 'Cascadia Code', monospace; font-size: 0.85em;">
                <div style="font-weight: 600; color: #94a3b8; margin-bottom: 8px; border-bottom: 1px solid #334155; padding-bottom: 8px;">
                    ▶ 全局底层运行日志 (Console)
                </div>
                <div style="max-height: 200px; overflow-y: auto; display: flex; flex-direction: column-reverse;">
                    <div>{"<br>".join(safe_logs)}</div>
                </div>
            </div>

            {cards_html}

        </div>
        """
        
    yield render_board()
    
    current_pid = None
    import time, json
    last_yield_time = time.time()
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            clean_line = ansi_escape.sub('', line).strip()
            if not clean_line:
                continue
                
            is_progress = clean_line.startswith("[UI_PROGRESS]")
            is_uiproduct = clean_line.startswith("[UI_PRODUCT]")
            is_uisuccess = clean_line.startswith("[UI_SUCCESS]")
            is_uifail = clean_line.startswith("[UI_FAIL]")
            is_uimodel = clean_line.startswith("[UI_MODEL]")
            
            if not is_progress and not is_uiproduct and not is_uisuccess and not is_uifail and not is_uimodel:
                logs.append(clean_line)
                if len(logs) > 30:
                    logs.pop(0)
                # If log contains a product ID, append to its per-card logs
                if current_pid and current_pid in clean_line and current_pid in products_dict:
                    # Strip timestamps or common prefixes to make it cleaner on the card
                    clean_msg = clean_line.split("]")[-1].strip() if "]" in clean_line else clean_line
                    if "INFO" not in clean_line: # ignore basic INFO lines to save space
                        products_dict[current_pid].setdefault("logs", []).append(f"▶ {clean_msg}")
            
            if is_uimodel:
                try:
                    model_name = clean_line.replace("[UI_MODEL]", "").strip()
                    current_model = model_name
                    if current_pid and current_pid in products_dict:
                        products_dict[current_pid]["used_model"] = model_name
                        models_att = products_dict[current_pid].setdefault("models_attempted", [])
                        if model_name not in models_att:
                            models_att.append(model_name)
                        products_dict[current_pid].setdefault("logs", []).append(f"<span style='color: #d946ef;'>🔄 正在尝试模型: {model_name}</span>")
                except:
                    pass
            elif is_uiproduct:
                try:
                    data = json.loads(clean_line.replace("[UI_PRODUCT]", "").strip())
                    pid = data["id"]
                    if pid not in products_dict:
                        products_dict[pid] = {"name": data["name"], "status": "⏳ 等待处理", "color": "#94a3b8", "logs": []}
                    products_dict[pid]["image"] = data.get("image", "")
                except:
                    pass
            elif is_uisuccess:
                try:
                    data = json.loads(clean_line.replace("[UI_SUCCESS]", "").strip())
                    pid = data["id"]
                    if pid in products_dict:
                        products_dict[pid]["url"] = data.get("url", "")
                        model = data.get("used_model", "")
                        if model and model != "unknown":
                            products_dict[pid]["used_model"] = model
                            products_dict[pid].setdefault("logs", []).append(f"<span style='color: #a855f7;'>✨ 最终使用大模型: <b>{model}</b></span>")
                except:
                    pass
            elif is_uifail:
                try:
                    data = json.loads(clean_line.replace("[UI_FAIL]", "").strip())
                    pid = data["id"]
                    reason = data.get("reason", "未知错误")
                    is_manual = data.get("is_manual", False)
                    if pid in products_dict:
                        status_color = "#f59e0b" if is_manual else "#ef4444"
                        products_dict[pid]["status"] = f"{'⚠️' if is_manual else '❌'} {reason}"
                        products_dict[pid]["color"] = status_color
                        current_status = f"❌ {pid} 失败" if not is_manual else f"⚠️ {pid} 待确认"
                        products_dict[pid].setdefault("logs", []).append(f"<span style='color: {'#fbbf24' if is_manual else '#f87171'}'>[报错] {reason}</span>")
                except:
                    pass
            elif "| size=" in clean_line and "lang=" in clean_line and clean_line.startswith("["):
                parts = clean_line.split("|")
                if len(parts) >= 2:
                    pid_part = parts[0].strip()
                    pid = pid_part.split("]")[1].strip() if "]" in pid_part else pid_part
                    name = parts[1].strip()
                    if pid not in products_dict:
                        products_dict[pid] = {"name": name, "status": "⏳ 等待处理", "color": "#94a3b8", "logs": []}
                        
            if "Gemini requires login" in clean_line:
                current_status = "⚠️ 等待浏览器登录"
                status_color = "#eab308"
            elif clean_line.startswith("Processing ") or clean_line.startswith("processing "):
                pid = clean_line.split()[-1].strip()
                if pid in products_dict:
                    current_pid = pid
                    current_product = f"{pid} - {products_dict[pid]['name']}"
                    current_status = "🔄 提取卖点 & 构思画面"
                    status_color = "#8b5cf6"
                    products_dict[pid]["status"] = current_status
                    products_dict[pid]["color"] = status_color
                    products_dict[pid].setdefault("logs", []).append("▶ 提取卖点 & 构思画面...")
            elif "Gemini done" in clean_line:
                current_status = "✅ 提示词生成完毕"
                status_color = "#06b6d4"
                if current_pid and current_pid in products_dict:
                    products_dict[current_pid]["status"] = current_status
                    products_dict[current_pid]["color"] = status_color
                    products_dict[current_pid].setdefault("logs", []).append("▶ Gemini 提示词生成完毕")
            elif "Lovart API: sent" in clean_line or "Lovart API: Sent" in clean_line:
                current_status = "🎨 提交生成任务"
                status_color = "#f59e0b"
                if current_pid and current_pid in products_dict:
                    products_dict[current_pid]["status"] = current_status
                    products_dict[current_pid]["color"] = status_color
                    products_dict[current_pid].setdefault("logs", []).append("▶ 正在向 Lovart 提交 API 生成请求...")
            elif is_progress:
                parts = clean_line.split("|")
                if len(parts) >= 2:
                    time_str = parts[0].replace("[UI_PROGRESS]", "").strip()
                    step_str = parts[1].strip()
                    current_status = f"🎨 绘制中 ({time_str} - {step_str})"
                    status_color = "#f59e0b"
                    if current_pid and current_pid in products_dict:
                        products_dict[current_pid]["status"] = current_status
                        products_dict[current_pid]["color"] = status_color
                        products_dict[current_pid].setdefault("logs", []).append(f"<span style='color: #60a5fa;'>⏳ 绘制进度: {step_str} | 已用时: {time_str}</span>")
            elif clean_line.startswith("OK") or "completed" in clean_line.lower() or "SUCCESS" in clean_line:
                current_status = "🎉 单个商品全部完成"
                status_color = "#10b981"
                for pid in products_dict:
                    if pid in clean_line:
                        products_dict[pid]["status"] = "🎉 成功生成"
                        products_dict[pid]["color"] = status_color
                        products_dict[pid].setdefault("logs", []).append("<span style='color: #4ade80;'>✅ 任务执行成功</span>")
            elif clean_line.startswith("SKIP"):
                for pid in products_dict:
                    if pid in clean_line:
                        products_dict[pid]["status"] = "⏭️ 已跳过"
                        products_dict[pid]["color"] = "#64748b"
                        products_dict[pid].setdefault("logs", []).append("⏭️ 命中缓存，任务已跳过")
            # Remove the old FAIL parsing since we rely on [UI_FAIL] now
                
            # Throttling rendering to avoid freezing UI
            if time.time() - last_yield_time > 0.5:
                yield render_board()
                last_yield_time = time.time()
                
    yield render_board()
            
    rc = process.poll()
    if rc == 0:
        current_status = "🏁 队列全自动化任务已全部结束！"
        status_color = "#10b981"
        current_product = "全盘扫描完毕"
    else:
        current_status = f"⚠️ 进程异常退出 (代码: {rc})"
        status_color = "#ef4444"
        
    logs.append(f"--- Process finished with exit code {rc} ---")
    yield render_board()


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


CUSTOM_CSS = """
/* Base Theme Variables - Sleek Dark Mode (Clean UI) */
:root {
    --primary-color: #8b5cf6;
    --primary-light: rgba(139, 92, 246, 0.15);
    --bg-color: #0f172a;      /* slate-900 */
    --panel-bg: #1e293b;      /* slate-800 */
    --panel-border: 1px solid #334155; /* slate-700 */
    --text-main: #f8fafc;     /* slate-50 */
    --text-sub: #94a3b8;      /* slate-400 */
    --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    --shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
}

/* Fix Dropdown Clipping */
.gradio-container {
    overflow: visible !important;
}
.wrap {
    overflow: visible !important;
}

/* Typography & Core Elements */
.gradient-text {
    color: var(--text-main);
    font-weight: 800;
    letter-spacing: -0.5px;
}

/* Panels */
.glass-panel {
    background: var(--panel-bg) !important;
    border: var(--panel-border) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.3s ease !important;
    overflow: visible !important;
    color: var(--text-main) !important;
}

/* Glowing Animations for Active Cards */
@keyframes pulse-glow {
    0% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0.4); }
    70% { box-shadow: 0 0 0 6px rgba(139, 92, 246, 0); }
    100% { box-shadow: 0 0 0 0 rgba(139, 92, 246, 0); }
}

/* Primary Button */
.start-btn {
    background: var(--primary-color) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 6px rgba(139, 92, 246, 0.3) !important;
    transition: all 0.2s ease !important;
}

.start-btn:hover {
    background: #7c3aed !important; /* purple-600 */
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4) !important;
}

.start-btn:active {
    transform: translateY(1px) !important;
}

/* Clean Input Components */
.glass-input textarea, .glass-input input, .glass-panel .gr-box, .glass-input .wrap {
    background: #0f172a !important; /* darker than panel */
    border: 1px solid #334155 !important;
    color: var(--text-main) !important;
    border-radius: 6px !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
    transition: all 0.2s ease !important;
}

.glass-input:focus-within textarea, .glass-input:focus-within input, .glass-panel .gr-box:focus-within {
    background: #1e293b !important;
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 2px var(--primary-light) !important;
}

/* Floating Icon Buttons (for Browse/Upload) */
.icon-btn {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-sub) !important;
    font-size: 1.2rem !important;
    padding: 0 10px !important;
    transition: all 0.2s ease !important;
    display: flex;
    align-items: center;
    justify-content: center;
}
.icon-btn:hover {
    color: var(--primary-color) !important;
    transform: scale(1.1);
}

/* Custom Input Label */
.input-label {
    margin-bottom: 4px !important;
}
.input-label p {
    font-size: 0.85em !important;
    font-weight: 600 !important;
    color: var(--text-sub) !important;
    margin: 0 !important;
    padding-left: 4px !important;
}

/* Pill Shaped Dropdowns & Popups */
.pill-dropdown .wrap, .pill-dropdown .gr-box {
    border-radius: 9999px !important;
    padding-left: 10px !important;
    padding-right: 10px !important;
}
.pill-dropdown input, .pill-dropdown .secondary-wrap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    cursor: pointer !important;
    caret-color: transparent !important;
    user-select: none !important;
    outline: none !important;
    text-align: center !important;
}
.pill-dropdown label span {
    font-size: 0.85em !important;
    font-weight: 600 !important;
    color: var(--text-sub) !important;
    margin-bottom: 4px !important;
    margin-left: 0 !important;
    display: block !important;
    text-align: center !important;
}
/* Style the actual dropdown menu (the popup) */
.pill-dropdown .options, .options {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5) !important;
    padding: 8px !important;
    overflow: hidden !important;
}
.options li, .options .item {
    border-radius: 9999px !important;
    margin-bottom: 2px !important;
    transition: all 0.2s !important;
    color: var(--text-main) !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
}
.options li:hover, .options .item:hover {
    background: rgba(139, 92, 246, 0.2) !important;
    color: #d8b4fe !important;
}
.options li.selected, .options .item.selected {
    background: rgba(139, 92, 246, 0.3) !important;
    color: #d8b4fe !important;
    font-weight: bold !important;
}

/* Transparent Columns to replace Groups */
.transparent-col {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    gap: 0 !important;
}

/* Unified Input Row styles */
.input-row {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.2) !important;
    transition: all 0.2s ease !important;
    display: flex;
    align-items: center;
}
.input-row:focus-within {
    background: #1e293b !important;
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 2px var(--primary-light) !important;
}
.input-row input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Status Cards */
.status-card {
    background: #1e293b !important;
    color: var(--text-main) !important;
    border-radius: 8px !important;
    padding: 16px;
    box-shadow: var(--shadow-sm) !important;
    border: 1px solid #334155 !important;
    display: flex;
    flex-direction: column;
    gap: 12px;
}
"""

def manual_save_keys(gemini, nvidia, access, secret):
    save_env(gemini, nvidia, access, secret)
    return "✅ 密钥已成功保存到 .env 文件中"


def pick_directory(current_dir):
    import subprocess
    import sys
    import os
    # Use a subprocess to run tkinter, avoiding Gradio background thread deadlocks
    script = "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.attributes('-topmost', True); root.withdraw(); print(filedialog.askdirectory())"
    try:
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
            
        # Use sys.executable to ensure we use the same Python environment
        result = subprocess.check_output([sys.executable, "-c", script], text=True, **kwargs).strip()
        if result:
            return result
    except Exception:
        pass
    return current_dir


def build_ui():
    config = load_config()
    default_output_dir = config.get("output_dir", str(Path("output").absolute()))
    with gr.Blocks(title="Lovart Image Automation WebUI", css=CUSTOM_CSS, js="() => document.documentElement.classList.add('dark')") as demo:
        gr.HTML("<h1 class='gradient-text' style='text-align: center; margin-top: 20px;'>🎨 Lovart Image Automation Pro</h1>")
        gr.Markdown("<p style='text-align: center; color: gray;'>全自动商品图生成与托管中心</p>")

        with gr.Tabs():
            # ================= TAB 1: 工作台 =================
            with gr.Tab("🚀 工作台 (Workspace)"):
                with gr.Column(elem_classes="glass-panel"):
                    gr.Markdown("### 📂 数据与核心模式")
                    with gr.Row():
                        with gr.Column(elem_classes="transparent-col"):
                            gr.Markdown("**📝 任务表格 (.xlsx)**", elem_classes="input-label")
                            with gr.Row(elem_classes="input-row"):
                                excel_file = gr.Textbox(show_label=False, placeholder="请点击右侧文件夹图标选择文件...", interactive=False, scale=10, container=False)
                                excel_upload = gr.UploadButton("📂", file_types=[".xlsx"], elem_classes="icon-btn", scale=1, min_width=40)
                        
                        with gr.Column(elem_classes="transparent-col"):
                            gr.Markdown("**📁 自定义输出目录**", elem_classes="input-label")
                            with gr.Row(elem_classes="input-row"):
                                custom_output_dir = gr.Textbox(
                                    show_label=False,
                                    placeholder="留空则保存在 output 内",
                                    value=default_output_dir,
                                    scale=10,
                                    container=False
                                )
                                dir_picker_btn = gr.Button("📂", elem_classes="icon-btn", scale=1, min_width=40)
                        
                    excel_upload.upload(lambda f: f.name if hasattr(f, 'name') else str(f), inputs=excel_upload, outputs=excel_file)
                    dir_picker_btn.click(fn=pick_directory, inputs=[custom_output_dir], outputs=[custom_output_dir])
                    
                    with gr.Row():
                        prompt_source = gr.Dropdown(
                            choices=["gemini_api", "gemini_browser", "nvidia"], 
                            value="gemini_browser", 
                            label="提示词引擎",
                            elem_classes=["glass-input", "pill-dropdown"]
                        )
                        lovart_mode = gr.Dropdown(
                            choices=["unlimited", "fast"], 
                            value="unlimited", 
                            label="绘图通道",
                            elem_classes=["glass-input", "pill-dropdown"]
                        )
                        lovart_image_model = gr.Dropdown(
                            choices=["auto", "gpt_image_2", "nano_banana", "seedream_v4_5", "midjourney"], 
                            value="auto", 
                            label="绘图大模型",
                            elem_classes=["glass-input", "pill-dropdown"]
                        )

                start_btn = gr.Button("🚀 开始执行自动化任务 (Start Process)", elem_classes="start-btn", size="lg")
                
                progress_dashboard = gr.HTML(
                    value="<div style='text-align:center; padding: 20px; color: gray;'>任务准备就绪，点击上方按钮开始</div>",
                    elem_classes="glass-panel"
                )

            # ================= TAB 2: 密钥设置 =================
            with gr.Tab("🔑 密钥配置 (Credentials)"):
                with gr.Column(elem_classes="glass-panel"):
                    gr.Markdown("### 🔒 API 密钥管理")
                    gr.Markdown("在下方输入您的密钥，修改完成后请点击**保存密钥**按钮，系统将加密写入 `.env` 文件。")
                    
                    gemini_key = gr.Textbox(label="GEMINI_API_KEY", value=get_env("GEMINI_API_KEY"), type="password")
                    nvidia_key = gr.Textbox(label="NVIDIA_API_KEY (Kimi)", value=get_env("NVIDIA_API_KEY"), type="password")
                    lovart_access = gr.Textbox(label="LOVART_ACCESS_KEY", value=get_env("LOVART_ACCESS_KEY"), type="password")
                    lovart_secret = gr.Textbox(label="LOVART_SECRET_KEY", value=get_env("LOVART_SECRET_KEY"), type="password")
                    
                    save_keys_btn = gr.Button("💾 保存密钥 (Save Keys)", variant="primary")
                    save_status = gr.Markdown("")
                    
                    key_inputs = [gemini_key, nvidia_key, lovart_access, lovart_secret]
                    save_keys_btn.click(fn=manual_save_keys, inputs=key_inputs, outputs=save_status)

            # ================= TAB 3: 系统更新 =================
            with gr.Tab("⚙️ 系统更新 (OTA)"):
                with gr.Column(elem_classes="glass-panel"):
                    from version import VERSION
                    gr.HTML(f"<h3 style='margin-bottom: 0;'>🔄 OTA 自动热更新引擎</h3><p style='color: gray;'>当前客户端版本: <b>v{VERSION}</b></p>")
                    
                    check_update_btn = gr.Button("🔍 检查新版本并全自动覆盖升级", variant="secondary")
                    update_log = gr.Textbox(label="更新状态日志", lines=8, autoscroll=True)

        # 绑定按钮事件
        start_btn.click(
            fn=run_process,
            inputs=[
                excel_file, custom_output_dir, prompt_source, lovart_mode, lovart_image_model,
                gemini_key, nvidia_key, lovart_access, lovart_secret
            ],
            outputs=progress_dashboard
        )
        
        check_update_btn.click(
            fn=ui_check_update,
            inputs=[],
            outputs=update_log
        )
        
    return demo


if __name__ == "__main__":
    demo = build_ui()
    import os
    output_dir = os.path.abspath("output")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, allowed_paths=[output_dir])
