import os
import subprocess
import threading
import time
import atexit
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import gradio as gr
import yaml

from model_provider import (
    DiscoveredModel,
    ModelProviderError,
    discover_models,
    model_choice_labels,
    test_selected_model,
)
from prompt_settings import (
    DEFAULT_PROMPT_SETTINGS,
    effective_rules_preview,
    get_prompt_settings,
    merge_prompt_settings,
    normalize_prompt_settings,
)


PROMPT_FORM_FIELDS = (
    "detail_page_count",
    "design_style",
    "required_sections",
    "image_quality",
    "logo_policy",
    "copy_style",
    "copy_detail_level",
    "product_fidelity",
    "white_background_requirements",
    "scene_requirements",
    "allow_questions",
    "default_language",
    "missing_image_size_policy",
    "extra_requirements",
)

active_processes = []

def cleanup_processes():
    for p in active_processes:
        try:
            if p.poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)], capture_output=True)
                else:
                    p.kill()
        except Exception:
            pass

atexit.register(cleanup_processes)


def load_config(path: str | Path = "config.yaml") -> dict:
    target = Path(path)
    if not target.exists():
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
  poll_interval: 10
  timeout: 600
  upload_attempts: 3
  upload_retry_delay: 2
output_dir: output
"""
        save_config(yaml.safe_load(default_config) or {}, target)

    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


def save_config(config_data: dict, path: str | Path = "config.yaml"):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp")
    try:
        text = yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False)
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def prompt_settings_to_form(config) -> tuple:
    settings = get_prompt_settings(config)
    return tuple(deepcopy(settings[field]) for field in PROMPT_FORM_FIELDS)


def form_to_prompt_settings(
    detail_page_count,
    design_style,
    required_sections,
    image_quality,
    logo_policy,
    copy_style,
    copy_detail_level,
    product_fidelity,
    white_background_requirements,
    scene_requirements,
    allow_questions,
    default_language,
    missing_image_size_policy,
    extra_requirements,
):
    return normalize_prompt_settings({
        "detail_page_count": detail_page_count,
        "design_style": design_style,
        "required_sections": required_sections,
        "image_quality": image_quality,
        "logo_policy": logo_policy,
        "copy_style": copy_style,
        "copy_detail_level": copy_detail_level,
        "product_fidelity": product_fidelity,
        "white_background_requirements": white_background_requirements,
        "scene_requirements": scene_requirements,
        "allow_questions": allow_questions,
        "default_language": default_language,
        "missing_image_size_policy": missing_image_size_policy,
        "extra_requirements": extra_requirements,
    })


def save_prompt_settings_from_form(*values, config_path="config.yaml") -> tuple[str, str]:
    target = Path(config_path)
    current = yaml.safe_load(target.read_text(encoding="utf-8")) or {} if target.exists() else {}
    try:
        settings = form_to_prompt_settings(*values)
        updated = merge_prompt_settings(current, settings)
    except (TypeError, ValueError) as exc:
        existing_settings = get_prompt_settings(current)
        return f"❌ {exc}", effective_rules_preview(existing_settings)

    save_config(updated, target)
    return "✅ 提示词设置已保存", effective_rules_preview(settings)


def reset_prompt_settings_form() -> tuple:
    defaults = normalize_prompt_settings(DEFAULT_PROMPT_SETTINGS)
    return (*tuple(deepcopy(defaults[field]) for field in PROMPT_FORM_FIELDS), effective_rules_preview(defaults))


def refresh_provider_models(provider, api_key, base_url, current_model):
    try:
        models = discover_models(provider, api_key, base_url)
    except ModelProviderError as exc:
        choices = [(current_model, current_model)] if current_model else []
        return f"❌ {exc.user_message}", choices, current_model, []

    choices = model_choice_labels(models)
    model_ids = [model.model_id for model in models]
    selected = current_model if current_model in model_ids else (model_ids[0] if model_ids else current_model)
    if selected and selected not in model_ids:
        choices.append((selected, selected))
    return f"✅ 成功获取 {len(models)} 个可用模型。", choices, selected, [asdict(model) for model in models]


def test_provider_model(provider, api_key, base_url, model_id):
    try:
        result = test_selected_model(provider, api_key, base_url, model_id)
    except ModelProviderError as exc:
        return f"❌ {exc.user_message} 测试可能产生极少量 API 用量。"
    status_icon = "✅" if result.ok else "❌"
    return f"{status_icon} {result.message}（{result.latency_ms} ms）。测试可能产生极少量 API 用量。"


test_provider_model.__test__ = False


def _configured_provider_model(config, prompt_source):
    config_section = "gemini_api" if prompt_source == "gemini_api" else "nvidia_api"
    provider_config = config.get(config_section, {}) or {}
    direct_model = provider_config.get("model", "")
    if direct_model:
        return direct_model
    if prompt_source == "nvidia":
        legacy_models = provider_config.get("models", {})
        legacy_choice = provider_config.get("model_choice", "")
        if isinstance(legacy_models, dict):
            return legacy_models.get(legacy_choice, "")
    return ""


def resolve_model_dropdown(prompt_source, gemini_catalog, nvidia_catalog, config):
    if prompt_source == "gemini_browser":
        page_managed = "由浏览器页面选择"
        return [(page_managed, page_managed)], page_managed, False

    if prompt_source == "gemini_api":
        catalog = gemini_catalog
    elif prompt_source == "nvidia":
        catalog = nvidia_catalog
    else:
        return [], "", False

    models = [DiscoveredModel(**item) for item in catalog]
    choices = model_choice_labels(models)
    model_ids = [model.model_id for model in models]
    selected = _configured_provider_model(config, prompt_source)
    if not selected and model_ids:
        selected = model_ids[0]
    if selected and selected not in model_ids:
        choices.append((selected, selected))
    return choices, selected, True


def retain_workspace_model_selection(prompt_source, workspace_model, gemini_model, nvidia_model):
    if prompt_source == "gemini_api" and workspace_model:
        return workspace_model, nvidia_model
    if prompt_source == "nvidia" and workspace_model:
        return gemini_model, workspace_model
    return gemini_model, nvidia_model


def persist_selected_model(config, prompt_source, model_id):
    updated = deepcopy(config)
    config_section = {
        "gemini_api": "gemini_api",
        "nvidia": "nvidia_api",
    }.get(prompt_source)
    if config_section:
        updated.setdefault(config_section, {})["model"] = model_id
    return updated


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


def run_process(excel_file, custom_output_dir, prompt_source, prompt_model, lovart_mode, lovart_image_model, gemini_key, nvidia_key, lovart_access, lovart_secret):
    # Save env and configs
    save_env(gemini_key, nvidia_key, lovart_access, lovart_secret)
    
    config = persist_selected_model(load_config(), prompt_source, prompt_model)
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
    active_processes.append(process)
    
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
    import time, json, threading, queue
    
    last_yield_time = time.time()
    
    q = queue.Queue()
    def _read_output(out, q):
        try:
            for line in iter(out.readline, ''):
                q.put(line)
        except Exception:
            pass
        finally:
            out.close()
            
    t = threading.Thread(target=_read_output, args=(process.stdout, q), daemon=True)
    t.start()
    
    while True:
        try:
            line = q.get(timeout=1.0)
        except queue.Empty:
            if process.poll() is not None and not t.is_alive():
                break
            # Heartbeat to prevent Gradio/WebSocket from dropping the connection
            if time.time() - last_yield_time >= 1.0:
                yield render_board()
                last_yield_time = time.time()
            continue

        if not line:
            if process.poll() is not None:
                break
            continue

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
    try:
        kwargs = {}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
            
        if getattr(sys, 'frozen', False):
            cmd = [sys.executable, "--run-tkinter-dir"]
        else:
            script = "import tkinter as tk; from tkinter import filedialog; root = tk.Tk(); root.attributes('-topmost', True); root.withdraw(); print(filedialog.askdirectory())"
            cmd = [sys.executable, "-c", script]

        result = subprocess.check_output(cmd, text=True, **kwargs).strip()
        if result:
            return result
    except Exception:
        pass
    return current_dir


def build_ui():
    config = load_config()
    default_output_dir = config.get("output_dir", str(Path("output").absolute()))
    prompt_form_values = prompt_settings_to_form(config)
    prompt_preview_value = effective_rules_preview(get_prompt_settings(config))
    required_section_choices = list(DEFAULT_PROMPT_SETTINGS["required_sections"])
    for section in prompt_form_values[2]:
        if section not in required_section_choices:
            required_section_choices.append(section)
    gemini_config = config.get("gemini_api", {})
    nvidia_config = config.get("nvidia_api", {})
    gemini_saved_model = _configured_provider_model(config, "gemini_api")
    nvidia_saved_model = _configured_provider_model(config, "nvidia")
    gemini_base_url_value = gemini_config.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
    nvidia_base_url_value = nvidia_config.get("base_url", "https://integrate.api.nvidia.com/v1")

    def refresh_provider_controls(provider, api_key, base_url, current_model, prompt_source_value):
        status, choices, selected, catalog = refresh_provider_models(
            provider, api_key, base_url, current_model
        )
        provider_update = gr.update(choices=choices, value=selected)
        active_source = "gemini_api" if provider == "gemini" else "nvidia"
        workspace_update = (
            gr.update(choices=choices, value=selected, interactive=True)
            if prompt_source_value == active_source
            else gr.skip()
        )
        return status, provider_update, workspace_update, catalog

    def resolve_workspace_model(prompt_source_value, gemini_catalog, nvidia_catalog, gemini_model, nvidia_model):
        live_config = deepcopy(config)
        live_config.setdefault("gemini_api", {})["model"] = gemini_model or ""
        live_config.setdefault("nvidia_api", {})["model"] = nvidia_model or ""
        choices, selected, interactive = resolve_model_dropdown(
            prompt_source_value, gemini_catalog, nvidia_catalog, live_config
        )
        return gr.update(choices=choices, value=selected, interactive=interactive)

    def sync_workspace_model(prompt_source_value, provider_source, model_id):
        return model_id if prompt_source_value == provider_source else gr.skip()

    def retain_workspace_selection(prompt_source_value, workspace_model, gemini_model, nvidia_model):
        updated_gemini, updated_nvidia = retain_workspace_model_selection(
            prompt_source_value, workspace_model, gemini_model, nvidia_model
        )
        gemini_update = gr.update(value=updated_gemini) if updated_gemini != gemini_model else gr.skip()
        nvidia_update = gr.update(value=updated_nvidia) if updated_nvidia != nvidia_model else gr.skip()
        return gemini_update, nvidia_update

    with gr.Blocks(title="Lovart Image Automation WebUI", css=CUSTOM_CSS, js="() => document.documentElement.classList.add('dark')") as demo:
        gemini_catalog_state = gr.State([])
        nvidia_catalog_state = gr.State([])
        with gr.Row():
            gr.HTML("<h1 class='gradient-text' style='text-align: center; margin-top: 20px; flex-grow: 1;'>🎨 Lovart Image Automation Pro</h1>")
            shutdown_btn = gr.Button("🛑 完全退出并关闭服务", variant="stop", scale=0, min_width=180, elem_classes="action-btn")

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
                        prompt_model = gr.Dropdown(
                            choices=[("由浏览器页面选择", "由浏览器页面选择")],
                            value="由浏览器页面选择",
                            label="提示词模型",
                            interactive=False,
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

            # ================= TAB 2: API 与模型 =================
            with gr.Tab("🔌 API 与模型"):
                with gr.Column(elem_classes="glass-panel"):
                    gr.Markdown("### 🔒 API 密钥与模型管理")
                    gr.Markdown("在下方输入您的密钥，修改完成后请点击**保存密钥**按钮，系统将加密写入 `.env` 文件。")
                    
                    gemini_key = gr.Textbox(label="GEMINI_API_KEY", value=get_env("GEMINI_API_KEY"), type="password")
                    gemini_base_url = gr.Textbox(label="Gemini API 地址", value=gemini_base_url_value)
                    gemini_model = gr.Dropdown(
                        choices=[(gemini_saved_model, gemini_saved_model)] if gemini_saved_model else [],
                        value=gemini_saved_model or None,
                        label="Gemini 模型",
                        allow_custom_value=True,
                    )
                    gemini_refresh_btn = gr.Button("刷新 Gemini 模型")
                    gr.Markdown("测试可能产生极少量 API 用量。")
                    gemini_test_btn = gr.Button("测试 Gemini 模型")
                    gemini_status = gr.Markdown("")

                    nvidia_key = gr.Textbox(label="NVIDIA_API_KEY (Kimi)", value=get_env("NVIDIA_API_KEY"), type="password")
                    nvidia_base_url = gr.Textbox(label="NVIDIA API 地址", value=nvidia_base_url_value)
                    nvidia_model = gr.Dropdown(
                        choices=[(nvidia_saved_model, nvidia_saved_model)] if nvidia_saved_model else [],
                        value=nvidia_saved_model or None,
                        label="NVIDIA 模型",
                        allow_custom_value=True,
                    )
                    nvidia_refresh_btn = gr.Button("刷新 NVIDIA 模型")
                    gr.Markdown("测试可能产生极少量 API 用量。")
                    nvidia_test_btn = gr.Button("测试 NVIDIA 模型")
                    nvidia_status = gr.Markdown("")

                    lovart_access = gr.Textbox(label="LOVART_ACCESS_KEY", value=get_env("LOVART_ACCESS_KEY"), type="password")
                    lovart_secret = gr.Textbox(label="LOVART_SECRET_KEY", value=get_env("LOVART_SECRET_KEY"), type="password")
                    
                    save_keys_btn = gr.Button("💾 保存密钥 (Save Keys)", variant="primary")
                    save_status = gr.Markdown("")
                    
                    key_inputs = [gemini_key, nvidia_key, lovart_access, lovart_secret]
                    save_keys_btn.click(fn=manual_save_keys, inputs=key_inputs, outputs=save_status)

                    gemini_refresh_btn.click(
                        fn=lambda key, url, model, source: refresh_provider_controls("gemini", key, url, model, source),
                        inputs=[gemini_key, gemini_base_url, gemini_model, prompt_source],
                        outputs=[gemini_status, gemini_model, prompt_model, gemini_catalog_state],
                    )
                    nvidia_refresh_btn.click(
                        fn=lambda key, url, model, source: refresh_provider_controls("nvidia", key, url, model, source),
                        inputs=[nvidia_key, nvidia_base_url, nvidia_model, prompt_source],
                        outputs=[nvidia_status, nvidia_model, prompt_model, nvidia_catalog_state],
                    )
                    gemini_test_btn.click(
                        fn=lambda key, url, model: test_provider_model("gemini", key, url, model),
                        inputs=[gemini_key, gemini_base_url, gemini_model],
                        outputs=gemini_status,
                    )
                    nvidia_test_btn.click(
                        fn=lambda key, url, model: test_provider_model("nvidia", key, url, model),
                        inputs=[nvidia_key, nvidia_base_url, nvidia_model],
                        outputs=nvidia_status,
                    )

            # ================= TAB 3: 提示词设置 =================
            with gr.Tab("📝 提示词设置"):
                with gr.Column(elem_classes="glass-panel"):
                    gr.Markdown("### 📝 长期提示词参数")
                    gr.Markdown(
                        "⚠️ **优先级说明：Excel 中已填写的商品名、语言、图片尺寸/比例、卖点和参考图属性始终优先；这里的设置仅作为 Excel 未填写时的长期默认值。**"
                    )

                    with gr.Row():
                        prompt_detail_page_count = gr.Number(
                            label="详情页屏数（1-50，一屏一张成品图）",
                            value=prompt_form_values[0],
                            precision=0,
                        )
                        prompt_image_quality = gr.Textbox(
                            label="图片画质",
                            value=prompt_form_values[3],
                        )
                        prompt_allow_questions = gr.Checkbox(
                            label="允许模型反问",
                            value=prompt_form_values[10],
                        )

                    prompt_design_style = gr.Textbox(
                        label="整体设计风格",
                        value=prompt_form_values[1],
                    )
                    prompt_required_sections = gr.CheckboxGroup(
                        choices=required_section_choices,
                        value=prompt_form_values[2],
                        label="每屏必须包含的内容",
                    )

                    with gr.Row():
                        prompt_logo_policy = gr.Textbox(
                            label="Logo 规则",
                            value=prompt_form_values[4],
                        )
                        prompt_copy_style = gr.Textbox(
                            label="文案风格",
                            value=prompt_form_values[5],
                        )
                        prompt_copy_detail_level = gr.Textbox(
                            label="文案详细程度",
                            value=prompt_form_values[6],
                        )

                    prompt_product_fidelity = gr.Textbox(
                        label="产品还原强调程度",
                        value=prompt_form_values[7],
                    )
                    prompt_white_background_requirements = gr.Textbox(
                        label="白底图精修要求",
                        value=prompt_form_values[8],
                        lines=4,
                    )
                    prompt_scene_requirements = gr.Textbox(
                        label="场景图生成要求",
                        value=prompt_form_values[9],
                        lines=4,
                    )

                    with gr.Row():
                        prompt_default_language = gr.Textbox(
                            label="Excel 未填写语言时的默认语言",
                            value=prompt_form_values[11],
                        )
                        prompt_missing_image_size_policy = gr.Textbox(
                            label="Excel 未填写图片尺寸时的处理规则",
                            value=prompt_form_values[12],
                        )

                    prompt_extra_requirements = gr.Textbox(
                        label="自定义额外要求",
                        value=prompt_form_values[13],
                        lines=5,
                    )

                    prompt_form_inputs = [
                        prompt_detail_page_count,
                        prompt_design_style,
                        prompt_required_sections,
                        prompt_image_quality,
                        prompt_logo_policy,
                        prompt_copy_style,
                        prompt_copy_detail_level,
                        prompt_product_fidelity,
                        prompt_white_background_requirements,
                        prompt_scene_requirements,
                        prompt_allow_questions,
                        prompt_default_language,
                        prompt_missing_image_size_policy,
                        prompt_extra_requirements,
                    ]
                    with gr.Row():
                        prompt_save_btn = gr.Button("💾 保存设置", variant="primary")
                        prompt_reset_btn = gr.Button("↩️ 恢复默认值")
                    prompt_save_status = gr.Markdown("")
                    prompt_effective_preview = gr.Textbox(
                        label="当前最终生效规则预览",
                        value=prompt_preview_value,
                        lines=18,
                        interactive=False,
                    )

                    prompt_save_btn.click(
                        fn=save_prompt_settings_from_form,
                        inputs=prompt_form_inputs,
                        outputs=[prompt_save_status, prompt_effective_preview],
                    )
                    prompt_reset_btn.click(
                        fn=reset_prompt_settings_form,
                        inputs=[],
                        outputs=[*prompt_form_inputs, prompt_effective_preview],
                    )

            # ================= TAB 4: 系统更新 =================
            with gr.Tab("⚙️ 系统更新 (OTA)"):
                with gr.Column(elem_classes="glass-panel"):
                    from version import VERSION
                    gr.HTML(f"<h3 style='margin-bottom: 0;'>🔄 OTA 自动热更新引擎</h3><p style='color: gray;'>当前客户端版本: <b>v{VERSION}</b></p>")
                    
                    check_update_btn = gr.Button("🔍 检查新版本并全自动覆盖升级", variant="secondary")
                    update_log = gr.Textbox(label="更新状态日志", lines=8, autoscroll=True)

        # 绑定按钮事件
        prompt_source.change(
            fn=resolve_workspace_model,
            inputs=[prompt_source, gemini_catalog_state, nvidia_catalog_state, gemini_model, nvidia_model],
            outputs=prompt_model,
        )
        prompt_model.input(
            fn=retain_workspace_selection,
            inputs=[prompt_source, prompt_model, gemini_model, nvidia_model],
            outputs=[gemini_model, nvidia_model],
        )
        gemini_model.change(
            fn=lambda source, model: sync_workspace_model(source, "gemini_api", model),
            inputs=[prompt_source, gemini_model],
            outputs=prompt_model,
        )
        nvidia_model.change(
            fn=lambda source, model: sync_workspace_model(source, "nvidia", model),
            inputs=[prompt_source, nvidia_model],
            outputs=prompt_model,
        )
        start_btn.click(
            fn=run_process,
            inputs=[
                excel_file, custom_output_dir, prompt_source, prompt_model, lovart_mode, lovart_image_model,
                gemini_key, nvidia_key, lovart_access, lovart_secret
            ],
            outputs=progress_dashboard
        )
        
        def shutdown_server():
            import os
            import threading
            import time
            def kill():
                time.sleep(1)
                cleanup_processes()
                os._exit(0)
            threading.Thread(target=kill).start()
            return gr.update(value="进程已结束，请关闭本页面", interactive=False)

        shutdown_btn.click(
            fn=shutdown_server,
            inputs=[],
            outputs=[shutdown_btn]
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
