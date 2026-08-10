import os
import json
import subprocess
import threading
import time
import atexit
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

import gradio as gr
import yaml

from failed_retry import (
    RETRY_ERROR_TYPE_CHOICES,
    RETRY_MODE_FINITE,
    RETRY_MODE_INFINITE,
    RETRY_MODE_OFF,
    FailedRetryPolicy,
    normalize_retry_delay,
    normalize_retry_error_types,
    normalize_retry_mode,
    normalize_retry_rounds,
)
from model_provider import (
    DiscoveredModel,
    ModelProviderError,
    discover_models,
    model_choice_labels,
    test_selected_model,
    validate_base_url,
    validate_model_id,
)
from prompt_settings import (
    DEFAULT_PROMPT_SETTINGS,
    effective_rules_preview,
    get_prompt_settings,
    merge_prompt_settings,
    normalize_prompt_settings,
)
from gemini_browser_session import (
    GeminiPageState,
    build_login_helper_command,
    clear_stale_login_runtime,
    login_helper_is_active,
    login_runtime_paths,
    read_login_status,
    request_login_helper_close,
)
from lovart_api import AgentSkill, AgentSkillError
from lovart_bot import LOVART_IMAGE_MODELS, LOVART_MODEL_LABELS, unlimited_model_catalog
from image_generation import normalize_image_provider
from openai_image_api import (
    OpenAIImageAPI,
    OpenAIImageAPIConfig,
    OpenAIImageAPIError,
    normalize_openai_image_base_url,
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
API_SETTINGS_SAVE_SUCCESS = "✅ 密钥、API 地址和模型已保存"
OPENAI_IMAGE_TEST_BUTTON_LABEL = "真实图像编辑测试（可能产生一次图片费用）"
_gemini_login_launch_lock = threading.Lock()
_gemini_login_launches: dict[str, float] = {}
_GEMINI_LOGIN_LAUNCH_GRACE_SECONDS = 10.0
_GEMINI_PROFILE_BUSY_MESSAGE = "Gemini 浏览器账户目录正在使用中，请等待当前浏览器任务结束后再试。"


def open_gemini_login_browser(config_path: str | Path = "config.yaml") -> str:
    paths = login_runtime_paths(config_path)
    launch_key = str(paths.owner_lock_path.resolve())
    with _gemini_login_launch_lock:
        now = time.monotonic()
        launch_started = _gemini_login_launches.get(launch_key)
        if launch_started is not None and now - launch_started >= _GEMINI_LOGIN_LAUNCH_GRACE_SECONDS:
            _gemini_login_launches.pop(launch_key, None)
        if login_helper_is_active(paths):
            _gemini_login_launches.pop(launch_key, None)
            return _GEMINI_PROFILE_BUSY_MESSAGE
        clear_stale_login_runtime(paths)
        if login_helper_is_active(paths):
            return _GEMINI_PROFILE_BUSY_MESSAGE
        if launch_key in _gemini_login_launches:
            return _GEMINI_PROFILE_BUSY_MESSAGE
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        kwargs: dict[str, object] = {"env": env}
        import sys
        if os.name == "nt" and getattr(sys, "frozen", False):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            subprocess.Popen(build_login_helper_command(config_path), **kwargs)
        except OSError:
            return "无法打开 Gemini 登录浏览器，请检查本地安装后重试。"
        _gemini_login_launches[launch_key] = now
    return "Gemini 登录浏览器已打开，请在新窗口中完成登录后再检查。"


def check_gemini_login_and_close(config_path: str | Path = "config.yaml") -> str:
    paths = login_runtime_paths(config_path)
    status = read_login_status(paths.status_path)
    if not login_helper_is_active(paths):
        return "Gemini 登录助手未运行；请先打开登录浏览器。"
    if status is None or status.state != GeminiPageState.READY or not status.ready:
        return "Gemini 尚未完成登录，请继续在登录浏览器中操作。"
    request_login_helper_close(paths.close_request_path)
    return "Gemini 登录已确认，正在安全关闭登录浏览器。"


def guard_gemini_browser_task(
    prompt_source: str, config_path: str | Path = "config.yaml"
) -> str | None:
    if prompt_source == "gemini_browser" and login_helper_is_active(login_runtime_paths(config_path)):
        return _GEMINI_PROFILE_BUSY_MESSAGE
    return None


def build_gemini_health_check_command(
    config_path: str | Path,
    status_file: str | Path,
    *,
    executable: str | None = None,
    frozen: bool | None = None,
) -> list[str]:
    import sys

    program = executable or sys.executable
    args = [
        "--gemini-health-check",
        "--config",
        str(Path(config_path).resolve()),
        "--status-file",
        str(Path(status_file).resolve()),
    ]
    is_frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    if is_frozen:
        return [program, *args]
    return [program, str(Path(__file__).with_name("app.py").resolve()), *args]


def _health_check_runs_root() -> Path:
    return Path("runs/gemini_health_check").resolve()


def _start_gemini_health_check_process(
    config_path: str | Path, status_file: Path
) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    kwargs: dict[str, object] = {
        "env": env,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    process = subprocess.Popen(
        build_gemini_health_check_command(config_path, status_file),
        **kwargs,
    )
    active_processes.append(process)
    return process


def _read_health_check_records(
    status_file: Path, offset: int
) -> tuple[list[dict], int]:
    if not status_file.exists():
        return [], offset
    with status_file.open("rb") as handle:
        handle.seek(offset)
        data = handle.read()
    complete_end = data.rfind(b"\n")
    if complete_end < 0:
        return [], offset
    consumed = data[: complete_end + 1]
    records = []
    for raw_line in consumed.splitlines():
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return records, offset + len(consumed)


def _render_health_check_results(
    results: list[dict], *, final_message: str | None = None
) -> str:
    icons = {"pass": "✅", "fail": "❌", "skip": "⚠️"}
    counts = {"pass": 0, "fail": 0, "skip": 0}
    lines = ["### Gemini 完整体检", ""]
    for result in results:
        state = str(result.get("state", "fail"))
        if state not in counts:
            state = "fail"
        counts[state] += 1
        name = str(result.get("name", "未命名检查项"))
        message = str(result.get("message", "")).strip()
        detail = f"：{message}" if message else ""
        lines.append(f"{icons[state]} **{name}**{detail}")
    if not results:
        lines.append("正在等待体检结果…")
    if final_message:
        lines.extend(["", final_message])
    lines.extend(
        [
            "",
            (
                f"**汇总：正常 {counts['pass']} · "
                f"异常 {counts['fail']} · 跳过 {counts['skip']}**"
            ),
        ]
    )
    return "\n".join(lines)


def _stop_health_check_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt" and getattr(process, "pid", None):
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def ui_run_gemini_health_check(
    config_path: str | Path = "config.yaml",
):
    run_dir = _health_check_runs_root() / str(time.time_ns())
    status_file = run_dir / "status.jsonl"
    process = None
    results: list[dict] = []
    yield "正在启动 Gemini 完整体检…"
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        process = _start_gemini_health_check_process(config_path, status_file)
        offset = 0
        complete = False
        complete_exit_code = None
        deadline = time.monotonic() + 1800
        while True:
            records, offset = _read_health_check_records(status_file, offset)
            for record in records:
                event = record.get("event")
                if event == "result" and isinstance(record.get("result"), dict):
                    results.append(record["result"])
                    yield _render_health_check_results(results)
                elif event == "complete":
                    complete = True
                    complete_exit_code = int(record.get("exit_code", 1))
            if complete:
                final_message = None
                if complete_exit_code:
                    final_message = "❌ Gemini 完整体检未通过，请查看异常项目。"
                yield _render_health_check_results(
                    results,
                    final_message=final_message,
                )
                break
            returncode = process.poll()
            if returncode is not None:
                final_records, offset = _read_health_check_records(status_file, offset)
                for record in final_records:
                    if (
                        record.get("event") == "result"
                        and isinstance(record.get("result"), dict)
                    ):
                        results.append(record["result"])
                    elif record.get("event") == "complete":
                        complete = True
                        complete_exit_code = int(record.get("exit_code", 1))
                if complete:
                    final_message = None
                    if complete_exit_code:
                        final_message = "❌ Gemini 完整体检未通过，请查看异常项目。"
                    yield _render_health_check_results(
                        results,
                        final_message=final_message,
                    )
                    break
                yield _render_health_check_results(
                    results,
                    final_message=f"❌ 体检进程提前退出（代码 {returncode}）。",
                )
                break
            if time.monotonic() >= deadline:
                _stop_health_check_process(process)
                yield _render_health_check_results(
                    results,
                    final_message="❌ 体检超时，已停止本次独立体检进程。",
                )
                break
            time.sleep(0.25)
    except Exception:
        if process is not None:
            _stop_health_check_process(process)
        yield _render_health_check_results(
            results,
            final_message="❌ 无法启动 Gemini 完整体检，请稍后重试。",
        )
    finally:
        if process is not None and process.poll() is None:
            _stop_health_check_process(process)
        if process in active_processes:
            active_processes.remove(process)
        try:
            status_file.unlink(missing_ok=True)
            run_dir.rmdir()
        except OSError:
            pass


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
  network_attempts: 5
  page_ready_timeout: 90
  product_attempts: 2
  retry_delays: [3, 6, 12, 20]
gemini:
  preamble_file: preamble.txt
  base_url: https://gemini.google.com
  thinking_mode: true
  allow_regular_chat_fallback: false
  reply_timeout: 300
  upload_timeout: 120
  upload_attempts: 3
gemini_api:
  base_url: https://generativelanguage.googleapis.com/v1beta
  model: gemini-2.5-flash-lite
nvidia_api:
  base_url: https://integrate.api.nvidia.com/v1
  model: moonshotai/kimi-k2.5
  # model_choice/models are retained for legacy configuration compatibility only.
  model_choice: kimi
  send_images: true
  models:
    kimi: moonshotai/kimi-k2.5
openai_image:
  base_url: ""
  model: gpt-image-2
  resolution: 1K
image_generation:
  support_provider: lovart
  detail_provider: lovart
prompt_settings:
  detail_page_count: 12
  design_style: "温馨感、高级感"
  required_sections: ["主标题", "副标题", "信息布局", "排版形式"]
  image_quality: "1K"
  logo_policy: "不出现 Logo"
  copy_style: "适合跨境电商，具体、不空泛"
  copy_detail_level: "详细"
  product_fidelity: "严格还原"
  white_background_requirements: "白底、超清摄影、突出高级感，产品造型与原图一致"
  scene_requirements: "重新设计场景，产品特征与原图保持一致，超清摄影"
  allow_questions: false
  default_language: "巴西葡萄牙语"
  missing_image_size_policy: "不使用默认固定图片比例"
  extra_requirements: ""
lovart:
  base_url: https://lgw.lovart.ai
  image_model: auto
  model_selection: prefer
  unlimited_models: []
  unlimited_model_catalog: []
  reasoning_mode: fast
  wait_forever_on_credit_prompt: true
  max_confirmation_rounds: 5
  max_auto_confirm_credits: 10
  wait_timeout: 10800
  poll_interval: 10
  poll_request_timeout: 10
  poll_request_attempts: 1
  timeout: 600
  upload_attempts: 3
  upload_retry_delay: 2
  failed_retry_mode: finite
  failed_retry_rounds: 2
  failed_retry_delay: 15
  failed_retry_error_types:
    - lovart_service
    - network
    - timeout
    - gemini_page
    - gemini_upload
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


def _capture_file_snapshot(path: str | Path) -> tuple[bool, bytes]:
    target = Path(path)
    return (target.exists(), target.read_bytes() if target.exists() else b"")


def _restore_file_snapshot(path: str | Path, snapshot: tuple[bool, bytes]):
    target = Path(path)
    existed, data = snapshot
    if not existed:
        target.unlink(missing_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.rollback.tmp")
    temp.write_bytes(data)
    try:
        os.replace(temp, target)
    except Exception as exc:
        backup_path = temp.resolve()
        raise OSError(
            f"恢复替换失败；原始副本已保留在 {backup_path}: {exc}"
        ) from exc
    if temp.exists():
        temp.unlink()


def _restore_file_snapshots(snapshots) -> list[str]:
    restore_errors = []
    for path, snapshot in snapshots.items():
        try:
            _restore_file_snapshot(path, snapshot)
        except Exception as restore_error:
            restore_errors.append(f"{path}: {restore_error}")
    return restore_errors


def _save_config_and_env_transaction(
    config_data,
    gemini_key,
    nvidia_key,
    lovart_access,
    lovart_secret,
    openai_image_key=None,
    clear_openai_image_key=False,
    config_path="config.yaml",
    env_path=".env",
    snapshots=None,
):
    """Save config and credentials as one compensating two-file transaction."""
    if snapshots is None:
        snapshots = {
            Path(config_path): _capture_file_snapshot(config_path),
            Path(env_path): _capture_file_snapshot(env_path),
        }
    try:
        save_config(config_data, config_path)
        save_env(
            gemini_key,
            nvidia_key,
            lovart_access,
            lovart_secret,
            openai_image_key=openai_image_key,
            clear_openai_image_key=clear_openai_image_key,
            env_path=env_path,
        )
    except Exception as primary:
        restore_errors = _restore_file_snapshots(snapshots)
        message = str(primary)
        if restore_errors:
            message += "；恢复也失败：" + "；".join(restore_errors)
        raise OSError(message) from primary


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
    try:
        current = yaml.safe_load(target.read_text(encoding="utf-8")) or {} if target.exists() else {}
        if not isinstance(current, dict):
            raise ValueError("config.yaml 顶层必须是配置对象")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return (
            f"❌ 读取 config.yaml 失败，请修复文件后重试：{exc}",
            effective_rules_preview(DEFAULT_PROMPT_SETTINGS),
        )
    try:
        settings = form_to_prompt_settings(*values)
        updated = merge_prompt_settings(current, settings)
    except (TypeError, ValueError) as exc:
        existing_settings = get_prompt_settings(current)
        return f"❌ {exc}", effective_rules_preview(existing_settings)

    try:
        save_config(updated, target)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return f"❌ 配置保存失败，原文件已保留：{exc}", effective_rules_preview(get_prompt_settings(current))
    return "✅ 提示词设置已保存", effective_rules_preview(settings)


def save_gemini_browser_settings(
    allow_regular_chat_fallback,
    config_path="config.yaml",
) -> str:
    target = Path(config_path)
    try:
        current = yaml.safe_load(target.read_text(encoding="utf-8")) or {} if target.exists() else {}
        if not isinstance(current, dict):
            raise ValueError("config.yaml 顶层必须是配置对象")
        updated = deepcopy(current)
        updated.setdefault("gemini", {})["allow_regular_chat_fallback"] = bool(
            allow_regular_chat_fallback
        )
        save_config(updated, target)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return f"❌ Gemini 浏览器设置保存失败，原配置已保留：{exc}"
    return "✅ Gemini 浏览器设置已保存"


def reset_prompt_settings_form() -> tuple:
    defaults = normalize_prompt_settings(DEFAULT_PROMPT_SETTINGS)
    values = [deepcopy(defaults[field]) for field in PROMPT_FORM_FIELDS]
    values[PROMPT_FORM_FIELDS.index("required_sections")] = "\n".join(defaults["required_sections"])
    return (*values, effective_rules_preview(defaults))


def refresh_provider_models(provider, api_key, base_url, current_model):
    try:
        models = discover_models(provider, api_key, base_url)
    except ModelProviderError as exc:
        choices = [(current_model, current_model)] if current_model else []
        return f"❌ {exc.user_message}", choices, current_model, []

    choices = model_choice_labels(models)
    model_ids = [model.model_id for model in models]
    selected = select_model_id(models, current_model)
    if selected and selected not in model_ids and not models:
        choices.append((selected, selected))
    return f"✅ 成功获取 {len(models)} 个可用模型。", choices, selected, [asdict(model) for model in models]


def select_model_id(models, current_model=""):
    """Select the current model, then a recommendation, then the first model."""
    model_ids = [model.model_id for model in models]
    if current_model in model_ids:
        return current_model
    recommended = next(
        (model.model_id for model in models if model.recommendation == "recommended"), None
    )
    return recommended or (model_ids[0] if model_ids else current_model)


def update_catalog_image_status(catalog, model_id, image_input_status):
    """Return a copied runtime catalog with one model's probe status updated."""
    updated = deepcopy(catalog or [])
    for item in updated:
        if item.get("model_id") == model_id:
            item["image_input_status"] = image_input_status
    return updated


def test_provider_model(provider, api_key, base_url, model_id):
    try:
        result = test_selected_model(provider, api_key, base_url, model_id)
    except ModelProviderError as exc:
        return f"❌ {exc.user_message} 测试可能产生极少量 API 用量。"
    status_icon = "✅" if result.ok else "❌"
    return f"{status_icon} {result.message}（{result.latency_ms} ms）。测试可能产生极少量 API 用量。"


def probe_provider_model(provider, api_key, base_url, model_id, catalog):
    """Probe one model and return status plus a relabeled, selection-preserving catalog."""
    working_catalog = deepcopy(catalog or [])
    try:
        normalized_model_id = validate_model_id(model_id)
    except ModelProviderError:
        normalized_model_id = None
    effective_model_id = normalized_model_id if normalized_model_id is not None else model_id
    if normalized_model_id and not any(
        item.get("model_id") == normalized_model_id for item in working_catalog
    ):
        working_catalog.append(asdict(DiscoveredModel(
            provider=str(provider or "").strip().lower(),
            model_id=normalized_model_id,
            display_name=normalized_model_id,
            supports_generation=True,
            supports_thinking=None,
            image_input_status="unknown",
            recommendation="available",
        )))
    try:
        result = test_selected_model(provider, api_key, base_url, effective_model_id)
        succeeded = bool(result.ok)
        status_icon = "✅" if succeeded else "❌"
        status = (
            f"{status_icon} {result.message}（{result.latency_ms} ms）。"
            "测试可能产生极少量 API 用量。"
        )
    except ModelProviderError as exc:
        succeeded = False
        status = f"❌ {exc.user_message} 测试可能产生极少量 API 用量。"
    updated_catalog = update_catalog_image_status(
        working_catalog, effective_model_id, "verified" if succeeded else "failed"
    )
    models = [DiscoveredModel(**item) for item in updated_catalog]
    return status, model_choice_labels(models), effective_model_id, updated_catalog


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
    configured = _configured_provider_model(config, prompt_source)
    selected = select_model_id(models, configured)
    if selected and selected not in model_ids and not models:
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


def persist_provider_settings(
    config,
    gemini_base_url,
    gemini_model,
    nvidia_base_url,
    nvidia_model,
):
    """Validate and copy both providers' long-term endpoint/model settings."""
    normalized_gemini_url = validate_base_url(gemini_base_url)
    normalized_gemini_model = validate_model_id(gemini_model)
    normalized_nvidia_url = validate_base_url(nvidia_base_url)
    normalized_nvidia_model = validate_model_id(nvidia_model)
    updated = deepcopy(config)
    updated.setdefault("gemini_api", {}).update({
        "base_url": normalized_gemini_url,
        "model": normalized_gemini_model,
    })
    updated.setdefault("nvidia_api", {}).update({
        "base_url": normalized_nvidia_url,
        "model": normalized_nvidia_model,
    })
    return updated


def normalize_resolution(resolution) -> str:
    normalized = str(resolution or "1K").strip().upper() or "1K"
    if normalized not in {"1K", "2K", "4K"}:
        raise ValueError("GPT Image 分辨率必须是 1K、2K 或 4K")
    return normalized


def default_merge_reference_images(base_url) -> bool:
    try:
        hostname = (
            urlsplit(str(base_url or "").strip()).hostname or ""
        ).rstrip(".").lower()
    except ValueError:
        return False
    return hostname in {"hapiopen.cc", "image.hapiopen.cc"}


def resolve_merge_reference_images(value, base_url) -> bool:
    if value is None:
        return default_merge_reference_images(base_url)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return default_merge_reference_images(base_url)


def begin_openai_image_test():
    """Immediately acknowledge the paid test and prevent duplicate submissions."""
    return (
        "⏳ **进度：0/1** 已收到请求，正在上传测试图并等待生成。通常需要 "
        "30–120 秒，请勿重复点击或关闭页面。",
        gr.update(value="⏳ 正在生成（0/1）…", interactive=False),
    )


def reset_openai_image_test_button():
    """Restore the paid-test button after either success or failure."""
    return gr.update(value=OPENAI_IMAGE_TEST_BUTTON_LABEL, interactive=True)


def test_openai_image_edit(
    api_key,
    base_url,
    model,
    resolution,
    merge_reference_images=None,
) -> str:
    """Run the explicitly requested, potentially billable GPT Image edit probe."""
    try:
        config = {
            "openai_image": {
                "base_url": base_url,
                "model": model,
                "resolution": resolution,
                "merge_reference_images": (
                    resolve_merge_reference_images(
                        merge_reference_images,
                        base_url,
                    )
                ),
            }
        }
        client = OpenAIImageAPI(
            OpenAIImageAPIConfig.from_config(config, api_key=api_key),
            logger=None,
        )
        result = client.test_edit(Path("output") / ".api-tests")
    except (OpenAIImageAPIError, OSError, ValueError) as exc:
        message = exc.user_message if isinstance(exc, OpenAIImageAPIError) else str(exc)
        return f"❌ **进度：0/1** GPT Image 图生图测试失败：{message}"
    return (
        f"✅ **进度：1/1** GPT Image 图生图测试成功：{Path(result.local_path).name}。"
        "本次测试可能已产生一次图片费用。"
    )


def persist_openai_image_settings(
    config,
    base_url,
    model,
    resolution,
    support_provider,
    detail_provider,
    merge_reference_images=None,
):
    updated = deepcopy(config)
    normalized_base_url = normalize_openai_image_base_url(base_url)
    if merge_reference_images is None:
        existing = updated.get("openai_image", {})
        if isinstance(existing, dict) and "merge_reference_images" in existing:
            merge_reference_images = resolve_merge_reference_images(
                existing["merge_reference_images"],
                normalized_base_url,
            )
        else:
            merge_reference_images = default_merge_reference_images(normalized_base_url)
    updated["openai_image"] = {
        **updated.get("openai_image", {}),
        "base_url": normalized_base_url,
        "model": str(model or "gpt-image-2").strip() or "gpt-image-2",
        "resolution": normalize_resolution(resolution),
        "merge_reference_images": resolve_merge_reference_images(
            merge_reference_images,
            normalized_base_url,
        ),
    }
    updated["openai_image"].pop("api_key", None)
    updated["image_generation"] = {
        **updated.get("image_generation", {}),
        "support_provider": normalize_image_provider(support_provider),
        "detail_provider": normalize_image_provider(detail_provider),
    }
    return updated


def persist_image_routing_settings(config, support_provider, detail_provider):
    updated = deepcopy(config)
    openai_image = updated.get("openai_image")
    if isinstance(openai_image, dict):
        openai_image.pop("api_key", None)
    updated["image_generation"] = {
        **updated.get("image_generation", {}),
        "support_provider": normalize_image_provider(support_provider),
        "detail_provider": normalize_image_provider(detail_provider),
    }
    return updated


def save_api_settings(
    gemini_key,
    nvidia_key,
    lovart_access,
    lovart_secret,
    openai_image_key,
    gemini_base_url,
    gemini_model,
    nvidia_base_url,
    nvidia_model,
    openai_image_base_url,
    openai_image_model,
    openai_image_resolution,
    support_provider,
    detail_provider,
    clear_openai_image_key=False,
    merge_reference_images=None,
    config_path="config.yaml",
    env_path=".env",
):
    """Validate and persist provider endpoints/models, then save credentials."""
    target = Path(config_path)
    try:
        current = yaml.safe_load(target.read_text(encoding="utf-8")) or {} if target.exists() else {}
        if not isinstance(current, dict):
            raise ValueError("config.yaml 顶层必须是配置对象")
        updated = persist_provider_settings(
            current, gemini_base_url, gemini_model, nvidia_base_url, nvidia_model
        )
        updated = persist_openai_image_settings(
            updated,
            openai_image_base_url,
            openai_image_model,
            openai_image_resolution,
            support_provider,
            detail_provider,
            merge_reference_images,
        )
        _save_config_and_env_transaction(
            updated,
            gemini_key,
            nvidia_key,
            lovart_access,
            lovart_secret,
            openai_image_key=openai_image_key,
            clear_openai_image_key=clear_openai_image_key,
            config_path=target,
            env_path=env_path,
        )
    except (ModelProviderError, OpenAIImageAPIError, OSError, ValueError, yaml.YAMLError) as exc:
        message = exc.user_message if isinstance(exc, ModelProviderError) else str(exc)
        return f"❌ API 与模型设置保存失败，原配置未被部分覆盖：{message}"
    return API_SETTINGS_SAVE_SUCCESS


def normalize_lovart_model_catalog(catalog) -> list[dict]:
    normalized = []
    seen = set()
    for item in catalog or []:
        if not isinstance(item, dict):
            continue
        model = str(item.get("model") or "").strip()
        tool_name = LOVART_IMAGE_MODELS.get(model)
        if not tool_name or model in seen:
            continue
        seen.add(model)
        normalized.append({
            "model": model,
            "tool_name": tool_name,
            "label": str(item.get("label") or LOVART_MODEL_LABELS.get(model, model)),
            "restriction": str(item.get("restriction") or ""),
        })
    return normalized


def lovart_model_choices(catalog, order=None):
    normalized = normalize_lovart_model_catalog(catalog)
    by_model = {item["model"]: item for item in normalized}
    ordered_models = [model for model in (order or []) if model in by_model]
    ordered_models.extend(model for model in by_model if model not in ordered_models)
    choices = []
    for model in ordered_models:
        item = by_model[model]
        suffix = f" ({item['restriction']})" if item["restriction"] else ""
        choices.append((f"{item['label']}{suffix}", model))
    return choices


def format_lovart_model_order(selected, order, catalog) -> str:
    selected_set = set(selected or [])
    labels = {model: label for label, model in lovart_model_choices(catalog, order)}
    ordered = [model for model in (order or []) if model in selected_set]
    if not ordered:
        return "尚未选择无限模型。"
    return "  >  ".join(
        f"{index}. {labels.get(model, model)}"
        for index, model in enumerate(ordered, 1)
    )


def move_lovart_model(model, order, direction):
    current = [item for item in (order or []) if isinstance(item, str)]
    if model not in current or direction not in {-1, 1}:
        return current
    index = current.index(model)
    target = index + direction
    if 0 <= target < len(current):
        current[index], current[target] = current[target], current[index]
    return current


def detect_lovart_unlimited_models(
    access_key,
    secret_key,
    *,
    config_path="config.yaml",
):
    access_key = str(access_key or "").strip()
    secret_key = str(secret_key or "").strip()
    if not access_key or not secret_key:
        return "❌ 请先填写 Lovart Access Key 和 Secret Key。", []

    config = load_config(config_path)
    lovart_config = config.get("lovart", {}) or {}
    client = AgentSkill(
        base_url=os.environ.get(
            "LOVART_BASE_URL", lovart_config.get("base_url", "https://lgw.lovart.ai")
        ),
        access_key=access_key,
        secret_key=secret_key,
        timeout=min(int(lovart_config.get("timeout", 600)), 60),
        poll_interval=lovart_config.get("poll_interval", 10),
    )
    try:
        mode_state = client.query_mode()
    except (AgentSkillError, OSError, ValueError) as exc:
        return f"❌ Lovart 无限模型检测失败：{exc}", []

    if mode_state.get("unlimited_enable") is False:
        return "❌ 当前 Lovart 账号没有启用无限生成权限。", []
    catalog = normalize_lovart_model_catalog(unlimited_model_catalog(mode_state))
    if not catalog:
        return "❌ Lovart 没有返回本客户端支持的无限生成模型。", []

    channel = "无限生成" if mode_state.get("unlimited") else "高速生成"
    details = "、".join(
        item["label"] + (f" ({item['restriction']})" if item["restriction"] else "")
        for item in catalog
    )
    return f"✅ 检测到 {len(catalog)} 个无限模型；当前通道：{channel}。\n\n{details}", catalog


def save_lovart_unlimited_models(
    selected_models,
    model_order,
    catalog,
    *,
    config_path="config.yaml",
):
    target = Path(config_path)
    try:
        normalized_catalog = normalize_lovart_model_catalog(catalog)
        available = {item["model"] for item in normalized_catalog}
        selected = [model for model in (selected_models or []) if model in available]
        selected_set = set(selected)
        ordered = [model for model in (model_order or []) if model in selected_set]
        ordered.extend(model for model in selected if model not in ordered)
        if not ordered:
            raise ValueError("请至少勾选一个检测到的无限模型")

        current = yaml.safe_load(target.read_text(encoding="utf-8")) or {} if target.exists() else {}
        if not isinstance(current, dict):
            raise ValueError("config.yaml 顶层必须是配置对象")
        current.setdefault("lovart", {}).update({
            "unlimited_models": ordered,
            "unlimited_model_catalog": normalized_catalog,
            "unlimited_models_checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        save_config(current, target)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return f"❌ 无限模型设置保存失败：{exc}"
    return "✅ 无限模型及尝试顺序已保存；任务执行时仍会实时复核套餐权限。"


def save_failed_retry_settings(
    mode,
    rounds,
    delay,
    error_types,
    config_path="config.yaml",
):
    target = Path(config_path)
    try:
        current = yaml.safe_load(target.read_text(encoding="utf-8")) or {} if target.exists() else {}
        if not isinstance(current, dict):
            raise ValueError("config.yaml 顶层必须是配置对象")
        normalized_mode = normalize_retry_mode(mode)
        normalized_rounds = normalize_retry_rounds(rounds)
        normalized_delay = normalize_retry_delay(delay)
        normalized_types = normalize_retry_error_types(error_types)
        current.setdefault("lovart", {}).update({
            "failed_retry_mode": normalized_mode,
            "failed_retry_rounds": normalized_rounds,
            "failed_retry_delay": normalized_delay,
            "failed_retry_error_types": list(normalized_types),
        })
        save_config(current, target)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return f"❌ 失败任务重试设置保存失败：{exc}"
    return "✅ 失败任务重试设置已保存，下次运行任务时生效"


def failed_retry_rounds_update(mode):
    return gr.update(interactive=mode == RETRY_MODE_FINITE)


def save_env(
    gemini_key: str,
    nvidia_key: str,
    lovart_access: str,
    lovart_secret: str,
    openai_image_key: str | None = None,
    clear_openai_image_key: bool = False,
    env_path: str | Path = ".env",
):
    target = Path(env_path)
    lines = []
    openai_key_provided = bool(str(openai_image_key or "").strip())
    newline = "\n"
    if target.exists():
        with target.open("r", encoding="utf-8", newline="") as f:
            for line in f.readlines():
                if line.endswith("\r\n"):
                    newline = "\r\n"
                if any(line.startswith(k) for k in ["GEMINI_API_KEY=", "NVIDIA_API_KEY=", "LOVART_ACCESS_KEY=", "LOVART_SECRET_KEY="]):
                    continue
                if line.startswith("OPENAI_IMAGE_API_KEY=") and (clear_openai_image_key or openai_key_provided):
                    continue
                lines.append(line)

    if lines and not lines[-1].endswith(("\n", "\r")):
        lines.append(newline)
    lines.append(f"GEMINI_API_KEY={gemini_key}{newline}")
    lines.append(f"NVIDIA_API_KEY={nvidia_key}{newline}")
    lines.append(f"LOVART_ACCESS_KEY={lovart_access}{newline}")
    lines.append(f"LOVART_SECRET_KEY={lovart_secret}{newline}")
    if openai_key_provided and not clear_openai_image_key:
        lines.append(f"OPENAI_IMAGE_API_KEY={str(openai_image_key).strip()}{newline}")
    
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp")
    try:
        temp.write_text("".join(lines), encoding="utf-8", newline="")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def get_env(key: str) -> str:
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip()
    return ""


def _openai_image_key_is_saved(env_path: str | Path = ".env") -> bool:
    target = Path(env_path)
    if not target.exists():
        return False
    with target.open("r", encoding="utf-8") as stream:
        return any(
            line.startswith("OPENAI_IMAGE_API_KEY=")
            and bool(line.split("=", 1)[1].strip())
            for line in stream
        )


def openai_image_key_status(env_path: str | Path = ".env") -> str:
    return (
        "GPT Image 密钥状态：已保存"
        if _openai_image_key_is_saved(env_path)
        else "GPT Image 密钥状态：未保存"
    )


def run_process(
    excel_file,
    custom_output_dir,
    prompt_source,
    prompt_model,
    lovart_mode,
    lovart_image_model,
    gemini_base_url,
    nvidia_base_url,
    gemini_key,
    nvidia_key,
    lovart_access,
    lovart_secret,
    openai_image_key=None,
    openai_image_base_url="",
    openai_image_model="gpt-image-2",
    openai_image_resolution="1K",
    support_provider="lovart",
    detail_provider="lovart",
    clear_openai_image_key=False,
    merge_reference_images=None,
    *,
    config_path="config.yaml",
    env_path=".env",
):
    guard_message = guard_gemini_browser_task(prompt_source, config_path=config_path)
    if guard_message:
        yield f"❌ {guard_message}"
        return
    config_target = Path(config_path)
    env_target = Path(env_path)
    transaction_snapshots = {
        config_target: _capture_file_snapshot(config_target),
        env_target: _capture_file_snapshot(env_target),
    }
    transaction_started = False
    try:
        config = persist_selected_model(load_config(config_path), prompt_source, prompt_model)
        gemini_model = _configured_provider_model(config, "gemini_api") or "gemini-2.5-flash-lite"
        nvidia_model = _configured_provider_model(config, "nvidia") or "moonshotai/kimi-k2.5"
        config = persist_provider_settings(
            config,
            gemini_base_url,
            gemini_model,
            nvidia_base_url,
            nvidia_model,
        )
        config = persist_image_routing_settings(
            config,
            support_provider,
            detail_provider,
        )
        uses_openai_image = "openai_image" in {
            normalize_image_provider(support_provider),
            normalize_image_provider(detail_provider),
        }
        if uses_openai_image:
            submitted_openai_key = bool(str(openai_image_key or "").strip())
            if clear_openai_image_key or not (
                submitted_openai_key or _openai_image_key_is_saved(env_path)
            ):
                raise OpenAIImageAPIError(
                    "missing_key",
                    "请先填写或保存 GPT Image API 密钥。",
                )
            config = persist_openai_image_settings(
                config,
                openai_image_base_url,
                openai_image_model,
                openai_image_resolution,
                support_provider,
                detail_provider,
                merge_reference_images,
            )
        if "lovart" not in config:
            config["lovart"] = {}
        config["lovart"]["image_model"] = lovart_image_model
        config["output_dir"] = custom_output_dir.strip() if custom_output_dir else ""
        transaction_started = True
        _save_config_and_env_transaction(
            config,
            gemini_key,
            nvidia_key,
            lovart_access,
            lovart_secret,
            openai_image_key=openai_image_key,
            clear_openai_image_key=clear_openai_image_key,
            config_path=config_path,
            env_path=env_path,
            snapshots=transaction_snapshots,
        )
    except (ModelProviderError, OpenAIImageAPIError, OSError, ValueError, yaml.YAMLError) as exc:
        message = (
            exc.user_message
            if isinstance(exc, (ModelProviderError, OpenAIImageAPIError))
            else str(exc)
        )
        if not transaction_started and not transaction_snapshots[config_target][0]:
            restore_errors = _restore_file_snapshots({
                config_target: transaction_snapshots[config_target]
            })
            if restore_errors:
                message += "；恢复也失败：" + "；".join(restore_errors)
        yield f"❌ 启动前配置保存失败：{message}"
        return

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
        "--lovart-reasoning", "fast",
        "--support-provider", normalize_image_provider(support_provider),
        "--detail-provider", normalize_image_provider(detail_provider),
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
        is_uidetail = clean_line.startswith("[UI_DETAIL_PROGRESS]")
        is_uistatus = clean_line.startswith("[UI_STATUS]")
        
        if not any((
            is_progress,
            is_uiproduct,
            is_uisuccess,
            is_uifail,
            is_uimodel,
            is_uidetail,
            is_uistatus,
        )):
            logs.append(clean_line)
            if len(logs) > 30:
                logs.pop(0)
            # If log contains a product ID, append to its per-card logs
            if current_pid and current_pid in clean_line and current_pid in products_dict:
                # Strip timestamps or common prefixes to make it cleaner on the card
                clean_msg = clean_line.split("]")[-1].strip() if "]" in clean_line else clean_line
                if "INFO" not in clean_line: # ignore basic INFO lines to save space
                    products_dict[current_pid].setdefault("logs", []).append(f"▶ {clean_msg}")
        
        if is_uistatus:
            try:
                data = json.loads(clean_line.replace("[UI_STATUS]", "", 1).strip())
                pid = str(data.get("id") or "")
                stage = str(data.get("stage") or "product")
                message = str(data.get("message") or "🔄 正在处理")
                if pid in products_dict:
                    current_pid = pid
                    current_product = f"{pid} - {products_dict[pid]['name']}"
                    current_status = message
                    status_color = {
                        "product": "#8b5cf6",
                        "support_white": "#f59e0b",
                        "support_scene": "#f59e0b",
                        "prompt": "#06b6d4",
                        "detail": "#3b82f6",
                    }.get(stage, "#8b5cf6")
                    products_dict[pid]["status"] = message
                    products_dict[pid]["color"] = status_color
                    stage_log = f"▶ {html.escape(message)}"
                    product_logs = products_dict[pid].setdefault("logs", [])
                    if not product_logs or product_logs[-1] != stage_log:
                        product_logs.append(stage_log)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        elif is_uidetail:
            try:
                data = json.loads(clean_line.replace("[UI_DETAIL_PROGRESS]", "", 1).strip())
                current = max(0, int(data.get("current", 0)))
                target = max(0, int(data.get("target", 0)))
                completed = max(0, int(data.get("completed", 0)))
                failed = [int(index) for index in data.get("failed", [])]
                failed_text = f"，失败 {failed}" if failed else ""
                current_status = f"详情图 {current}/{target}，已完成 {completed}{failed_text}"
                status_color = "#f59e0b" if failed else "#3b82f6"
                if current_pid and current_pid in products_dict:
                    products_dict[current_pid]["status"] = current_status
                    products_dict[current_pid]["color"] = status_color
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        elif is_uimodel:
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
                    products_dict[pid]["status"] = "🎉 成功生成"
                    products_dict[pid]["color"] = "#10b981"
                    current_status = f"🎉 {pid} 已完成"
                    status_color = "#10b981"
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
                    progress_line = f"<span data-live-progress='true' style='color: #60a5fa;'>⏳ 绘制进度: {step_str} | 已用时: {time_str}</span>"
                    product_logs = products_dict[current_pid].setdefault("logs", [])
                    if product_logs and "data-live-progress='true'" in product_logs[-1]:
                        product_logs[-1] = progress_line
                    else:
                        product_logs.append(progress_line)
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
    from updater import (
        UpdateStatus,
        check_update_details,
        download_and_install_update,
    )
    import queue
    import threading
    
    yield "正在检查更新，请稍候…"
    result = check_update_details()
    if result.status is UpdateStatus.ERROR:
        yield f"检查更新失败：{result.message or '暂时无法连接更新服务，请稍后重试。'}"
        return
    if result.status is UpdateStatus.UP_TO_DATE:
        yield "当前已是最新版本，无需更新。"
        return
        
    yield (
        f"发现新版本：v{result.version}\n"
        f"更新内容：{result.changelog}\n\n"
        "发布信息校验成功，准备下载并验证完整性…"
    )
    
    q = queue.Queue()
    threading.Thread(
        target=download_and_install_update,
        args=(result.url, q),
        kwargs={
            "expected_sha256": result.sha256,
            "expected_size": result.size,
        },
        daemon=True,
    ).start()
    
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

DARK_MODE_JS = "() => document.documentElement.classList.add('dark')"


def gradio_launch_kwargs() -> dict[str, str]:
    """Return Gradio 6 launch-time styling options shared by both entry points."""
    return {"css": CUSTOM_CSS, "js": DARK_MODE_JS}

def manual_save_keys(gemini, nvidia, access, secret):
    save_env(gemini, nvidia, access, secret)
    return "✅ 密钥已成功保存到 .env 文件中"


def save_api_settings_from_existing_controls(
    gemini_key,
    nvidia_key,
    lovart_access,
    lovart_secret,
    gemini_base_url,
    gemini_model,
    nvidia_base_url,
    nvidia_model,
):
    """Save existing provider controls while Task 8 owns GPT Image UI inputs."""
    try:
        current = load_config()
        openai_image = current.get("openai_image", {}) if isinstance(current, dict) else {}
        image_generation = current.get("image_generation", {}) if isinstance(current, dict) else {}
        if not isinstance(openai_image, dict):
            openai_image = {}
        if not isinstance(image_generation, dict):
            image_generation = {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return f"API settings save failed: {exc}"
    return save_api_settings(
        gemini_key,
        nvidia_key,
        lovart_access,
        lovart_secret,
        None,
        gemini_base_url,
        gemini_model,
        nvidia_base_url,
        nvidia_model,
        openai_image.get("base_url", ""),
        openai_image.get("model", "gpt-image-2"),
        openai_image.get("resolution", "1K"),
        image_generation.get("support_provider", "lovart"),
        image_generation.get("detail_provider", "lovart"),
    )


def save_api_settings_from_ui(
    gemini_key,
    nvidia_key,
    lovart_access,
    lovart_secret,
    openai_image_key,
    gemini_base_url,
    gemini_model,
    nvidia_base_url,
    nvidia_model,
    openai_image_base_url,
    openai_image_model,
    openai_image_resolution,
    support_provider,
    detail_provider,
    clear_openai_image_key,
    merge_reference_images=None,
    *,
    config_path="config.yaml",
    env_path=".env",
):
    status = save_api_settings(
        gemini_key,
        nvidia_key,
        lovart_access,
        lovart_secret,
        openai_image_key,
        gemini_base_url,
        gemini_model,
        nvidia_base_url,
        nvidia_model,
        openai_image_base_url,
        openai_image_model,
        openai_image_resolution,
        support_provider,
        detail_provider,
        clear_openai_image_key=clear_openai_image_key,
        merge_reference_images=merge_reference_images,
        config_path=config_path,
        env_path=env_path,
    )
    clear_update = False if status == API_SETTINGS_SAVE_SUCCESS else gr.skip()
    return status, openai_image_key_status(env_path), clear_update


def run_process_from_ui(
    excel_file,
    custom_output_dir,
    prompt_source,
    prompt_model,
    lovart_mode,
    lovart_image_model,
    gemini_base_url,
    nvidia_base_url,
    gemini_key,
    nvidia_key,
    lovart_access,
    lovart_secret,
    openai_image_key,
    openai_image_base_url,
    openai_image_model,
    openai_image_resolution,
    support_provider,
    detail_provider,
    clear_openai_image_key,
    merge_reference_images=None,
    *,
    config_path="config.yaml",
    env_path=".env",
):
    process_updates = run_process(
        excel_file,
        custom_output_dir,
        prompt_source,
        prompt_model,
        lovart_mode,
        lovart_image_model,
        gemini_base_url,
        nvidia_base_url,
        gemini_key,
        nvidia_key,
        lovart_access,
        lovart_secret,
        openai_image_key,
        openai_image_base_url,
        openai_image_model,
        openai_image_resolution,
        support_provider,
        detail_provider,
        clear_openai_image_key,
        merge_reference_images,
        config_path=config_path,
        env_path=env_path,
    )
    synchronized = False
    for dashboard in process_updates:
        if not synchronized and str(dashboard).startswith("Starting"):
            synchronized = True
            yield dashboard, openai_image_key_status(env_path), False
        else:
            yield dashboard, gr.skip(), gr.skip()


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
    gemini_config = config.get("gemini_api", {})
    nvidia_config = config.get("nvidia_api", {})
    openai_image_config = config.get("openai_image", {}) or {}
    if not isinstance(openai_image_config, dict):
        openai_image_config = {}
    configured_merge_references = openai_image_config.get("merge_reference_images")
    merge_reference_images_value = resolve_merge_reference_images(
        configured_merge_references,
        openai_image_config.get("base_url", ""),
    )
    image_generation_config = config.get("image_generation", {}) or {}
    if not isinstance(image_generation_config, dict):
        image_generation_config = {}
    gemini_saved_model = _configured_provider_model(config, "gemini_api")
    nvidia_saved_model = _configured_provider_model(config, "nvidia")
    gemini_base_url_value = gemini_config.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
    nvidia_base_url_value = nvidia_config.get("base_url", "https://integrate.api.nvidia.com/v1")
    allow_regular_chat_fallback_value = (
        config.get("gemini", {}).get("allow_regular_chat_fallback") is True
    )
    lovart_config = config.get("lovart", {}) or {}
    failed_retry_policy = FailedRetryPolicy.from_config(lovart_config)
    saved_lovart_catalog = normalize_lovart_model_catalog(
        lovart_config.get("unlimited_model_catalog") or []
    )
    saved_lovart_order = [
        model
        for model in (lovart_config.get("unlimited_models") or [])
        if isinstance(model, str) and model in LOVART_IMAGE_MODELS and model != "auto"
    ]
    if not saved_lovart_catalog and saved_lovart_order:
        saved_lovart_catalog = [
            {
                "model": model,
                "tool_name": LOVART_IMAGE_MODELS[model],
                "label": LOVART_MODEL_LABELS.get(model, model),
                "restriction": "",
            }
            for model in saved_lovart_order
        ]
    saved_lovart_choices = lovart_model_choices(saved_lovart_catalog, saved_lovart_order)

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

    def probe_provider_controls(
        provider, api_key, base_url, current_model, prompt_source_value, catalog
    ):
        status, choices, selected, updated_catalog = probe_provider_model(
            provider, api_key, base_url, current_model, catalog
        )
        provider_update = gr.update(choices=choices, value=selected)
        active_source = "gemini_api" if provider == "gemini" else "nvidia"
        workspace_update = (
            gr.update(choices=choices, value=selected, interactive=True)
            if prompt_source_value == active_source
            else gr.skip()
        )
        return status, provider_update, workspace_update, updated_catalog

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

    def detect_lovart_controls(access_key, secret_key):
        status, catalog = detect_lovart_unlimited_models(access_key, secret_key)
        if not catalog:
            return status, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        order = [item["model"] for item in catalog]
        choices = lovart_model_choices(catalog, order)
        return (
            status,
            gr.update(choices=choices, value=order),
            gr.update(choices=choices, value=order[0]),
            catalog,
            order,
            format_lovart_model_order(order, order, catalog),
        )

    def move_lovart_controls(model, selected, catalog, order, direction):
        updated_order = move_lovart_model(model, order, direction)
        choices = lovart_model_choices(catalog, updated_order)
        selected_set = set(selected or [])
        updated_selected = [item for item in updated_order if item in selected_set]
        return (
            gr.update(choices=choices, value=updated_selected),
            gr.update(choices=choices, value=model),
            updated_order,
            format_lovart_model_order(updated_selected, updated_order, catalog),
        )

    with gr.Blocks(title="Lovart Image Automation WebUI") as demo:
        gemini_catalog_state = gr.State([])
        nvidia_catalog_state = gr.State([])
        lovart_catalog_state = gr.State(saved_lovart_catalog)
        lovart_order_state = gr.State(saved_lovart_order)
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
                    with gr.Row():
                        support_provider = gr.Dropdown(
                            choices=[
                                ("Lovart", "lovart"),
                                ("OpenAI-compatible GPT Image", "openai_image"),
                            ],
                            value=image_generation_config.get("support_provider", "lovart"),
                            label="白底图和场景图来源",
                            elem_classes=["glass-input", "pill-dropdown"],
                        )
                        detail_provider = gr.Dropdown(
                            choices=[
                                ("Lovart", "lovart"),
                                ("OpenAI-compatible GPT Image", "openai_image"),
                            ],
                            value=image_generation_config.get("detail_provider", "lovart"),
                            label="最终套图来源",
                            elem_classes=["glass-input", "pill-dropdown"],
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

                    gr.Markdown("### Gemini 浏览器账户")
                    gemini_login_status = gr.Markdown(
                        "Gemini 登录浏览器尚未打开。",
                        elem_id="gemini-login-status",
                    )
                    with gr.Row():
                        open_gemini_login_btn = gr.Button("打开 Gemini 登录浏览器")
                        check_gemini_login_btn = gr.Button("检查登录并关闭浏览器")
                        gemini_health_check_btn = gr.Button(
                            "Gemini 一键完整体检",
                            variant="secondary",
                        )
                    gemini_health_check_result = gr.Markdown(
                        "尚未运行 Gemini 完整体检。",
                        elem_id="gemini-health-check-result",
                    )
                    allow_regular_chat_fallback = gr.Checkbox(
                        label="临时聊天不可用时，允许使用普通聊天继续",
                        value=allow_regular_chat_fallback_value,
                    )
                    save_gemini_browser_btn = gr.Button("保存 Gemini 浏览器设置")
                    gemini_browser_save_status = gr.Markdown("")
                    save_gemini_browser_btn.click(
                        fn=save_gemini_browser_settings,
                        inputs=[allow_regular_chat_fallback],
                        outputs=gemini_browser_save_status,
                    )

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

                    gr.Markdown("### OpenAI-compatible GPT Image")
                    gr.Markdown(
                        "保存设置不会调用图像 API。只有点击下方的明确付费测试按钮才会发起真实图像编辑请求。"
                    )
                    openai_image_key = gr.Textbox(
                        label="GPT Image API 密钥",
                        value="",
                        type="password",
                        placeholder="留空保留已保存密钥",
                    )
                    openai_image_key_indicator = gr.Markdown(openai_image_key_status())
                    clear_openai_image_key = gr.Checkbox(
                        label="清除已保存 GPT Image 密钥",
                        value=False,
                    )
                    openai_image_base_url = gr.Textbox(
                        label="GPT Image API 地址",
                        value=openai_image_config.get("base_url", ""),
                        placeholder="例如：https://api.openai.com/v1",
                        info="可以带或不带 /v1，请按服务商提供的完整 Base URL 填写",
                    )
                    with gr.Row():
                        openai_image_model = gr.Textbox(
                            label="GPT Image 模型",
                            value=openai_image_config.get("model", "gpt-image-2"),
                        )
                        openai_image_resolution = gr.Radio(
                            choices=["1K", "2K", "4K"],
                            value=openai_image_config.get("resolution", "1K"),
                            label="GPT Image 分辨率",
                        )
                    merge_reference_images = gr.Checkbox(
                        label="将多张参考图合并为一张上传",
                        value=merge_reference_images_value,
                        info=(
                            "兼容只接受单个 image 文件的代理。开启后会先在本地生成参考拼图；"
                            "HAPI 地址默认开启，其他 OpenAI 兼容地址默认关闭。"
                        ),
                    )
                    openai_image_test_status = gr.Markdown(
                        "点击下方按钮后，这里会立即显示生成状态和进度。"
                    )
                    openai_image_test_btn = gr.Button(
                        OPENAI_IMAGE_TEST_BUTTON_LABEL,
                        variant="stop",
                    )
                    openai_image_test_start = openai_image_test_btn.click(
                        fn=begin_openai_image_test,
                        outputs=[openai_image_test_status, openai_image_test_btn],
                        queue=False,
                        api_name=False,
                    )
                    openai_image_test_run = openai_image_test_start.then(
                        fn=test_openai_image_edit,
                        inputs=[
                            openai_image_key,
                            openai_image_base_url,
                            openai_image_model,
                            openai_image_resolution,
                            merge_reference_images,
                        ],
                        outputs=openai_image_test_status,
                        api_name="test_openai_image_edit",
                    )
                    openai_image_test_run.then(
                        fn=reset_openai_image_test_button,
                        outputs=openai_image_test_btn,
                        queue=False,
                        api_name=False,
                    )
                    openai_image_test_run.failure(
                        fn=reset_openai_image_test_button,
                        outputs=openai_image_test_btn,
                        queue=False,
                        api_name=False,
                    )

                    lovart_access = gr.Textbox(label="LOVART_ACCESS_KEY", value=get_env("LOVART_ACCESS_KEY"), type="password")
                    lovart_secret = gr.Textbox(label="LOVART_SECRET_KEY", value=get_env("LOVART_SECRET_KEY"), type="password")

                    gr.Markdown("### Lovart 当前账号无限模型")
                    detect_lovart_models_btn = gr.Button(
                        "检测当前账号无限模型",
                        variant="secondary",
                    )
                    lovart_unlimited_models = gr.CheckboxGroup(
                        choices=saved_lovart_choices,
                        value=saved_lovart_order,
                        label="启用的无限模型",
                        show_select_all=True,
                    )
                    with gr.Row():
                        lovart_model_to_move = gr.Dropdown(
                            choices=saved_lovart_choices,
                            value=saved_lovart_order[0] if saved_lovart_order else None,
                            label="调整顺序",
                            scale=8,
                        )
                        move_lovart_up_btn = gr.Button("↑", min_width=48, scale=1)
                        move_lovart_down_btn = gr.Button("↓", min_width=48, scale=1)
                    lovart_model_order_preview = gr.Textbox(
                        value=format_lovart_model_order(
                            saved_lovart_order,
                            saved_lovart_order,
                            saved_lovart_catalog,
                        ),
                        label="任务尝试顺序",
                        interactive=False,
                    )
                    save_lovart_models_btn = gr.Button("保存无限模型及顺序")
                    lovart_model_status = gr.Textbox(
                        label="检测与保存状态",
                        interactive=False,
                        lines=2,
                    )

                    detect_lovart_models_btn.click(
                        fn=detect_lovart_controls,
                        inputs=[lovart_access, lovart_secret],
                        outputs=[
                            lovart_model_status,
                            lovart_unlimited_models,
                            lovart_model_to_move,
                            lovart_catalog_state,
                            lovart_order_state,
                            lovart_model_order_preview,
                        ],
                        api_name="detect_lovart_unlimited_models",
                    )
                    move_lovart_up_btn.click(
                        fn=lambda model, selected, catalog, order: move_lovart_controls(
                            model, selected, catalog, order, -1
                        ),
                        inputs=[
                            lovart_model_to_move,
                            lovart_unlimited_models,
                            lovart_catalog_state,
                            lovart_order_state,
                        ],
                        outputs=[
                            lovart_unlimited_models,
                            lovart_model_to_move,
                            lovart_order_state,
                            lovart_model_order_preview,
                        ],
                        api_name=False,
                    )
                    move_lovart_down_btn.click(
                        fn=lambda model, selected, catalog, order: move_lovart_controls(
                            model, selected, catalog, order, 1
                        ),
                        inputs=[
                            lovart_model_to_move,
                            lovart_unlimited_models,
                            lovart_catalog_state,
                            lovart_order_state,
                        ],
                        outputs=[
                            lovart_unlimited_models,
                            lovart_model_to_move,
                            lovart_order_state,
                            lovart_model_order_preview,
                        ],
                        api_name=False,
                    )
                    lovart_unlimited_models.change(
                        fn=format_lovart_model_order,
                        inputs=[
                            lovart_unlimited_models,
                            lovart_order_state,
                            lovart_catalog_state,
                        ],
                        outputs=lovart_model_order_preview,
                        api_name=False,
                    )
                    save_lovart_models_btn.click(
                        fn=save_lovart_unlimited_models,
                        inputs=[
                            lovart_unlimited_models,
                            lovart_order_state,
                            lovart_catalog_state,
                        ],
                        outputs=lovart_model_status,
                        api_name="save_lovart_unlimited_models",
                    )

                    gr.Markdown("### 失败任务补偿重试")
                    failed_retry_mode = gr.Radio(
                        choices=[
                            ("关闭", RETRY_MODE_OFF),
                            ("有限次数", RETRY_MODE_FINITE),
                            ("无限重试", RETRY_MODE_INFINITE),
                        ],
                        value=failed_retry_policy.mode,
                        label="重试模式",
                    )
                    with gr.Row():
                        failed_retry_rounds = gr.Number(
                            value=failed_retry_policy.rounds,
                            label="最多补偿轮次（不含首次）",
                            minimum=1,
                            precision=0,
                            interactive=failed_retry_policy.mode == RETRY_MODE_FINITE,
                        )
                        failed_retry_delay = gr.Number(
                            value=failed_retry_policy.delay,
                            label="每轮重试间隔（秒）",
                            minimum=0,
                        )
                    failed_retry_error_types = gr.CheckboxGroup(
                        choices=list(RETRY_ERROR_TYPE_CHOICES),
                        value=list(failed_retry_policy.error_types),
                        label="允许重试的错误类型",
                    )
                    save_failed_retry_btn = gr.Button("保存失败任务重试设置")
                    failed_retry_save_status = gr.Markdown("")
                    failed_retry_mode.change(
                        fn=failed_retry_rounds_update,
                        inputs=failed_retry_mode,
                        outputs=failed_retry_rounds,
                        api_name=False,
                    )
                    save_failed_retry_btn.click(
                        fn=save_failed_retry_settings,
                        inputs=[
                            failed_retry_mode,
                            failed_retry_rounds,
                            failed_retry_delay,
                            failed_retry_error_types,
                        ],
                        outputs=failed_retry_save_status,
                    )
                    
                    save_keys_btn = gr.Button("💾 保存密钥、API 地址和模型", variant="primary")
                    save_status = gr.Markdown("")
                    
                    key_inputs = [
                        gemini_key,
                        nvidia_key,
                        lovart_access,
                        lovart_secret,
                        openai_image_key,
                    ]
                    save_keys_btn.click(
                        fn=save_api_settings_from_ui,
                        inputs=[
                            *key_inputs,
                            gemini_base_url,
                            gemini_model,
                            nvidia_base_url,
                            nvidia_model,
                            openai_image_base_url,
                            openai_image_model,
                            openai_image_resolution,
                            support_provider,
                            detail_provider,
                            clear_openai_image_key,
                            merge_reference_images,
                        ],
                        outputs=[
                            save_status,
                            openai_image_key_indicator,
                            clear_openai_image_key,
                        ],
                        api_name="save_api_settings",
                    )
                    open_gemini_login_btn.click(
                        fn=open_gemini_login_browser,
                        outputs=gemini_login_status,
                        api_name="open_gemini_login_browser",
                    )
                    check_gemini_login_btn.click(
                        fn=check_gemini_login_and_close,
                        outputs=gemini_login_status,
                        api_name="check_gemini_login_and_close",
                    )
                    health_check_start = gemini_health_check_btn.click(
                        fn=lambda: gr.update(interactive=False),
                        outputs=gemini_health_check_btn,
                        queue=False,
                        api_name=False,
                    )
                    health_check_run = health_check_start.then(
                        fn=ui_run_gemini_health_check,
                        outputs=gemini_health_check_result,
                        api_name="run_gemini_health_check",
                    )
                    health_check_run.then(
                        fn=lambda: gr.update(interactive=True),
                        outputs=gemini_health_check_btn,
                        queue=False,
                        api_name=False,
                    )
                    health_check_run.failure(
                        fn=lambda: gr.update(interactive=True),
                        outputs=gemini_health_check_btn,
                        queue=False,
                        api_name=False,
                    )

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
                        fn=lambda key, url, model, source, catalog: probe_provider_controls(
                            "gemini", key, url, model, source, catalog
                        ),
                        inputs=[
                            gemini_key, gemini_base_url, gemini_model,
                            prompt_source, gemini_catalog_state,
                        ],
                        outputs=[
                            gemini_status, gemini_model, prompt_model, gemini_catalog_state,
                        ],
                        api_name="probe_gemini_model",
                    )
                    nvidia_test_btn.click(
                        fn=lambda key, url, model, source, catalog: probe_provider_controls(
                            "nvidia", key, url, model, source, catalog
                        ),
                        inputs=[
                            nvidia_key, nvidia_base_url, nvidia_model,
                            prompt_source, nvidia_catalog_state,
                        ],
                        outputs=[
                            nvidia_status, nvidia_model, prompt_model, nvidia_catalog_state,
                        ],
                        api_name="probe_nvidia_model",
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
                    prompt_required_sections = gr.Textbox(
                        value="\n".join(prompt_form_values[2]),
                        label="每屏必须包含的内容",
                        lines=5,
                        interactive=True,
                        info="每行或逗号分隔一项，可输入任意自定义内容",
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
            fn=run_process_from_ui,
            inputs=[
                excel_file, custom_output_dir, prompt_source, prompt_model, lovart_mode, lovart_image_model,
                gemini_base_url, nvidia_base_url,
                gemini_key, nvidia_key, lovart_access, lovart_secret,
                openai_image_key, openai_image_base_url, openai_image_model,
                openai_image_resolution, support_provider, detail_provider,
                clear_openai_image_key,
                merge_reference_images,
            ],
            outputs=[
                progress_dashboard,
                openai_image_key_indicator,
                clear_openai_image_key,
            ],
            api_name="run_process",
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
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        allowed_paths=[output_dir],
        **gradio_launch_kwargs(),
    )
