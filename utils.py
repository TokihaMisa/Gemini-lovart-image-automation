import csv
import json
import logging
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

from prompt_settings import locked_rules_text, normalize_prompt_settings


def get_resource_path(filename: str) -> Path:
    """Resolve file path, prioritizing CWD, then falling back to PyInstaller _internal MEIPASS."""
    cwd_path = Path(filename)
    if cwd_path.exists():
        return cwd_path
    if hasattr(sys, '_MEIPASS'):
        mei_path = Path(sys._MEIPASS) / filename
        if mei_path.exists():
            return mei_path
    return cwd_path


def load_config(path: str = "config.yaml") -> dict:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE lines into os.environ, treating .env as project-local truth."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_file = Path(log_dir) / f"run_{timestamp}.log"

    logger = logging.getLogger("image_automation")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def col_letter_to_index(letter: str) -> int:
    """Convert Excel column letter to 0-based index. A->0, B->1, etc."""
    result = 0
    for ch in letter.upper():
        result = result * 26 + (ord(ch) - ord("A") + 1)
    return result - 1


def col_letter_to_openpyxl_idx(letter: str) -> int:
    """Convert Excel column letter to 1-based openpyxl column index."""
    return col_letter_to_index(letter) + 1


def get_output_dir() -> str:
    import os
    return os.environ.get("LOVART_OUTPUT_DIR", "output")


def ensure_output_dir(product_id: str, base_dir: str = None) -> Path:
    if base_dir is None:
        base_dir = get_output_dir()
    return product_output_dir(product_id, base_dir)


def product_output_dir(product_id: str, base_dir: str = None) -> Path:
    """Return the canonical output directory for a product, searching subdirectories if categorized."""
    if base_dir is None:
        base_dir = get_output_dir()
    base = Path(base_dir)
    direct = base / str(product_id)
    if direct.exists():
        return direct
    
    if base.exists():
        for sub in base.iterdir():
            if sub.is_dir() and sub.name in ["1_完全做好", "2_待确认", "3_处理中", "4_异常"]:
                candidate = sub / str(product_id)
                if candidate.exists():
                    return candidate

    direct.mkdir(parents=True, exist_ok=True)
    return direct


def env_or_config(config: dict, key: str, env_name: str, default: str = "") -> str:
    """Read a secret/config value with environment variables taking priority."""
    value = os.environ.get(env_name)
    if value:
        return value
    return str(config.get(key, default) or "")


def read_status(product_dir: str | Path) -> dict:
    path = Path(product_dir) / "status.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def update_status(product_dir: str | Path, stage: str, **fields) -> dict:
    """Merge a stage flag and fields into output/<product_id>/status.json."""
    path = Path(product_dir)
    path.mkdir(parents=True, exist_ok=True)
    status = read_status(path)
    status[stage] = True
    status.update(fields)
    status["updated_at"] = datetime.now().isoformat(timespec="seconds")
    target_path = path / "status.json"
    temp_path = path / "status.json.tmp"
    try:
        temp_path.write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(target_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return status


def is_product_completed(product_dir: str | Path) -> bool:
    return bool(read_status(product_dir).get("lovart_done"))


RESULT_FIELDNAMES = ["product_id", "product_name", "status", "project_url", "error", "used_model"]
CSV_READ_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "mbcs")


def append_result(
    results_path: str | Path,
    product_id: str,
    product_name: str,
    project_url: str = "",
    status: str = "success",
    error: str = "",
    used_model: str = "",
) -> None:
    """Upsert one product outcome to results.csv using real CSV escaping."""
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_result_rows(path)
    new_row = {
        "product_id": product_id,
        "product_name": product_name,
        "status": status,
        "project_url": project_url,
        "error": error,
        "used_model": used_model,
    }

    by_id = {}
    order = []
    for row in rows:
        existing_id = row.get("product_id", "")
        if not existing_id:
            continue
        if existing_id not in by_id:
            order.append(existing_id)
        by_id[existing_id] = {field: row.get(field, "") for field in RESULT_FIELDNAMES}

    if product_id not in by_id:
        order.append(product_id)
    by_id[product_id] = new_row

    temp_path = path.with_suffix(".csv.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDNAMES)
            writer.writeheader()
            for key in order:
                writer.writerow(by_id[key])
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _read_result_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    _upgrade_results_csv_header(path)
    fieldnames, rows = _read_csv_dict_rows_with_fallback(path)
    if fieldnames != RESULT_FIELDNAMES:
        return []
    return rows


def _upgrade_results_csv_header(path: Path) -> None:
    """Upgrade legacy results.csv files before appending new rows."""
    rows = _read_csv_rows_with_fallback(path)
    if not rows or rows[0] == RESULT_FIELDNAMES:
        return
    legacy_header = rows[0]
    if legacy_header not in (
        ["product_id", "product_name", "project_url"],
        ["product_id", "product_name", "status", "project_url", "error"],
    ):
        return

    upgraded_rows = [RESULT_FIELDNAMES]
    for row in rows[1:]:
        product_id = row[0] if len(row) > 0 else ""
        product_name = row[1] if len(row) > 1 else ""
        if legacy_header == ["product_id", "product_name", "project_url"]:
            project_url = row[2] if len(row) > 2 else ""
            upgraded_rows.append([product_id, product_name, "success", project_url, "", ""])
        else:
            status = row[2] if len(row) > 2 else ""
            project_url = row[3] if len(row) > 3 else ""
            error = row[4] if len(row) > 4 else ""
            upgraded_rows.append([product_id, product_name, status, project_url, error, ""])

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(upgraded_rows)


def _read_csv_rows_with_fallback(path: Path) -> list[list[str]]:
    last_error = None
    for encoding in CSV_READ_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                return list(csv.reader(fh))
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except LookupError:
            continue
    if last_error:
        raise last_error
    return []


def _read_csv_dict_rows_with_fallback(path: Path) -> tuple[list[str], list[dict]]:
    last_error = None
    for encoding in CSV_READ_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as fh:
                reader = csv.DictReader(fh)
                fieldnames = reader.fieldnames or []
                rows = list(reader)
            return fieldnames, rows
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except LookupError:
            continue
    if last_error:
        raise last_error
    return [], []


def split_image_roles(image_paths: list[str]) -> dict:
    """Split workbook images by position into product/accessory/dimension/reference roles."""
    return {
        "product_image": image_paths[0] if len(image_paths) >= 1 else "",
        "accessory_image": image_paths[1] if len(image_paths) >= 2 else "",
        "dimension_image": image_paths[2] if len(image_paths) >= 3 else "",
        "reference_images": [path for path in image_paths[3:] if path] if len(image_paths) >= 4 else [],
    }


def build_final_lovart_images(
    white_image: str,
    scene_image: str,
    accessory_image: str = "",
    dimension_image: str = "",
    reference_sheet: str = "",
) -> list[str]:
    """Build final Lovart upload order, keeping the merged reference sheet last."""
    images = [white_image, scene_image]
    if accessory_image:
        images.append(accessory_image)
    if dimension_image:
        images.append(dimension_image)
    if reference_sheet:
        images.append(reference_sheet)
    return [path for path in images if path]


def merge_reference_images(reference_paths: list[str], output_path: str | Path, tile_size: int = 512) -> str:
    """Merge reference images into one contact sheet for Gemini/Lovart style reference."""
    if not reference_paths:
        return ""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required to merge reference images. Run: pip install Pillow==12.0.0") from exc

    images = []
    for path in reference_paths:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_size, tile_size), "white")
            x = (tile_size - img.width) // 2
            y = (tile_size - img.height) // 2
            tile.paste(img, (x, y))
            images.append(tile)

    columns = max(1, math.ceil(math.sqrt(len(images))))
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    for idx, tile in enumerate(images):
        x = (idx % columns) * tile_size
        y = (idx // columns) * tile_size
        sheet.paste(tile, (x, y))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=95)
    return str(output)


def image_size_instruction(image_size: str = "") -> str:
    """Return an image-size instruction only when the workbook provides one."""
    cleaned = str(image_size or "").strip()
    return f"图片尺寸/比例: {cleaned}\n" if cleaned else ""


def _effective_image_size_instruction(image_size: str, settings: dict) -> str:
    cleaned = str(image_size or "").strip()
    if cleaned:
        return f"图片尺寸/比例: {cleaned}\n"
    fallback = str(settings["missing_image_size_policy"] or "").strip()
    return f"图片尺寸/比例: {fallback}\n" if fallback else ""


def build_white_background_prompt(image_size: str = "", prompt_settings=None) -> str:
    settings = normalize_prompt_settings(prompt_settings)
    return (
        f"{settings['white_background_requirements']}\n"
        f"图片画质: {settings['image_quality']}\n"
        f"{_effective_image_size_instruction(image_size, settings)}"
    )


def build_scene_prompt(image_size: str = "", prompt_settings=None) -> str:
    settings = normalize_prompt_settings(prompt_settings)
    return (
        f"{settings['scene_requirements']}\n"
        f"图片画质: {settings['image_quality']}\n"
        f"{_effective_image_size_instruction(image_size, settings)}"
    )


WHITE_BACKGROUND_PROMPT = build_white_background_prompt()
SCENE_PROMPT = build_scene_prompt()


def build_design_prompt(
    product_name_cn: str,
    language: str,
    selling_points: str,
    image_size: str = "",
    prompt_settings=None,
) -> str:
    """Build the product-specific Gemini prompt in UTF-8 Chinese."""
    settings = normalize_prompt_settings(prompt_settings)
    output_language = str(language or "").strip() or str(settings["default_language"])
    sections = "、".join(settings["required_sections"])
    question_rule = "允许在信息确实不足时提出一个必要问题。" if settings["allow_questions"] else "请不要反问，直接根据现有信息生成最优提示词。"
    extra = str(settings["extra_requirements"] or "").strip()
    extra_block = f"\n额外要求：\n{extra}\n" if extra else ""
    return (
        f"上传图片是我的{product_name_cn}产品。\n"
        "【角色设定】你是一名资深电商设计师，擅长平面设计、信息层级和文字排版。\n"
        f"请设计一套包含{settings['detail_page_count']}屏的电商详情页；一屏对应一张详情成品图，不是多套设计版本。\n"
        "你当前只负责输出可交给 Lovart 的逐屏文字设计提示词，不要直接生成图片。\n"
        f"整体风格：{settings['design_style']}\n"
        f"每屏必须包含：{sections}\n"
        f"图片画质：{settings['image_quality']}\n"
        f"Logo 规则：{settings['logo_policy']}\n"
        f"文案要求：{settings['copy_style']}；详细程度：{settings['copy_detail_level']}\n"
        f"产品还原：{settings['product_fidelity']}\n"
        f"{_effective_image_size_instruction(image_size, settings)}"
        f"图片语言：{output_language}\n"
        f"{question_rule}\n"
        f"产品信息/卖点：\n{selling_points}\n"
        f"{extra_block}\n"
        f"【锁定规则】\n{locked_rules_text()}\n"
    )


def build_lovart_image_note(
    has_reference_sheet: bool,
    has_accessory_image: bool,
    has_dimension_image: bool,
    reference_images_are_product: bool = False,
) -> str:
    """Describe uploaded image roles for the final Lovart detail-page generation."""
    parts = [
        "上传图片说明：",
        "图1是白底产品图，图2是产品场景图，二者都是我的商品主体参考。",
    ]
    next_index = 3
    if has_accessory_image:
        parts.append(f"图{next_index}是配件图，属于商品组成部分或包装配件参考。")
        next_index += 1
    if has_dimension_image:
        parts.append(f"图{next_index}是尺寸图，属于商品真实尺寸和结构信息参考。")
        next_index += 1
    if has_reference_sheet:
        if reference_images_are_product:
            parts.append(
                f"最后一张图（图{next_index}）才是合并参考图，里面的参考图是同一个产品、同一个规格/颜色/款式；"
                "可以直接当作我的商品参考，用来理解外形、结构、材质、颜色和其他角度的样子。"
            )
            parts.append("所有上传图片都属于我的商品或商品信息，必须优先保持真实形态，不要把它改成其他产品。")
        else:
            parts.append(
                f"最后一张图（图{next_index}）才是合并参考图，只参考风格、排版氛围和视觉调性；"
                "不要把参考图里的产品当成我的产品，不要参考里面产品的外形、结构、颜色或款式。"
            )
            parts.append("除最后一张参考图以外，其余上传图片都属于我的商品或商品信息，必须优先保持真实形态。")
    else:
        parts.append("本次没有单独的风格参考图，所有上传图片都属于我的商品或商品信息。")
    return "\n".join(parts) + "\n\n"


def build_lovart_prompt(
    product_name_cn: str,
    language: str,
    selling_points: str,
    generated_prompt: str,
    image_note: str = "",
    image_size: str = "",
    prompt_settings=None,
) -> str:
    """Prepend product guardrails before sending Gemini's generated prompt to Lovart."""
    settings = normalize_prompt_settings(prompt_settings)
    output_language = str(language or "").strip() or str(settings["default_language"])
    sections = "、".join(settings["required_sections"])
    extra = str(settings["extra_requirements"] or "").strip()
    extra_line = f"- 额外要求：{extra}\n" if extra else ""
    prefix = (
        f"我的产品是：{product_name_cn}\n\n"
        f"{image_note}"
        "我上传的图片是产品真实外形、结构、颜色、材质和比例的强参考。创意场景、背景和排版可以重新设计，"
        "但产品主体必须严格贴近参考图片，不要改变真实形态，不要幻想出不存在的部件、颜色或结构。\n\n"
        "【角色设定】你是一名资深电商视觉设计师，擅长商品详情页、信息层级、卖点提炼和图片生成提示词设计。\n\n"
        f"我需要你为这个产品设计一套包含{settings['detail_page_count']}屏的完整电商详情页；一屏一张最终图片。\n"
        "设计要求：\n"
        f"- 整体风格：{settings['design_style']}\n"
        f"- 每屏必须包含：{sections}\n"
        f"- 图片画质：{settings['image_quality']}\n"
        f"- Logo 规则：{settings['logo_policy']}\n"
        f"- 文案要求：{settings['copy_style']}；详细程度：{settings['copy_detail_level']}\n"
        f"- 产品还原：{settings['product_fidelity']}\n"
        f"- {_effective_image_size_instruction(image_size, settings)}"
        f"- 图片语言：{output_language}\n"
        f"{extra_line}\n"
        f"产品信息/卖点：\n{selling_points}\n\n"
        f"【锁定规则】\n{locked_rules_text()}\n\n"
        "以下是 Gemini 已生成的详细提示词，请在此基础上执行：\n\n"
    )
    return f"{prefix}{generated_prompt.strip()}\n"


def build_lovart_confirmation_prompt(
    product_name_cn: str,
    language: str,
    selling_points: str,
    confirmation_text: str,
    confirmation_payload,
    project_id: str,
    thread_id: str,
    round_index: int,
    max_auto_confirm_credits: int,
    lovart_mode: str,
) -> str:
    """Build a strict Gemini prompt for deciding whether to confirm a Lovart gate."""
    payload_json = json.dumps(confirmation_payload, ensure_ascii=False, indent=2)
    return (
        "你需要判断 Lovart 返回的确认请求是否应该继续确认。\n"
        "请只根据下面的信息判断，不要重新设计图片，不要输出长篇解释。\n\n"
        "重要提醒：Lovart 返回的确认内容不一定是消耗 credits；在 unlimited 模式下，它也可能只是排队、继续生成、工具调用、选项选择或普通流程确认。\n\n"
        "决策规则：\n"
        "- 如果这是为了继续生成当前产品电商详情页图片的正常流程确认，请选择 CONFIRM。\n"
        "- 如果当前是 unlimited 模式，且确认内容只是继续排队/继续生成/允许调用生成工具，不要因为出现“确认”就误判为付费消耗。\n"
        "- 只有当确认内容明确要求异常高成本、超出 credits 上限、账号/权限/安全风险、删除/覆盖项目、与当前产品无关，或无法判断时，才选择 STOP。\n"
        "- 如果返回内容里有多个选项，请选择最适合继续生成当前图片任务的选项；没有选项时只判断是否确认。\n\n"
        "必须严格输出 JSON，不要使用 Markdown：\n"
        '{"decision":"CONFIRM或STOP","reason":"一句话说明","message_to_lovart":"如果需要发给Lovart的简短选择/说明，没有就留空"}\n\n'
        f"当前产品：{product_name_cn}\n"
        f"图片语言：{language}\n"
        f"产品信息/卖点：\n{selling_points}\n\n"
        f"当前 Lovart 模式：{lovart_mode}\n"
        f"自动确认 credits 上限：{max_auto_confirm_credits}\n"
        f"Lovart project_id：{project_id}\n"
        f"Lovart thread_id：{thread_id}\n"
        f"确认轮次：{round_index}\n\n"
        f"Lovart 可读确认内容：\n{confirmation_text or '(无可读文字)'}\n\n"
        f"Lovart 原始确认 JSON：\n{payload_json}\n"
    )


def parse_lovart_confirmation_decision(text: str) -> dict:
    """Parse Gemini's confirmation decision into a small normalized dict."""
    raw = (text or "").strip()
    data = None
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.S)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    data = None

    if isinstance(data, dict):
        decision = str(data.get("decision", "")).strip().upper()
        reason = str(data.get("reason", "") or "").strip()
        message = str(data.get("message_to_lovart", "") or "").strip()
    else:
        upper = raw.upper()
        decision = "CONFIRM" if "CONFIRM" in upper and "STOP" not in upper else "STOP"
        reason = raw[:500]
        message = ""

    if decision not in {"CONFIRM", "STOP"}:
        decision = "STOP"
        if not reason:
            reason = "Gemini did not return a clear CONFIRM decision."

    return {
        "decision": decision,
        "reason": reason,
        "message_to_lovart": message,
        "raw_response": raw,
    }


def sanitize_filename(value: str, default: str = "item") -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value)).strip(" ._")
    return cleaned[:80] or default


def create_run_dir(base_dir: str | Path = "runs") -> Path:
    run_dir = Path(base_dir) / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_summary(run_dir: str | Path, rows: list[dict]) -> None:
    """Write per-run summary.json and summary.csv."""
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = [
        "product_id",
        "product_name",
        "status",
        "project_url",
        "gemini_chars",
        "artifact_count",
        "duration_seconds",
        "error",
    ]
    with (path / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _fix_status_paths(status_file_dir: Path, old_base: str, new_base: str):
    status_file = status_file_dir / "status.json"
    if not status_file.exists():
        return
    content = status_file.read_text(encoding="utf-8")
    old_base_fwd = old_base.replace("\\", "/")
    new_base_fwd = new_base.replace("\\", "/")
    old_base_bck = old_base.replace("/", "\\")
    new_base_bck = new_base.replace("/", "\\")
    
    content = content.replace(old_base_fwd, new_base_fwd)
    content = content.replace(old_base_bck, new_base_bck)
    status_file.write_text(content, encoding="utf-8")


def organize_output_folders(base_dir: str = None):
    if base_dir is None:
        base_dir = get_output_dir()
    """Move product directories into categorized subfolders based on completion status."""
    import shutil
    base = Path(base_dir)
    if not base.exists():
        return

    cat_done = base / "1_完全做好"
    cat_pending = base / "2_待确认"
    cat_processing = base / "3_处理中"
    cat_error = base / "4_异常"
    
    for sub in [cat_done, cat_pending, cat_processing, cat_error]:
        sub.mkdir(exist_ok=True)
        
    product_dirs = []
    category_names = ["1_完全做好", "2_待确认", "3_处理中", "4_异常"]
    
    for item in base.iterdir():
        if not item.is_dir() or item.name.startswith((".git", "__pycache__")):
            continue
        if item.name in category_names:
            for subitem in item.iterdir():
                if subitem.is_dir():
                    product_dirs.append(subitem)
        else:
            product_dirs.append(item)
            
    for pdir in product_dirs:
        status = read_status(pdir)
        if not status:
            continue
            
        is_done = bool(status.get("lovart_done"))
        is_pending = bool(status.get("needs_manual_action"))
        is_error = bool(status.get("lovart_project_invalid") or status.get("lovart_still_running"))
        
        if is_done:
            target_cat = cat_done
        elif is_pending:
            target_cat = cat_pending
        elif is_error:
            target_cat = cat_error
        else:
            target_cat = cat_processing
            
        target_path = target_cat / pdir.name
        
        if pdir.parent != target_cat:
            if target_path.exists():
                shutil.rmtree(target_path)
            try:
                pdir.rename(target_path)
            except OSError:
                shutil.move(str(pdir), str(target_path))
            _fix_status_paths(target_path, old_base=str(pdir), new_base=str(target_path))
