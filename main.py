import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from playwright.sync_api import sync_playwright
from rich.console import Console
from rich.panel import Panel

from excel_reader import read_products
from gemini_api import GeminiAPI
from gemini_bot import GeminiBot
from lovart_bot import LOVART_IMAGE_MODELS, LovartBot
from nvidia_api import NvidiaAPI, resolve_nvidia_model
from utils import (
    _read_csv_dict_rows_with_fallback,
    append_result,
    build_final_lovart_images,
    build_scene_prompt,
    build_white_background_prompt,
    build_lovart_prompt,
    build_lovart_image_note,
    create_run_dir,
    env_or_config,
    is_product_completed,
    load_config,
    merge_reference_images,
    product_output_dir,
    read_status,
    setup_logging,
    split_image_roles,
    update_status,
    write_run_summary,
)

_shutdown_requested = False


def _on_sigint(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\nInterrupted. Finishing current product then exiting...")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Excel to Gemini to Lovart product image automation")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--prompt-source",
        choices=["ask", "gemini_api", "gemini_browser", "nvidia"],
        default="ask",
        help="Prompt generation source. Default asks interactively.",
    )
    parser.add_argument(
        "--gemini",
        choices=["ask", "api", "browser"],
        default=None,
        help="Backward-compatible alias for --prompt-source.",
    )
    parser.add_argument(
        "--nvidia-model",
        choices=["kimi"],
        default=None,
        help="NVIDIA API model choice when prompt source is nvidia.",
    )
    parser.add_argument(
        "--lovart",
        choices=["ask", "fast", "unlimited"],
        default="ask",
        help="Lovart generation mode. Default asks interactively.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N parsed products")
    parser.add_argument("--dry-run", action="store_true", help="Parse Excel and write run summary without Gemini/Lovart calls")
    parser.add_argument("--generate-template", action="store_true", help="Generate a standard Excel template")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Skip products already marked lovart_done")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Reprocess products even if status.json says done")
    parser.add_argument(
        "--lovart-image-model",
        default=None,
        help="Override lovart.image_model for this run. Supports comma-separated values.",
    )
    parser.add_argument(
        "--lovart-model-selection",
        choices=["prefer", "force"],
        default=None,
        help="Use preferred model hint or force Lovart to use only that image tool.",
    )
    parser.add_argument(
        "--lovart-reasoning",
        choices=["fast", "thinking"],
        default=None,
        help="Override Lovart chat reasoning mode.",
    )
    return parser.parse_args(argv)


def _apply_lovart_overrides(config: dict, args) -> None:
    lovart_cfg = config.setdefault("lovart", {})
    if args.lovart_image_model:
        lovart_cfg["image_model"] = args.lovart_image_model
    if args.lovart_model_selection:
        lovart_cfg["model_selection"] = args.lovart_model_selection
    if args.lovart_reasoning:
        lovart_cfg["reasoning_mode"] = args.lovart_reasoning


def _apply_prompt_source_aliases(args) -> None:
    if args.gemini and args.prompt_source == "ask":
        args.prompt_source = "gemini_api" if args.gemini == "api" else "gemini_browser"


def _ask_number(prompt: str, default: int, min_value: int, max_value: int) -> int:
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < min_value or value > max_value:
        return default
    return value


def _ask_numbers(prompt: str, default_values: list[int], min_value: int, max_value: int) -> list[int]:
    raw = input(prompt).strip()
    if not raw:
        return default_values
    selected = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if min_value <= value <= max_value and value not in selected:
            selected.append(value)
    return selected or default_values


def _choose_lovart_tool_options(config: dict, args) -> None:
    """Prompt for Lovart model/reasoning unless command-line overrides were provided."""
    lovart_cfg = config.setdefault("lovart", {})
    _apply_lovart_overrides(config, args)

    model_options = [
        ("auto", "Auto (Lovart chooses)"),
        ("gpt_image_2", "GPT Image 2"),
        ("nano_banana", "Nano Banana"),
        ("nano_banana_2", "Nano Banana 2"),
        ("nano_banana_pro", "Nano Banana Pro"),
        ("midjourney", "Midjourney"),
        ("seedream_v4", "Seedream 4"),
        ("seedream_v4_5", "Seedream 4.5"),
    ]

    if not args.lovart_image_model:
        current = str(lovart_cfg.get("image_model", "auto") or "auto")
        current_values = [item.strip() for item in current.split(",") if item.strip()]
        default_indexes = [
            i for i, (value, _) in enumerate(model_options, 1)
            if value in current_values
        ] or [1]
        print(f"\n{'=' * 50}")
        print("  Lovart image model:")
        for idx, (_, label) in enumerate(model_options, 1):
            marker = " (default)" if idx in default_indexes else ""
            print(f"    [{idx}] {label}{marker}")
        print(f"{'=' * 50}")
        selected = _ask_numbers(
            f"  Choose one or more, comma-separated (default={','.join(str(i) for i in default_indexes)}): ",
            default_indexes,
            1,
            len(model_options),
        )
        if 1 in selected and len(selected) > 1:
            selected = [idx for idx in selected if idx != 1]
        lovart_cfg["image_model"] = ",".join(model_options[idx - 1][0] for idx in selected)

    if not args.lovart_model_selection:
        current = str(lovart_cfg.get("model_selection", "prefer") or "prefer")
        default_idx = 2 if current == "force" else 1
        print(f"\n{'=' * 50}")
        print("  Lovart model selection:")
        print(f"    [1] Prefer (agent can still auto-plan){' (default)' if default_idx == 1 else ''}")
        print(f"    [2] Force  (only use selected image tool){' (default)' if default_idx == 2 else ''}")
        print(f"{'=' * 50}")
        selected = _ask_number(f"  Choose (1/2, default={default_idx}): ", default_idx, 1, 2)
        lovart_cfg["model_selection"] = "force" if selected == 2 else "prefer"

    if not args.lovart_reasoning:
        current = str(lovart_cfg.get("reasoning_mode", "fast") or "fast")
        default_idx = 2 if current == "thinking" else 1
        print(f"\n{'=' * 50}")
        print("  Lovart reasoning mode:")
        print(f"    [1] Fast{' (default)' if default_idx == 1 else ''}")
        print(f"    [2] Thinking{' (default)' if default_idx == 2 else ''}")
        print(f"{'=' * 50}")
        selected = _ask_number(f"  Choose (1/2, default={default_idx}): ", default_idx, 1, 2)
        lovart_cfg["reasoning_mode"] = "thinking" if selected == 2 else "fast"

    configured_models = [
        item.strip().lower().replace("-", "_")
        for item in str(lovart_cfg.get("image_model", "auto") or "auto").split(",")
        if item.strip()
    ]
    if not configured_models or any(model not in LOVART_IMAGE_MODELS for model in configured_models):
        lovart_cfg["image_model"] = "auto"


def _choose_prompt_source(config: dict, args) -> str:
    _apply_prompt_source_aliases(args)
    if args.nvidia_model:
        config.setdefault("nvidia_api", {})["model_choice"] = args.nvidia_model
    if args.prompt_source != "ask":
        return args.prompt_source

    while True:
        print(f"\n{'=' * 50}")
        print("  Prompt generation source:")
        print("    [1] Gemini Browser  (Playwright, reuses Chrome profile)")
        print("    [2] Gemini API      (direct API)")
        print("    [3] NVIDIA API      (Kimi, supports product images)")
        print(f"{'=' * 50}")
        selected = _ask_number("  Choose (1/2/3, default=2): ", 2, 1, 3)
        if selected == 1:
            return "gemini_browser"
        if selected == 2:
            if env_or_config(config.get("gemini_api", {}), "api_key", "GEMINI_API_KEY"):
                return "gemini_api"
            print("\n  GEMINI_API_KEY is not set. Choose Gemini Browser, fill .env, or choose another API source.")
            continue

        config.setdefault("nvidia_api", {})["model_choice"] = "kimi"
        if env_or_config(config.get("nvidia_api", {}), "api_key", "NVIDIA_API_KEY"):
            return "nvidia"
        print("\n  NVIDIA_API_KEY is not set. Choose Gemini Browser, fill .env, or choose another API source.")


def _choose_lovart_mode() -> bool:
    print(f"\n{'=' * 50}")
    print("  Lovart generation mode:")
    print("    [1] Fast      (uses credits, no queue)")
    print("    [2] Unlimited (free, may queue)")
    print(f"{'=' * 50}")
    return (input("  Choose (1/2, default=2): ").strip() or "2") == "1"


def _resolve_lovart_mode(choice: str) -> bool:
    if choice == "ask":
        return _choose_lovart_mode()
    return choice == "fast"


def _record_success(product, result: dict) -> str:
    project_id = result.get("project_id", "")
    project_url = f"https://www.lovart.ai/canvas?projectId={project_id}" if project_id else ""
    from utils import get_output_dir
    append_result(f"{get_output_dir()}/results.csv", product.id, product.name_cn, project_url, status="success", used_model=result.get("used_model", ""))
    return project_url


def _record_failure(product, status: str, error: str = "", project_url: str = "") -> None:
    from utils import get_output_dir
    append_result(f"{get_output_dir()}/results.csv", product.id, product.name_cn, project_url, status=status, error=error)
    if os.environ.get("UI_MODE") == "1":
        import json
        is_manual = (status == "needs_manual_action")
        print(f"[UI_FAIL] {json.dumps({'id': product.id, 'reason': error, 'is_manual': is_manual}, ensure_ascii=False)}")


def _lovart_project_url(project_id: str = "") -> str:
    return f"https://www.lovart.ai/canvas?projectId={project_id}" if project_id else ""


def _project_id_from_url(project_url: str = "") -> str:
    marker = "projectId="
    if marker not in project_url:
        return ""
    return project_url.split(marker, 1)[1].split("&", 1)[0].strip()


def _project_url_from_status(status: dict) -> str:
    return status.get("project_url") or _lovart_project_url(status.get("project_id", ""))


def _existing_project_id(status: dict) -> str:
    return status.get("project_id") or _project_id_from_url(status.get("project_url", ""))


def _can_reuse_lovart_project(lovart, project_id: str, logger) -> bool:
    if not project_id:
        return False
    if not hasattr(lovart, "validate_project"):
        return True
    try:
        return bool(lovart.validate_project(project_id))
    except Exception as exc:
        logger.warning(f"Lovart project validation failed for {project_id}: {exc}")
        return False


def _existing_path(path: str | Path | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    return str(candidate) if candidate.exists() else ""


def _find_support_image(product_dir: Path, status: dict, step_name: str, final_index: int) -> str:
    """Find an already downloaded Lovart support image for resume."""
    keys = [
        f"lovart_{step_name}_local_path",
        f"{step_name}_local_path",
    ]
    for key in keys:
        found = _existing_path(status.get(key))
        if found:
            return found

    final_images = status.get("lovart_final_images") or []
    if isinstance(final_images, list) and len(final_images) > final_index:
        found = _existing_path(final_images[final_index])
        if found:
            return found

    step_dir = product_dir / "lovart_steps" / step_name
    if step_dir.exists():
        image_exts = {".png", ".jpg", ".jpeg", ".webp"}
        candidates = [
            path for path in step_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_exts and path.stat().st_size > 0
        ]
        if candidates:
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return str(candidates[0])

    return ""


def _backfill_result_project_urls(results_path: str | Path = None) -> int:
    from utils import get_output_dir
    if results_path is None:
        results_path = f"{get_output_dir()}/results.csv"
    results_path = Path(results_path)
    if not results_path.exists() or results_path.stat().st_size == 0:
        return 0

    fieldnames, rows = _read_csv_dict_rows_with_fallback(results_path)

    if not rows:
        return 0

    changed = 0
    by_id = {}
    order = []
    for row in rows:
        product_id = row.get("product_id", "")
        if not product_id:
            continue
        if row.get("project_url"):
            pass
        else:
            status = read_status(product_output_dir(product_id))
            project_url = _project_url_from_status(status)
            if project_url:
                row["project_url"] = project_url
                changed += 1
        if product_id not in by_id:
            order.append(product_id)
        else:
            changed += 1
        by_id[product_id] = row

    if changed:
        try:
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for product_id in order:
                    writer.writerow(by_id[product_id])
        except PermissionError:
            return 0
    return changed


def _dry_run_products(products, logger, run_dir, output_dir=None):
    from utils import get_output_dir
    if output_dir is None:
        output_dir = get_output_dir()
    summary_rows = []
    for product in products:
        product_dir = product_output_dir(product.id, output_dir)
        update_status(
            product_dir,
            "dry_run",
            product_id=product.id,
            product_name=product.name_cn,
            image_size=getattr(product, "image_size", ""),
            language=product.language,
            image_count=len(product.image_paths),
        )
        logger.info(
            f"DRY-RUN {product.id} - {product.name_cn} "
            f"size={getattr(product, 'image_size', '') or '-'} ({len(product.image_paths)} image(s))"
        )
        summary_rows.append({
            "product_id": product.id,
            "product_name": product.name_cn,
            "status": "dry_run",
            "project_url": "",
            "gemini_chars": "",
            "artifact_count": "",
            "duration_seconds": 0,
            "error": "",
        })
    write_run_summary(run_dir, summary_rows)
    return 0, 0, len(products), 0


def _process_products(products, gemini, lovart, logger, run_dir, resume=True):
    console = Console()
    success = fail = skipped = still_running = 0
    summary_rows = []
    backfilled = _backfill_result_project_urls()
    if backfilled:
        logger.info(f"Backfilled {backfilled} Lovart project URL(s) in output/results.csv")

    for idx, product in enumerate(products, 1):
        if _shutdown_requested:
            break

        started = time.time()
        product_dir = product_output_dir(product.id)
        update_status(
            product_dir,
            "parsed",
            product_id=product.id,
            product_name=product.name_cn,
            image_size=getattr(product, "image_size", ""),
            language=product.language,
            image_count=len(product.image_paths),
        )

        if resume and is_product_completed(product_dir):
            skipped += 1
            status = read_status(product_dir)
            project_url = status.get("project_url", "")
            from utils import get_output_dir
            append_result(f"{get_output_dir()}/results.csv", product.id, product.name_cn, project_url, status="success")
            logger.info(f"SKIP [{idx}/{len(products)}] {product.id} already completed")
            if project_url:
                console.print(f"  [green]SKIP[/green] {product.id} already completed: [link={project_url}]{project_url}[/link]")
            summary_rows.append({
                "product_id": product.id,
                "product_name": product.name_cn,
                "status": "skipped",
                "project_url": project_url,
                "gemini_chars": status.get("gemini_chars", ""),
                "artifact_count": status.get("artifact_count", ""),
                "duration_seconds": 0,
                "error": "",
            })
            continue

        logger.info(f"[{idx}/{len(products)}] {product.id} - {product.name_cn}")
        console.print(Panel(
            f"[bold cyan]Product ID:[/bold cyan] {product.id}\n[bold cyan]Name:[/bold cyan] {product.name_cn}",
            title=f"[bold green]Processing [{idx}/{len(products)}][/bold green]",
            border_style="blue",
        ))

        try:
            image_roles = split_image_roles(product.image_paths)
            product_image = image_roles["product_image"]
            if not product_image:
                logger.error(f"Skipping '{product.id}': no main product image found in Excel.")
                update_status(product_dir, "failed", reason="No main product image found in Excel.")
                _record_failure(product, "failed", error="No main product image found in Excel")
                continue
            accessory_image = image_roles["accessory_image"]
            dimension_image = image_roles["dimension_image"]
            reference_images = image_roles["reference_images"]
            reference_sheet = ""
            if reference_images:
                reference_sheet = merge_reference_images(
                    reference_images,
                    product_dir / "reference_sheet.jpg",
                )
            update_status(
                product_dir,
                "image_roles_ready",
                product_image=product_image,
                accessory_image=accessory_image,
                dimension_image=dimension_image,
                reference_image_count=len(reference_images),
                reference_sheet=reference_sheet,
                reference_images_are_product=getattr(product, "reference_images_are_product", False),
            )

            status = read_status(product_dir)
            previous_lovart_project_id = _existing_project_id(status)
            restart_lovart_project = False
            if previous_lovart_project_id and _can_reuse_lovart_project(lovart, previous_lovart_project_id, logger):
                lovart_project_id = previous_lovart_project_id
                update_status(
                    product_dir,
                    "lovart_project_reused",
                    project_id=lovart_project_id,
                    project_url=_lovart_project_url(lovart_project_id),
                )
            else:
                if previous_lovart_project_id:
                    restart_lovart_project = True
                    logger.warning(
                        f"Lovart project {previous_lovart_project_id} for '{product.id}' is invalid; restarting product"
                    )
                    update_status(
                        product_dir,
                        "lovart_project_invalid",
                        previous_project_id=previous_lovart_project_id,
                        previous_project_url=_lovart_project_url(previous_lovart_project_id),
                    )
                lovart_project_id = lovart.create_project(product.id, product.name_cn)
                update_status(
                    product_dir,
                    "lovart_project_created",
                    project_id=lovart_project_id,
                    project_url=_lovart_project_url(lovart_project_id),
                )

            status = read_status(product_dir)
            white_image = "" if restart_lovart_project else _find_support_image(product_dir, status, "white_bg", 0)
            if white_image:
                logger.info(f"Lovart API: reusing white_bg image for '{product.id}'")
            else:
                white_result = lovart.create_support_image(
                    product_id=product.id,
                    step_name="white_bg",
                    prompt=build_white_background_prompt(getattr(product, "image_size", "")),
                    image_paths=[product_image],
                    project_id=lovart_project_id,
                    confirmation_advisor=gemini,
                    product_name_cn=product.name_cn,
                    language=product.language,
                    selling_points=product.selling_points,
                )
                white_image = (white_result or {}).get("local_path", "")
                if not white_image:
                    if white_result and white_result.get("final_status") == "timeout":
                        status = read_status(product_dir)
                        project_url = _project_url_from_status(status)
                        reason = "Lovart white-background image still running after local wait timeout"
                        logger.warning(f"STILL RUNNING [{idx}/{len(products)}] {product.id} white_bg")
                        still_running += 1
                        _record_failure(product, "lovart_still_running", reason, project_url)
                        summary_rows.append({
                            "product_id": product.id,
                            "product_name": product.name_cn,
                            "status": "lovart_still_running",
                            "project_url": project_url,
                            "gemini_chars": "",
                            "artifact_count": "",
                            "duration_seconds": round(time.time() - started, 2),
                            "error": reason,
                        })
                        continue
                    if white_result and white_result.get("final_status") == "pending_confirmation":
                        status = read_status(product_dir)
                        project_url = _project_url_from_status(status)
                        reason = white_result.get("warning") or "Lovart white-background image needs credit confirmation"
                        logger.warning(f"NEEDS MANUAL ACTION [{idx}/{len(products)}] {product.id} white_bg")
                        console.print(Panel(
                            f"[bold yellow]Manual Action Required for {product.id} (White BG)[/bold yellow]\n{reason}\nURL: {project_url}",
                            border_style="yellow"
                        ))
                        fail += 1
                        _record_failure(product, "needs_manual_action", reason, project_url)
                        summary_rows.append({
                            "product_id": product.id,
                            "product_name": product.name_cn,
                            "status": "needs_manual_action",
                            "project_url": project_url,
                            "gemini_chars": "",
                            "artifact_count": "",
                            "duration_seconds": round(time.time() - started, 2),
                            "error": reason,
                        })
                        continue
                    error_msg = (white_result or {}).get("warning") or (white_result or {}).get("error") or "Unknown API error"
                    raise RuntimeError(f"Lovart API 失败: {error_msg}")

            status = read_status(product_dir)
            scene_image = "" if restart_lovart_project else _find_support_image(product_dir, status, "scene", 1)
            if scene_image:
                logger.info(f"Lovart API: reusing scene image for '{product.id}'")
            else:
                scene_result = lovart.create_support_image(
                    product_id=product.id,
                    step_name="scene",
                    prompt=build_scene_prompt(getattr(product, "image_size", "")),
                    image_paths=[white_image],
                    project_id=lovart_project_id,
                    confirmation_advisor=gemini,
                    product_name_cn=product.name_cn,
                    language=product.language,
                    selling_points=product.selling_points,
                )
                scene_image = (scene_result or {}).get("local_path", "")
                if not scene_image:
                    if scene_result and scene_result.get("final_status") == "timeout":
                        status = read_status(product_dir)
                        project_url = _project_url_from_status(status)
                        reason = "Lovart scene image still running after local wait timeout"
                        logger.warning(f"STILL RUNNING [{idx}/{len(products)}] {product.id} scene")
                        still_running += 1
                        _record_failure(product, "lovart_still_running", reason, project_url)
                        summary_rows.append({
                            "product_id": product.id,
                            "product_name": product.name_cn,
                            "status": "lovart_still_running",
                            "project_url": project_url,
                            "gemini_chars": "",
                            "artifact_count": "",
                            "duration_seconds": round(time.time() - started, 2),
                            "error": reason,
                        })
                        continue
                    if scene_result and scene_result.get("final_status") == "pending_confirmation":
                        status = read_status(product_dir)
                        project_url = _project_url_from_status(status)
                        reason = scene_result.get("warning") or "Lovart scene image needs credit confirmation"
                        logger.warning(f"NEEDS MANUAL ACTION [{idx}/{len(products)}] {product.id} scene")
                        console.print(Panel(
                            f"[bold yellow]Manual Action Required for {product.id} (Scene)[/bold yellow]\n{reason}\nURL: {project_url}",
                            border_style="yellow"
                        ))
                        fail += 1
                        _record_failure(product, "needs_manual_action", reason, project_url)
                        summary_rows.append({
                            "product_id": product.id,
                            "product_name": product.name_cn,
                            "status": "needs_manual_action",
                            "project_url": project_url,
                            "gemini_chars": "",
                            "artifact_count": "",
                            "duration_seconds": round(time.time() - started, 2),
                            "error": reason,
                        })
                        continue
                    error_msg = (scene_result or {}).get("warning") or (scene_result or {}).get("error") or "Unknown API error"
                    raise RuntimeError(f"Lovart API 失败: {error_msg}")

            gemini_images = [white_image, scene_image]
            if reference_sheet:
                gemini_images.append(reference_sheet)
            lovart_images = build_final_lovart_images(
                white_image=white_image,
                scene_image=scene_image,
                accessory_image=accessory_image,
                dimension_image=dimension_image,
                reference_sheet=reference_sheet,
            )
            image_note = build_lovart_image_note(
                has_reference_sheet=bool(reference_sheet),
                has_accessory_image=bool(accessory_image),
                has_dimension_image=bool(dimension_image),
                reference_images_are_product=getattr(product, "reference_images_are_product", False),
            )
            update_status(
                product_dir,
                "lovart_final_images_ready",
                lovart_final_image_count=len(lovart_images),
                lovart_final_images=lovart_images,
                lovart_white_bg_local_path=white_image,
                lovart_scene_local_path=scene_image,
                project_id=lovart_project_id,
                project_url=_lovart_project_url(lovart_project_id),
            )

            prompt = gemini.generate_prompt(
                product_id=product.id,
                product_name_cn=product.name_cn,
                language=product.language,
                selling_points=product.selling_points,
                image_paths=gemini_images,
                image_size=getattr(product, "image_size", ""),
            )
            logger.info(f"Gemini done ({len(prompt)} chars)")
            if _shutdown_requested:
                break

            lovart_prompt = build_lovart_prompt(
                product_name_cn=product.name_cn,
                language=product.language,
                selling_points=product.selling_points,
                generated_prompt=prompt,
                image_note=image_note,
                image_size=getattr(product, "image_size", ""),
            )
            (product_dir / "lovart_prompt.txt").write_text(lovart_prompt, encoding="utf-8")
            update_status(product_dir, "lovart_prompt_ready", lovart_prompt_chars=len(lovart_prompt))
            logger.info(f"Lovart prompt ready ({len(lovart_prompt)} chars)")

            if os.environ.get("UI_MODE") == "1":
                print(f"[UI_MODEL] {lovart.tool_config.get('image_model', 'auto')}", flush=True)

            result = lovart.create_and_generate(
                product_id=product.id,
                prompt=lovart_prompt,
                image_paths=lovart_images,
                project_id=lovart_project_id,
                confirmation_advisor=gemini,
                product_name_cn=product.name_cn,
                language=product.language,
                selling_points=product.selling_points,
            )

            if result and result.get("generation_succeeded"):
                url = _record_success(product, result)
                logger.info(f"OK [{idx}/{len(products)}] {product.id} completed")
                if url:
                    print(f"\n  >>> {url}")
                if os.environ.get("UI_MODE") == "1":
                    import json
                    print(f"[UI_SUCCESS] {json.dumps({'id': product.id, 'url': url or '', 'used_model': result.get('used_model', 'unknown')}, ensure_ascii=False)}")
                success += 1
                status = read_status(product_dir)
                summary_rows.append({
                    "product_id": product.id,
                    "product_name": product.name_cn,
                    "status": "success",
                    "project_url": url,
                    "gemini_chars": len(prompt),
                    "artifact_count": status.get("artifact_count", ""),
                    "duration_seconds": round(time.time() - started, 2),
                    "error": "",
                    "used_model": result.get("used_model", "unknown")
                })
            elif result and result.get("final_status") == "pending_confirmation":
                logger.warning(f"NEEDS MANUAL ACTION [{idx}/{len(products)}] {product.id}")
                fail += 1
                status = read_status(product_dir)
                project_url = _project_url_from_status(status)
                _record_failure(product, "needs_manual_action", "Lovart pending confirmation on all fallback models", project_url)
                summary_rows.append({
                    "product_id": product.id,
                    "product_name": product.name_cn,
                    "status": "needs_manual_action",
                    "project_url": project_url,
                    "gemini_chars": len(prompt),
                    "artifact_count": "",
                    "duration_seconds": round(time.time() - started, 2),
                    "error": "Lovart pending confirmation on all fallback models",
                })
            elif result and result.get("final_status") == "timeout":
                status = read_status(product_dir)
                project_url = status.get("project_url", "")
                logger.warning(f"STILL RUNNING [{idx}/{len(products)}] {product.id}")
                still_running += 1
                if project_url:
                    print(f"\n  Lovart still running in background: {project_url}")
                _record_failure(
                    product,
                    "lovart_still_running",
                    "Lovart still running after local wait timeout",
                    project_url,
                )
                summary_rows.append({
                    "product_id": product.id,
                    "product_name": product.name_cn,
                    "status": "lovart_still_running",
                    "project_url": project_url,
                    "gemini_chars": len(prompt),
                    "artifact_count": "",
                    "duration_seconds": round(time.time() - started, 2),
                    "error": "Lovart still running after local wait timeout",
                })
            else:
                logger.warning(f"WARN [{idx}/{len(products)}] {product.id} failed")
                fail += 1
                reason = ""
                if result:
                    reason = result.get("warning") or result.get("final_status") or ""
                status = read_status(product_dir)
                project_url = _project_url_from_status(status)
                update_status(product_dir, "failed", reason=reason, project_url=project_url)
                _record_failure(product, "failed", reason, project_url)
                summary_rows.append({
                    "product_id": product.id,
                    "product_name": product.name_cn,
                    "status": "failed",
                    "project_url": project_url,
                    "gemini_chars": len(prompt),
                    "artifact_count": "",
                    "duration_seconds": round(time.time() - started, 2),
                    "error": reason,
                })
        except Exception as exc:
            status = read_status(product_dir)
            project_url = _project_url_from_status(status)
            update_status(product_dir, "failed", reason=str(exc), project_url=project_url)
            logger.error(f"FAIL [{idx}/{len(products)}] {product.id}: {exc}")
            fail += 1
            _record_failure(product, "failed", str(exc), project_url)
            summary_rows.append({
                "product_id": product.id,
                "product_name": product.name_cn,
                "status": "failed",
                "project_url": project_url,
                "gemini_chars": "",
                "artifact_count": "",
                "duration_seconds": round(time.time() - started, 2),
                "error": str(exc),
            })

        write_run_summary(run_dir, summary_rows)

    write_run_summary(run_dir, summary_rows)
    return success, fail, skipped, still_running


def _build_gemini_api(config, logger):
    api_cfg = config.get("gemini_api", {})
    api_key = env_or_config(api_cfg, "api_key", "GEMINI_API_KEY")
    model = api_cfg.get("model", "gemini-2.5-flash-lite")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        sys.exit(1)
    logger.info(f"Using Gemini API: {model}")
    return GeminiAPI(api_key=api_key, model=model, logger=logger)


def _build_nvidia_api(config, logger):
    nvidia_cfg = config.get("nvidia_api", {})
    api_key = env_or_config(nvidia_cfg, "api_key", "NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: NVIDIA_API_KEY is not set.")
        sys.exit(1)
    model = resolve_nvidia_model(nvidia_cfg)
    base_url = nvidia_cfg.get("base_url", "https://integrate.api.nvidia.com/v1")
    send_images = bool(nvidia_cfg.get("send_images", True))
    if logger:
        logger.info(f"Using NVIDIA API: {model}")
    return NvidiaAPI(
        api_key=api_key,
        model=model,
        base_url=base_url,
        logger=logger,
        send_images=send_images,
    )


def _default_browser_candidates() -> list[str]:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    local_app_data = Path.home() / "AppData" / "Local"
    candidates.extend([
        str(local_app_data / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(local_app_data / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
    ])
    return candidates


def resolve_browser_executable(browser_cfg: dict, candidate_paths: list[str] | None = None) -> str | None:
    configured = str(browser_cfg.get("chrome_exe", "") or "").strip()
    if configured and Path(configured).exists():
        return configured

    for candidate in candidate_paths if candidate_paths is not None else _default_browser_candidates():
        if candidate and Path(candidate).exists():
            return candidate

    return None


def _resolve_browser_executable_for_run(
    browser_cfg: dict,
    interactive: bool = True,
    candidate_paths: list[str] | None = None,
    input_func=input,
) -> str | None:
    chrome_exe = resolve_browser_executable(browser_cfg, candidate_paths=candidate_paths)
    if chrome_exe or not interactive:
        return chrome_exe

    configured = str(browser_cfg.get("chrome_exe", "") or "").strip()
    if configured:
        print(f"\n  Configured browser path was not found: {configured}")
    print("  Chrome/Edge was not found in common install paths.")
    print("  Paste chrome.exe/msedge.exe path, or press Enter to use Playwright bundled Chromium.")

    while True:
        manual_path = input_func("  Browser executable path (optional): ").strip().strip("\"'")
        if not manual_path:
            return None
        if Path(manual_path).exists():
            return manual_path
        print(f"  Browser executable not found: {manual_path}")


def _run_browser_flow(config, products, lovart, logger, run_dir, resume=True, wait_for_ready=True):
    browser_cfg = config["browser"]
    user_data_dir = Path(browser_cfg["user_data_dir"])
    if not user_data_dir.is_absolute():
        user_data_dir = Path.cwd() / user_data_dir
    user_data_dir.mkdir(parents=True, exist_ok=True)

    chrome_exe = _resolve_browser_executable_for_run(browser_cfg, interactive=wait_for_ready)
    if chrome_exe:
        logger.info(f"Using browser executable: {chrome_exe}")
    else:
        logger.warning("Chrome/Edge executable not found; using Playwright bundled Chromium")

    with sync_playwright() as pw:
        logger.info("Launching browser for Gemini")
        launch_options = {
            "user_data_dir": str(user_data_dir),
            "headless": False,
            "no_viewport": True,
            "args": [
                "--disable-blink-features=AutomationControlled",
                "--disable-features=ImprovedCookieControls",
            ],
        }
        if chrome_exe:
            launch_options["executable_path"] = chrome_exe
        context = pw.chromium.launch_persistent_context(**launch_options)
        page = context.pages[0] if context.pages else context.new_page()
        page.goto(config["gemini"]["base_url"], wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(3000)

        if any(keyword in page.url.lower() for keyword in ["login", "accounts.google.com", "signin"]):
            print("\n  Gemini requires login. Log in, then press Enter.")
            input("  Press Enter...")
            page.goto(config["gemini"]["base_url"], wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2000)

        logger.info("Gemini browser ready")
        if wait_for_ready:
            input("\nReady. Press Enter to start...")
        gemini = GeminiBot(page, config, logger, run_dir=run_dir)
        result = _process_products(products, gemini, lovart, logger, run_dir, resume=resume)
        context.close()
        return result


def _generate_excel_template():
    import openpyxl
    target = Path("data/标准测试模板.xlsx")
    target.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "商品测试"
    headers = ["商品ID", "商品名称", "尺寸要求", "多语言", "卖点描述", "商品图1(产品图)", "商品图2(配件图)", "参考图是否是同产品", "参考图1"]
    ws.append(headers)
    ws.append(["T001", "测试商品", "11:15", "英文", "防水防刮，多色可选", "", "", "否", ""])
    wb.save(target)
    print(f"\n✅ 成功生成标准 Excel 模板: {target.absolute()}\n  请填入您的商品信息并在图片列嵌入(或DISPIMG)真实图片。")
    sys.exit(0)


def main(argv=None):
    args = parse_args(argv)
    
    if args.generate_template:
        _generate_excel_template()

    # Auto-diagnostic: Check required .env keys before doing anything else
    try:
        from setup_wizard import missing_or_placeholder_env_keys
        missing_keys = missing_or_placeholder_env_keys(Path(".env"))
        if missing_keys:
            print("\n[!] Auto-Diagnostic Failed: Required environment variables are missing or invalid:")
            for key in missing_keys:
                print(f"  - {key}")
            print("\nPlease fill them in `.env` before running.")
            sys.exit(1)
    except ImportError:
        pass

    config = load_config(args.config)
    logger = setup_logging()
    run_dir = create_run_dir()
    logger.info("Image Automation started")
    logger.info(f"Run artifacts: {run_dir}")

    if args.limit is not None and args.limit < 1:
        logger.error("--limit must be >= 1")
        sys.exit(1)

    try:
        products = read_products(config, logger, limit=args.limit)
    except Exception as exc:
        logger.error(f"Failed to read Excel: {exc}")
        sys.exit(1)

    if not products:
        logger.error("No products found in Excel")
        sys.exit(1)

    for idx, product in enumerate(products, 1):
        print(
            f"  [{idx}] {product.id} | {product.name_cn} | "
            f"size={getattr(product, 'image_size', '') or '-'} | "
            f"lang={product.language} | {len(product.image_paths)} image(s)"
        )
        if os.environ.get("UI_MODE") == "1":
            import json
            from utils import split_image_roles
            roles = split_image_roles(product.image_paths)
            img = str(roles["product_image"]).replace("\\", "/") if roles["product_image"] else ""
            print(f"[UI_PRODUCT] {json.dumps({'id': product.id, 'name': product.name_cn, 'image': img}, ensure_ascii=False)}")

    if args.dry_run:
        success, fail, skipped, still_running = _dry_run_products(products, logger, run_dir)
        print(f"\nDRY-RUN DONE - Parsed: {skipped}, Total: {len(products)}")
        print(f"Run summary: {run_dir}")
        logger.info(f"Dry-run complete. Parsed={skipped}")
        return

    prompt_source = _choose_prompt_source(config, args)
    fast_mode = _resolve_lovart_mode(args.lovart)
    _choose_lovart_tool_options(config, args)

    lovart = LovartBot(config, logger)
    lovart.set_fast_mode(fast_mode)
    logger.info(f"Lovart mode: {'fast' if fast_mode else 'unlimited'}")

    signal.signal(signal.SIGINT, _on_sigint)

    if prompt_source == "gemini_api":
        gemini = _build_gemini_api(config, logger)
        if args.prompt_source == "ask" or args.lovart == "ask":
            input("\nReady. Press Enter to start...")
        success, fail, skipped, still_running = _process_products(products, gemini, lovart, logger, run_dir, resume=args.resume)
    elif prompt_source == "nvidia":
        prompt_client = _build_nvidia_api(config, logger)
        if args.prompt_source == "ask" or args.lovart == "ask":
            input("\nReady. Press Enter to start...")
        success, fail, skipped, still_running = _process_products(products, prompt_client, lovart, logger, run_dir, resume=args.resume)
    else:
        success, fail, skipped, still_running = _run_browser_flow(
            config,
            products,
            lovart,
            logger,
            run_dir,
            resume=args.resume,
            wait_for_ready=args.prompt_source == "ask" or args.lovart == "ask",
        )

    print(
        f"\nDONE - Success: {success}, Failed: {fail}, "
        f"Still running: {still_running}, Skipped: {skipped}, Total: {len(products)}"
    )
    print(f"Run summary: {run_dir}")
    logger.info(
        f"Session complete. Success={success}, Failed={fail}, "
        f"StillRunning={still_running}, Skipped={skipped}"
    )
    
    from utils import organize_output_folders
    organize_output_folders()


if __name__ == "__main__":
    main()
