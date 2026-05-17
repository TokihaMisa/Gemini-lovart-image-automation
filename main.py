import argparse
import signal
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from excel_reader import read_products
from gemini_api import GeminiAPI
from gemini_bot import GeminiBot
from lovart_bot import LOVART_IMAGE_MODELS, LovartBot
from nvidia_api import NvidiaAPI, resolve_nvidia_model
from utils import (
    SCENE_PROMPT,
    WHITE_BACKGROUND_PROMPT,
    append_result,
    build_final_lovart_images,
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
        return "gemini_api"

    config.setdefault("nvidia_api", {})["model_choice"] = "kimi"
    return "nvidia"


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
    append_result("output/results.csv", product.id, product.name_cn, project_url, status="success")
    return project_url


def _record_failure(product, status: str, error: str = "", project_url: str = "") -> None:
    append_result("output/results.csv", product.id, product.name_cn, project_url, status=status, error=error)


def _dry_run_products(products, logger, run_dir, output_dir="output"):
    summary_rows = []
    for product in products:
        product_dir = product_output_dir(product.id, output_dir)
        update_status(
            product_dir,
            "dry_run",
            product_id=product.id,
            product_name=product.name_cn,
            language=product.language,
            image_count=len(product.image_paths),
        )
        logger.info(f"DRY-RUN {product.id} - {product.name_cn} ({len(product.image_paths)} image(s))")
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
    success = fail = skipped = still_running = 0
    summary_rows = []

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
            language=product.language,
            image_count=len(product.image_paths),
        )

        if resume and is_product_completed(product_dir):
            skipped += 1
            status = read_status(product_dir)
            logger.info(f"SKIP [{idx}/{len(products)}] {product.id} already completed")
            if status.get("project_url"):
                print(f"\n  SKIP {product.id} already completed: {status['project_url']}")
            summary_rows.append({
                "product_id": product.id,
                "product_name": product.name_cn,
                "status": "skipped",
                "project_url": status.get("project_url", ""),
                "gemini_chars": status.get("gemini_chars", ""),
                "artifact_count": status.get("artifact_count", ""),
                "duration_seconds": 0,
                "error": "",
            })
            continue

        logger.info(f"[{idx}/{len(products)}] {product.id} - {product.name_cn}")
        print(f"\n{'-' * 50}\n[{idx}/{len(products)}] {product.id} | {product.name_cn}\n{'-' * 50}")

        try:
            image_roles = split_image_roles(product.image_paths)
            product_image = image_roles["product_image"]
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
            )

            lovart_project_id = lovart.create_project(product.id)
            update_status(product_dir, "lovart_project_created", project_id=lovart_project_id)

            white_result = lovart.create_support_image(
                product_id=product.id,
                step_name="white_bg",
                prompt=WHITE_BACKGROUND_PROMPT,
                image_paths=[product_image],
                project_id=lovart_project_id,
                confirmation_advisor=gemini,
                product_name_cn=product.name_cn,
                language=product.language,
                selling_points=product.selling_points,
            )
            white_image = (white_result or {}).get("local_path", "")
            if not white_image:
                raise RuntimeError("Lovart white-background image generation did not return a local image")

            scene_result = lovart.create_support_image(
                product_id=product.id,
                step_name="scene",
                prompt=SCENE_PROMPT,
                image_paths=[white_image],
                project_id=lovart_project_id,
                confirmation_advisor=gemini,
                product_name_cn=product.name_cn,
                language=product.language,
                selling_points=product.selling_points,
            )
            scene_image = (scene_result or {}).get("local_path", "")
            if not scene_image:
                raise RuntimeError("Lovart scene image generation did not return a local image")

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
            )
            update_status(
                product_dir,
                "lovart_final_images_ready",
                lovart_final_image_count=len(lovart_images),
                lovart_final_images=lovart_images,
            )

            prompt = gemini.generate_prompt(
                product_id=product.id,
                product_name_cn=product.name_cn,
                language=product.language,
                selling_points=product.selling_points,
                image_paths=gemini_images,
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
            )
            (product_dir / "lovart_prompt.txt").write_text(lovart_prompt, encoding="utf-8")
            update_status(product_dir, "lovart_prompt_ready", lovart_prompt_chars=len(lovart_prompt))
            logger.info(f"Lovart prompt ready ({len(lovart_prompt)} chars)")

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
                })
            elif result and result.get("final_status") == "pending_confirmation":
                logger.warning(f"NEEDS MANUAL ACTION [{idx}/{len(products)}] {product.id}")
                fail += 1
                _record_failure(product, "needs_manual_action", "Lovart pending confirmation")
                summary_rows.append({
                    "product_id": product.id,
                    "product_name": product.name_cn,
                    "status": "needs_manual_action",
                    "project_url": "",
                    "gemini_chars": len(prompt),
                    "artifact_count": "",
                    "duration_seconds": round(time.time() - started, 2),
                    "error": "Lovart pending confirmation",
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
                _record_failure(product, "failed", reason)
                summary_rows.append({
                    "product_id": product.id,
                    "product_name": product.name_cn,
                    "status": "failed",
                    "project_url": "",
                    "gemini_chars": len(prompt),
                    "artifact_count": "",
                    "duration_seconds": round(time.time() - started, 2),
                    "error": reason,
                })
        except Exception as exc:
            update_status(product_dir, "failed", reason=str(exc))
            logger.error(f"FAIL [{idx}/{len(products)}] {product.id}: {exc}")
            fail += 1
            _record_failure(product, "failed", str(exc))
            summary_rows.append({
                "product_id": product.id,
                "product_name": product.name_cn,
                "status": "failed",
                "project_url": "",
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


def _run_browser_flow(config, products, lovart, logger, run_dir, resume=True, wait_for_ready=True):
    browser_cfg = config["browser"]
    user_data_dir = Path(browser_cfg["user_data_dir"])
    if not user_data_dir.is_absolute():
        user_data_dir = Path.cwd() / user_data_dir
    user_data_dir.mkdir(parents=True, exist_ok=True)

    chrome_exe = browser_cfg.get("chrome_exe", "")
    if chrome_exe and not Path(chrome_exe).exists():
        logger.error(f"Chrome executable not found: {chrome_exe}")
        sys.exit(1)

    with sync_playwright() as pw:
        logger.info("Launching browser for Gemini")
        context = pw.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            executable_path=chrome_exe,
            headless=False,
            no_viewport=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=ImprovedCookieControls",
            ],
        )
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


def main(argv=None):
    args = parse_args(argv)
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
        print(f"  [{idx}] {product.id} | {product.name_cn} | lang={product.language} | {len(product.image_paths)} image(s)")

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


if __name__ == "__main__":
    main()
