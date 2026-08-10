import argparse
from collections.abc import Mapping
import csv
import io
import json
import os
import signal
import sys
import time
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8')
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding='utf-8')

from playwright.sync_api import sync_playwright

from playwright.sync_api import sync_playwright
from rich.console import Console
from rich.panel import Panel

from excel_reader import read_products
from failed_retry import FailedRetryPolicy, classify_retry_failure
from gemini_api import GeminiAPI
from gemini_bot import GeminiBot
from image_generation import (
    DetailScreen,
    GenerationRouting,
    PROVIDER_LOVART,
    PROVIDER_OPENAI_IMAGE,
    build_detail_input_fingerprint,
    compose_detail_image_prompt,
    ensure_detail_page_count_snapshot,
    normalize_image_provider,
    routing_from_config,
    split_detail_screens,
)
from image_providers import (
    DetailSetRequest,
    LazyImageProviderRegistry,
    LovartImageProvider,
    OpenAIImageProvider,
    SupportImageRequest,
    detail_screen_prompt_hash,
    is_valid_image_file,
    read_completed_detail_indexes,
)
from gemini_browser_session import (
    GeminiAuthenticationError,
    GeminiLoginRequiredError,
    GeminiPageNotReadyError,
    GeminiPermanentTlsError,
    GeminiResourceNotFoundError,
    GeminiPageState,
    acquire_login_helper_owner,
    build_browser_launch_options,
    login_runtime_paths,
    navigate_gemini_with_retry,
    release_login_helper_owner,
    resolve_browser_executable,
    resolve_user_data_dir,
    retry_policy_from_config,
)
from network_retry import RetryKind, classify_network_error
from lovart_bot import LOVART_IMAGE_MODELS, LovartBot
from nvidia_api import NvidiaAPI, resolve_nvidia_model
from openai_image_api import OpenAIImageAPI, OpenAIImageAPIConfig
from prompt_settings import get_prompt_settings, normalize_prompt_settings
from utils import (
    _read_csv_dict_rows_with_fallback,
    append_result,
    build_final_lovart_images,
    build_scene_prompt,
    build_white_background_prompt,
    build_detail_prompt,
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


def _is_ui_mode() -> bool:
    return os.environ.get("UI_MODE") == "1"


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
    parser.add_argument(
        "--support-provider",
        choices=[PROVIDER_LOVART, PROVIDER_OPENAI_IMAGE],
        default=None,
        help="Override the provider used for white-background and scene support images.",
    )
    parser.add_argument(
        "--detail-provider",
        choices=[PROVIDER_LOVART, PROVIDER_OPENAI_IMAGE],
        default=None,
        help="Override the provider used for the final detail image set.",
    )
    return parser.parse_args(argv)


def _routing_with_cli_overrides(routing: GenerationRouting, args) -> GenerationRouting:
    return GenerationRouting(
        support_provider=normalize_image_provider(
            getattr(args, "support_provider", None) or routing.support_provider
        ),
        detail_provider=normalize_image_provider(
            getattr(args, "detail_provider", None) or routing.detail_provider
        ),
        detail_page_count=routing.detail_page_count,
    )


def _emit_ui_detail_progress(current, target, completed, failed) -> None:
    if not _is_ui_mode():
        return
    payload = {
        "current": max(0, int(current)),
        "target": max(0, int(target)),
        "completed": max(0, int(completed)),
        "failed": sorted({max(1, int(index)) for index in failed}),
    }
    print(f"[UI_DETAIL_PROGRESS] {json.dumps(payload)}", flush=True)


def _emit_ui_status(product_id: str, stage: str, message: str) -> None:
    if not _is_ui_mode():
        return
    payload = {
        "id": str(product_id),
        "stage": str(stage),
        "message": str(message),
    }
    print(f"[UI_STATUS] {json.dumps(payload)}", flush=True)


def _detail_execution_settings(provider) -> dict[str, object]:
    describe = getattr(provider, "detail_execution_settings", None)
    if not callable(describe):
        return {}
    settings = describe()
    return dict(settings) if isinstance(settings, Mapping) else {}


def _detail_fingerprint_execution_settings(
    settings: Mapping[str, object] | None,
) -> dict[str, object]:
    """Treat HAPI's standard and image-only gateways as the same image backend."""
    normalized = dict(settings or {})
    base_url = str(normalized.get("base_url") or "").rstrip("/").lower()
    if base_url in {
        "https://image.hapiopen.cc",
        "https://image.hapiopen.cc/v1",
    }:
        normalized["base_url"] = "https://hapiopen.cc/v1"
    return normalized


def _compose_detail_request_screens(
    screens,
    provider_name,
    *,
    product,
    image_note,
    prompt_settings,
    target_count,
):
    if provider_name != PROVIDER_OPENAI_IMAGE:
        return tuple(screens)
    compose_settings = dict(prompt_settings)
    compose_settings["detail_page_count"] = target_count
    return tuple(
        DetailScreen(
            screen.index,
            compose_detail_image_prompt(
                screen=screen,
                product_name_cn=product.name_cn,
                language=product.language,
                selling_points=product.selling_points,
                image_note=image_note,
                image_size=getattr(product, "image_size", ""),
                prompt_settings=compose_settings,
            ),
        )
        for screen in screens
    )


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
    if _is_ui_mode():
        return default
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
    if _is_ui_mode():
        return default_values
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
    if _is_ui_mode():
        if env_or_config(config.get("gemini_api", {}), "api_key", "GEMINI_API_KEY"):
            return "gemini_api"
        if env_or_config(config.get("nvidia_api", {}), "api_key", "NVIDIA_API_KEY"):
            config.setdefault("nvidia_api", {})["model_choice"] = "kimi"
            return "nvidia"
        return "gemini_browser"

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
    if _is_ui_mode():
        return False
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


def _record_failure(
    product,
    status: str,
    error: str = "",
    project_url: str = "",
    used_model: str = "",
) -> None:
    from utils import get_output_dir
    append_result(
        f"{get_output_dir()}/results.csv",
        product.id,
        product.name_cn,
        project_url,
        status=status,
        error=error,
        used_model=used_model,
    )
    if os.environ.get("UI_MODE") == "1":
        import json
        is_manual = (status == "needs_manual_action")
        print(f"[UI_FAIL] {json.dumps({'id': product.id, 'reason': error, 'is_manual': is_manual})}")


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


def _existing_path(path: str | Path | None) -> str:
    if not path:
        return ""
    candidate = Path(path)
    return str(candidate) if is_valid_image_file(candidate) else ""


def _find_support_image(
    product_dir: Path,
    status: dict,
    step_name: str,
    final_index: int,
    include_lovart_legacy: bool = True,
    provider_name: str = "",
) -> str:
    """Find a resumable support image, with optional Lovart legacy fallback."""
    if include_lovart_legacy:
        found = _existing_path(status.get(f"lovart_{step_name}_local_path"))
        if found:
            return found

    generic_path = _existing_path(status.get(f"{step_name}_local_path"))
    saved_provider = str(status.get(f"{step_name}_provider") or "")
    if generic_path and (
        saved_provider == provider_name
        or include_lovart_legacy and not saved_provider
    ):
        return generic_path

    if include_lovart_legacy:
        final_images = status.get("lovart_final_images") or []
        if isinstance(final_images, list) and len(final_images) > final_index:
            found = _existing_path(final_images[final_index])
            if found:
                return found

    if provider_name == PROVIDER_OPENAI_IMAGE:
        canonical = product_dir / "gpt_image" / "support" / f"{step_name}.png"
        if is_valid_image_file(canonical):
            return str(canonical)

    step_dir = product_dir / "lovart_steps" / step_name
    if include_lovart_legacy and step_dir.exists():
        image_exts = {".png", ".jpg", ".jpeg", ".webp"}
        candidates = [
            path for path in step_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in image_exts
            and is_valid_image_file(path)
        ]
        if candidates:
            candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
            return str(candidates[0])

    return ""


def _is_completed_for_support_provider(
    product_dir: Path,
    status: dict,
    provider_name: str,
) -> bool:
    if not is_product_completed(product_dir):
        return False
    if status.get("lovart_support_resume_invalidated"):
        return False
    include_lovart_legacy = provider_name == PROVIDER_LOVART
    return all(
        _find_support_image(
            product_dir,
            status,
            step_name,
            final_index,
            include_lovart_legacy=include_lovart_legacy,
            provider_name=provider_name,
        )
        for step_name, final_index in (("white_bg", 0), ("scene", 1))
    )


def _is_completed_for_detail_provider(
    product_dir: Path,
    status: dict,
    provider_name: str,
    configured_count: int,
    screen_prompt_hashes: Mapping[int, str] | None = None,
) -> bool:
    saved_provider = str(status.get("detail_provider") or "")
    snapshot = status.get("detail_page_count_snapshot")
    if (
        provider_name == PROVIDER_LOVART
        and status.get("lovart_done")
        and saved_provider in {"", PROVIDER_LOVART}
        and snapshot is None
    ):
        snapshot = ensure_detail_page_count_snapshot(product_dir, configured_count)
        status = update_status(
            product_dir,
            "legacy_lovart_completion_migrated",
            detail_provider=PROVIDER_LOVART,
            detail_generation_complete=True,
            detail_page_count_snapshot=snapshot,
        )
        saved_provider = PROVIDER_LOVART
    elif provider_name == PROVIDER_LOVART and status.get("lovart_done") and not saved_provider:
        status = update_status(
            product_dir,
            "legacy_lovart_completion_migrated",
            detail_provider=PROVIDER_LOVART,
            detail_generation_complete=True,
        )
        saved_provider = PROVIDER_LOVART
    if saved_provider != provider_name or snapshot is None:
        return False
    if not status.get("detail_generation_complete"):
        return False
    if provider_name != PROVIDER_OPENAI_IMAGE:
        return True
    try:
        target_count = int(status["detail_page_count_snapshot"])
    except (KeyError, TypeError, ValueError):
        return False
    if screen_prompt_hashes is None:
        completed_count = 0
    else:
        completed_count = len(
            read_completed_detail_indexes(
                product_dir,
                target_count,
                str(status.get("detail_input_fingerprint") or ""),
                screen_prompt_hashes,
            )
        )
    if completed_count == target_count:
        return True
    update_status(
        product_dir,
        "detail_completion_invalid",
        detail_generation_complete=False,
        partial_complete=completed_count > 0,
        detail_completed_count=completed_count,
        artifact_count=completed_count,
    )
    return False


def _prepare_detail_input_state(
    product_dir: Path,
    input_fingerprint: str,
    support_provider: str,
    detail_provider: str,
    *,
    resume: bool,
) -> None:
    status = read_status(product_dir)
    saved_fingerprint = str(status.get("detail_input_fingerprint") or "")
    legacy_lovart_completion = bool(
        resume
        and not saved_fingerprint
        and support_provider == PROVIDER_LOVART
        and detail_provider == PROVIDER_LOVART
        and status.get("lovart_done")
        and str(status.get("detail_provider") or "") in {"", PROVIDER_LOVART}
    )
    has_detail_state = bool(
        status.get("detail_provider")
        or status.get("detail_generation_complete")
        or status.get("partial_complete")
        or status.get("detail_checkpoints")
        or status.get("detail_images")
        or status.get("lovart_done")
        or (product_dir / "detail_prompt.txt").exists()
        or (product_dir / "lovart_prompt.txt").exists()
    )
    if resume and saved_fingerprint == input_fingerprint:
        return
    if resume and (not has_detail_state or legacy_lovart_completion):
        update_status(
            product_dir,
            "detail_inputs_fingerprinted",
            detail_input_fingerprint=input_fingerprint,
        )
        return

    for prompt_name in ("detail_prompt.txt", "lovart_prompt.txt"):
        prompt_path = product_dir / prompt_name
        if prompt_path.exists():
            prompt_path.unlink()
    output_dir = product_dir / "gpt_image" / "detail"
    if output_dir.exists():
        for output_path in output_dir.iterdir():
            if output_path.is_file():
                output_path.unlink()
    update_status(
        product_dir,
        "detail_inputs_invalidated",
        detail_input_fingerprint=input_fingerprint,
        detail_provider="",
        detail_checkpoints={},
        detail_images=[],
        detail_completed_count=0,
        detail_failed_indexes=[],
        detail_generation_complete=False,
        partial_complete=False,
        artifact_count=0,
        used_model="",
        lovart_done=False,
        failed=False,
        needs_manual_action=False,
        lovart_still_running=False,
        reason="",
        detail_prompt_ready=False,
        lovart_prompt_ready=False,
        detail_result_recorded=False,
        detail_generation_done=False,
        detail_prompt_chars=0,
        lovart_prompt_chars=0,
        gemini_chars=0,
    )


class _SupportImageGenerationError(RuntimeError):
    def __init__(self, step_name: str, result) -> None:
        self.step_name = step_name
        self.result = result
        message = result.error or f"{step_name} support image generation failed"
        super().__init__(message)


def _default_lovart_routing(prompt_settings) -> GenerationRouting:
    settings = normalize_prompt_settings(prompt_settings)
    return GenerationRouting(
        support_provider=PROVIDER_LOVART,
        detail_provider=PROVIDER_LOVART,
        detail_page_count=settings["detail_page_count"],
    )


def _build_image_provider_registry(config, logger, lovart=None):
    def build_lovart_provider():
        bot = lovart or LovartBot(config, logger)
        runtime = config.get("_runtime", {})
        if lovart is None and isinstance(runtime, dict) and "lovart_fast_mode" in runtime:
            bot.set_fast_mode(bool(runtime["lovart_fast_mode"]))
        return LovartImageProvider(bot, logger=logger)

    def build_openai_provider():
        api_key = str(os.environ.get("OPENAI_IMAGE_API_KEY") or "").strip()
        api_config = OpenAIImageAPIConfig.from_config(config, api_key=api_key)
        return OpenAIImageProvider(OpenAIImageAPI(api_config, logger=logger), logger=logger)

    return LazyImageProviderRegistry(build_lovart_provider, build_openai_provider)


def _legacy_lovart_registry(lovart, logger):
    if lovart is None:
        raise ValueError("A Lovart client is required for legacy image-provider routing")
    return LazyImageProviderRegistry(
        lambda: LovartImageProvider(lovart, logger=logger),
        lambda: (_ for _ in ()).throw(
            RuntimeError("OpenAI image provider requires an explicit image registry")
        ),
    )


def _generate_support_images(
    product,
    product_dir,
    provider,
    prompt_settings,
    existing_status,
) -> tuple[str, str]:
    product_dir = Path(product_dir)
    image_roles = split_image_roles(product.image_paths)
    product_image = image_roles["product_image"]
    restart = False
    prepare = getattr(provider, "prepare_support_images", None)
    if callable(prepare):
        restart = bool(prepare(product, product_dir, existing_status))
    is_lovart_provider = isinstance(provider, LovartImageProvider)
    if is_lovart_provider:
        provider_name = PROVIDER_LOVART
    elif isinstance(provider, OpenAIImageProvider):
        provider_name = PROVIDER_OPENAI_IMAGE
    else:
        provider_name = str(getattr(provider, "name", "") or "")

    def generate(step_name: str, prompt: str, image_paths: tuple[str, ...], final_index: int) -> str:
        stage = "support_white" if step_name == "white_bg" else "support_scene"
        label = "白底图" if step_name == "white_bg" else "场景图"
        status = read_status(product_dir)
        existing_path = _find_support_image(
            product_dir,
            status,
            step_name,
            final_index,
            include_lovart_legacy=is_lovart_provider and not restart,
            provider_name=provider_name,
        )
        if existing_path:
            _emit_ui_status(product.id, stage, f"♻️ 正在复用已完成{label}")
            update_status(
                product_dir,
                f"{step_name}_ready",
                **{
                    f"{step_name}_local_path": existing_path,
                    f"{step_name}_provider": provider_name,
                },
            )
            return existing_path
        _emit_ui_status(product.id, stage, f"🎨 正在生成{label}")
        result = provider.generate_support_image(
            SupportImageRequest(
                product_id=product.id,
                product_dir=product_dir,
                step_name=step_name,
                prompt=prompt,
                image_paths=image_paths,
                image_size=getattr(product, "image_size", ""),
                product_name_cn=product.name_cn,
                language=product.language,
                selling_points=product.selling_points,
                confirmation_advisor=getattr(provider, "confirmation_advisor", None),
                status_callback=lambda message: _emit_ui_status(
                    product.id,
                    stage,
                    message,
                ),
            )
        )
        if not result.succeeded or not result.local_paths:
            raise _SupportImageGenerationError(step_name, result)
        local_path = result.local_paths[0]
        update_status(
            product_dir,
            f"{step_name}_ready",
            **{
                f"{step_name}_local_path": local_path,
                f"{step_name}_provider": provider_name,
            },
        )
        return local_path

    white_image = generate(
        "white_bg",
        build_white_background_prompt(
            getattr(product, "image_size", ""),
            prompt_settings=prompt_settings,
        ),
        (product_image,),
        0,
    )
    scene_image = generate(
        "scene",
        build_scene_prompt(
            getattr(product, "image_size", ""),
            prompt_settings=prompt_settings,
        ),
        (white_image,),
        1,
    )
    complete = getattr(provider, "complete_support_images", None)
    if callable(complete):
        complete(product_dir)
    return white_image, scene_image


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
            with results_path.open("w", encoding="utf-8", newline="") as fh:
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


def _process_products_once(
    products,
    gemini,
    lovart,
    logger,
    run_dir,
    resume=True,
    prompt_settings=None,
    image_registry=None,
    routing=None,
):
    prompt_settings = normalize_prompt_settings(prompt_settings)
    registry = image_registry or _legacy_lovart_registry(lovart, logger)
    effective_routing = routing or _default_lovart_routing(prompt_settings)
    support_provider = registry.get(effective_routing.support_provider)
    detail_provider = None
    if hasattr(support_provider, "confirmation_advisor"):
        support_provider.confirmation_advisor = gemini
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
        target_count = ensure_detail_page_count_snapshot(
            product_dir,
            effective_routing.detail_page_count,
            replace_existing=not resume,
        )

        status = read_status(product_dir)
        if resume and is_product_completed(product_dir):
            validate_completed_support = getattr(
                support_provider,
                "validate_completed_support",
                None,
            )
            if callable(validate_completed_support):
                validate_completed_support(product, product_dir, status)
                status = read_status(product_dir)

        logger.info(f"[{idx}/{len(products)}] {product.id} - {product.name_cn}")
        _emit_ui_status(
            product.id,
            "product",
            f"🔄 正在处理商品（{idx}/{len(products)}）",
        )
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

            try:
                white_image, scene_image = _generate_support_images(
                    product,
                    product_dir,
                    support_provider,
                    prompt_settings,
                    read_status(product_dir),
                )
            except _SupportImageGenerationError as support_error:
                raw_result = support_error.result.raw_result or {}
                final_status = raw_result.get("final_status")
                status = read_status(product_dir)
                project_url = _project_url_from_status(status)
                label = "white-background" if support_error.step_name == "white_bg" else "scene"
                if final_status == "timeout":
                    reason = f"Lovart {label} image still running after local wait timeout"
                    logger.warning(
                        f"STILL RUNNING [{idx}/{len(products)}] {product.id} {support_error.step_name}"
                    )
                    still_running += 1
                    outcome = "lovart_still_running"
                elif final_status == "pending_confirmation":
                    reason = support_error.result.error or f"Lovart {label} image needs credit confirmation"
                    logger.warning(
                        f"NEEDS MANUAL ACTION [{idx}/{len(products)}] "
                        f"{product.id} {support_error.step_name}"
                    )
                    console.print(Panel(
                        f"[bold yellow]Manual Action Required for {product.id} "
                        f"({label})[/bold yellow]\n{reason}\nURL: {project_url}",
                        border_style="yellow",
                    ))
                    fail += 1
                    outcome = "needs_manual_action"
                else:
                    raise
                _record_failure(product, outcome, reason, project_url)
                summary_rows.append({
                    "product_id": product.id,
                    "product_name": product.name_cn,
                    "status": outcome,
                    "project_url": project_url,
                    "gemini_chars": "",
                    "artifact_count": "",
                    "duration_seconds": round(time.time() - started, 2),
                    "error": reason,
                })
                continue

            status = read_status(product_dir)
            lovart_project_id = _existing_project_id(status)
            if detail_provider is None:
                detail_provider = registry.get(effective_routing.detail_provider)
                if hasattr(detail_provider, "confirmation_advisor"):
                    detail_provider.confirmation_advisor = gemini

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
            support_fields = {
                "white_bg_local_path": white_image,
                "scene_local_path": scene_image,
            }
            if effective_routing.support_provider == PROVIDER_LOVART:
                support_fields.update(
                    lovart_final_image_count=len(lovart_images),
                    lovart_final_images=lovart_images,
                    lovart_white_bg_local_path=white_image,
                    lovart_scene_local_path=scene_image,
                    project_id=lovart_project_id,
                    project_url=_lovart_project_url(lovart_project_id),
                )
                support_stage = "lovart_final_images_ready"
            else:
                support_stage = "support_images_ready"
            update_status(product_dir, support_stage, **support_fields)

            fingerprint_settings = dict(prompt_settings)
            fingerprint_settings["detail_page_count"] = target_count
            detail_input_fingerprint = build_detail_input_fingerprint(
                support_provider=effective_routing.support_provider,
                detail_provider=effective_routing.detail_provider,
                product_id=product.id,
                product_name_cn=product.name_cn,
                language=product.language,
                selling_points=product.selling_points,
                image_size=getattr(product, "image_size", ""),
                reference_images_are_product=getattr(
                    product,
                    "reference_images_are_product",
                    False,
                ),
                prompt_settings=fingerprint_settings,
                target_count=target_count,
                image_inputs={
                    "product_source": (product_image,),
                    "white_bg": (white_image,),
                    "scene": (scene_image,),
                    "accessory": (accessory_image,) if accessory_image else (),
                    "dimension": (dimension_image,) if dimension_image else (),
                    "reference_images": tuple(reference_images),
                    "reference_sheet": (reference_sheet,) if reference_sheet else (),
                },
                detail_execution_settings=_detail_fingerprint_execution_settings(
                    _detail_execution_settings(detail_provider)
                ),
            )
            _prepare_detail_input_state(
                product_dir,
                detail_input_fingerprint,
                effective_routing.support_provider,
                effective_routing.detail_provider,
                resume=resume,
            )
            status = read_status(product_dir)
            detail_prompt_path = product_dir / "detail_prompt.txt"
            completion_prompt_hashes = None
            if (
                resume
                and effective_routing.detail_provider == PROVIDER_OPENAI_IMAGE
                and detail_prompt_path.exists()
            ):
                try:
                    saved_detail_prompt = detail_prompt_path.read_text(encoding="utf-8")
                    saved_screens = split_detail_screens(saved_detail_prompt, target_count)
                    saved_request_screens = _compose_detail_request_screens(
                        saved_screens,
                        effective_routing.detail_provider,
                        product=product,
                        image_note=image_note,
                        prompt_settings=prompt_settings,
                        target_count=target_count,
                    )
                    completion_prompt_hashes = {
                        screen.index: detail_screen_prompt_hash(
                            screen,
                            target_count,
                            getattr(product, "image_size", ""),
                        )
                        for screen in saved_request_screens
                    }
                except (OSError, TypeError, ValueError):
                    completion_prompt_hashes = None
            if resume and _is_completed_for_detail_provider(
                product_dir,
                status,
                effective_routing.detail_provider,
                target_count,
                completion_prompt_hashes,
            ):
                status = read_status(product_dir)
                skipped += 1
                project_url = status.get("project_url", "")
                from utils import get_output_dir
                append_result(
                    f"{get_output_dir()}/results.csv",
                    product.id,
                    product.name_cn,
                    project_url,
                    status="success",
                    used_model=str(status.get("used_model") or ""),
                    preserve_existing_model=True,
                )
                logger.info(
                    f"SKIP [{idx}/{len(products)}] {product.id} already completed"
                )
                if project_url:
                    console.print(
                        f"  [green]SKIP[/green] {product.id} already completed: "
                        f"[link={project_url}]{project_url}[/link]"
                    )
                summary_rows.append({
                    "product_id": product.id,
                    "product_name": product.name_cn,
                    "status": "skipped",
                    "project_url": project_url,
                    "gemini_chars": status.get("gemini_chars", ""),
                    "artifact_count": status.get("artifact_count", ""),
                    "duration_seconds": 0,
                    "error": "",
                    "used_model": status.get("used_model", ""),
                })
                continue

            detail_prompt = ""
            screens = []
            gemini_chars = 0
            if resume and detail_prompt_path.exists():
                try:
                    detail_prompt = detail_prompt_path.read_text(encoding="utf-8")
                    screens = split_detail_screens(detail_prompt, target_count)
                    gemini_chars = int(read_status(product_dir).get("gemini_chars") or 0)
                    logger.info(f"Detail prompt resumed ({len(detail_prompt)} chars)")
                except (OSError, TypeError, ValueError):
                    detail_prompt = ""
                    screens = []

            if not screens:
                _emit_ui_status(product.id, "prompt", "🧠 正在生成详情提示词")
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
                screens = split_detail_screens(prompt, target_count)
                detail_prompt_settings = dict(prompt_settings)
                detail_prompt_settings["detail_page_count"] = target_count
                detail_prompt = build_detail_prompt(
                    product_name_cn=product.name_cn,
                    language=product.language,
                    selling_points=product.selling_points,
                    generated_prompt=prompt,
                    image_note=image_note,
                    image_size=getattr(product, "image_size", ""),
                    prompt_settings=detail_prompt_settings,
                )
                gemini_chars = len(prompt)
            else:
                _emit_ui_status(product.id, "prompt", "♻️ 正在复用已生成详情提示词")

            detail_prompt_path.write_text(detail_prompt, encoding="utf-8")
            update_status(
                product_dir,
                "detail_prompt_ready",
                detail_prompt_chars=len(detail_prompt),
                gemini_chars=gemini_chars,
            )
            if effective_routing.detail_provider == PROVIDER_LOVART:
                (product_dir / "lovart_prompt.txt").write_text(
                    detail_prompt, encoding="utf-8"
                )
                update_status(
                    product_dir,
                    "lovart_prompt_ready",
                    lovart_prompt_chars=len(detail_prompt),
                )
            logger.info(f"Detail prompt ready ({len(detail_prompt)} chars)")

            if os.environ.get("UI_MODE") == "1":
                if effective_routing.detail_provider == PROVIDER_LOVART:
                    tool_config = getattr(lovart, "tool_config", {}) or {}
                    selected_model = tool_config.get("image_model", "auto")
                else:
                    selected_model = PROVIDER_OPENAI_IMAGE
                print(f"[UI_MODEL] {selected_model}", flush=True)

            request_screens = _compose_detail_request_screens(
                screens,
                effective_routing.detail_provider,
                product=product,
                image_note=image_note,
                prompt_settings=prompt_settings,
                target_count=target_count,
            )
            status_before_detail = read_status(product_dir)
            previous_detail_provider = str(
                status_before_detail.get("detail_provider") or ""
            )
            previous_used_model = str(status_before_detail.get("used_model") or "")
            execution_settings = _detail_execution_settings(detail_provider)
            configured_detail_model = str(
                execution_settings.get("model")
                or execution_settings.get("image_model")
                or ""
            ).strip()
            initial_used_model = configured_detail_model or (
                previous_used_model
                if previous_detail_provider == effective_routing.detail_provider
                else ""
            )
            update_status(
                product_dir,
                "detail_generation_started",
                detail_provider=effective_routing.detail_provider,
                used_model=initial_used_model,
            )
            _emit_ui_status(
                product.id,
                "detail",
                f"🖼️ 正在生成详情图（目标 {target_count} 张）",
            )
            detail_result = detail_provider.generate_detail_set(
                DetailSetRequest(
                    product_id=product.id,
                    product_dir=product_dir,
                    screens=request_screens,
                    image_paths=tuple(lovart_images),
                    image_size=getattr(product, "image_size", ""),
                    target_count=target_count,
                    prompt=detail_prompt,
                    project_id=lovart_project_id,
                    product_name_cn=product.name_cn,
                    language=product.language,
                    selling_points=product.selling_points,
                    input_fingerprint=detail_input_fingerprint,
                    resume=resume,
                    confirmation_advisor=gemini,
                    progress_callback=_emit_ui_detail_progress,
                    status_callback=lambda message: _emit_ui_status(
                        product.id,
                        "detail",
                        message,
                    ),
                )
            )
            raw_result = dict(detail_result.raw_result or {})
            status = read_status(product_dir)
            completed_count = max(0, int(detail_result.completed_count))
            detail_images = list(detail_result.local_paths)
            used_model = str(
                detail_result.used_model
                or raw_result.get("used_model")
                or configured_detail_model
                or (
                    previous_used_model
                    if previous_detail_provider == effective_routing.detail_provider
                    else ""
                )
                or ""
            )
            if effective_routing.detail_provider == PROVIDER_OPENAI_IMAGE:
                detail_complete = bool(
                    detail_result.succeeded
                    and completed_count == target_count
                    and len(detail_images) == target_count
                    and all(is_valid_image_file(path) for path in detail_images)
                )
                artifact_count = completed_count
            else:
                detail_complete = bool(detail_result.succeeded)
                artifact_count = max(0, int(detail_result.artifact_count))
            partial_complete = bool(
                not detail_complete
                and (detail_result.partial_complete or 0 < completed_count < target_count)
            )
            if effective_routing.detail_provider == PROVIDER_LOVART:
                project_id = str(
                    raw_result.get("project_id")
                    or status.get("project_id")
                    or lovart_project_id
                )
                project_url = str(
                    raw_result.get("project_url")
                    or status.get("project_url")
                    or _lovart_project_url(project_id)
                )
            else:
                if effective_routing.support_provider == PROVIDER_LOVART:
                    project_id = str(status.get("project_id") or lovart_project_id)
                    project_url = str(
                        status.get("project_url") or _lovart_project_url(project_id)
                    )
                else:
                    project_id = ""
                    project_url = ""
            update_status(
                product_dir,
                "detail_result_recorded",
                detail_provider=effective_routing.detail_provider,
                detail_images=detail_images,
                detail_completed_count=completed_count,
                detail_failed_indexes=list(detail_result.failed_indexes),
                detail_generation_complete=detail_complete,
                partial_complete=partial_complete,
                artifact_count=artifact_count,
                used_model=used_model,
                lovart_done=(
                    detail_complete
                    if effective_routing.detail_provider == PROVIDER_LOVART
                    else False
                ),
                project_id=project_id,
                project_url=project_url,
                reason=detail_result.error,
            )

            if detail_complete:
                update_status(
                    product_dir,
                    "detail_generation_done",
                    detail_generation_complete=True,
                    partial_complete=False,
                    failed=False,
                    needs_manual_action=False,
                    lovart_still_running=False,
                    reason="",
                )
                result = dict(raw_result)
                result.update(project_id=project_id, used_model=used_model)
                url = _record_success(product, result)
                logger.info(f"OK [{idx}/{len(products)}] {product.id} completed")
                if url:
                    print(f"\n  >>> {url}")
                if os.environ.get("UI_MODE") == "1":
                    print(f"[UI_SUCCESS] {json.dumps({'id': product.id, 'url': url or '', 'used_model': used_model or 'unknown'})}")
                success += 1
                status = read_status(product_dir)
                summary_rows.append({
                    "product_id": product.id,
                    "product_name": product.name_cn,
                    "status": "success",
                    "project_url": url,
                    "gemini_chars": gemini_chars,
                    "artifact_count": artifact_count,
                    "duration_seconds": round(time.time() - started, 2),
                    "error": "",
                    "used_model": used_model or "unknown",
                })
            elif raw_result.get("final_status") == "pending_confirmation":
                logger.warning(f"NEEDS MANUAL ACTION [{idx}/{len(products)}] {product.id}")
                fail += 1
                reason = detail_result.error or "Lovart pending confirmation on all fallback models"
                update_status(
                    product_dir,
                    "needs_manual_action",
                    detail_generation_complete=False,
                    partial_complete=partial_complete,
                    failed=False,
                    needs_manual_action=True,
                    lovart_still_running=False,
                    reason=reason,
                    project_url=project_url,
                )
                _record_failure(
                    product,
                    "needs_manual_action",
                    reason,
                    project_url,
                    used_model=used_model,
                )
                summary_rows.append({
                    "product_id": product.id,
                    "product_name": product.name_cn,
                    "status": "needs_manual_action",
                    "project_url": project_url,
                    "gemini_chars": gemini_chars,
                    "artifact_count": artifact_count,
                    "duration_seconds": round(time.time() - started, 2),
                    "error": reason,
                    "used_model": used_model,
                })
            elif raw_result.get("final_status") == "timeout":
                logger.warning(f"STILL RUNNING [{idx}/{len(products)}] {product.id}")
                still_running += 1
                if project_url:
                    print(f"\n  Lovart still running in background: {project_url}")
                _record_failure(
                    product,
                    "lovart_still_running",
                    "Lovart still running after local wait timeout",
                    project_url,
                    used_model=used_model,
                )
                update_status(
                    product_dir,
                    "lovart_still_running",
                    detail_generation_complete=False,
                    partial_complete=partial_complete,
                    failed=False,
                    needs_manual_action=False,
                    reason="Lovart still running after local wait timeout",
                    project_url=project_url,
                )
                summary_rows.append({
                    "product_id": product.id,
                    "product_name": product.name_cn,
                    "status": "lovart_still_running",
                    "project_url": project_url,
                    "gemini_chars": gemini_chars,
                    "artifact_count": artifact_count,
                    "duration_seconds": round(time.time() - started, 2),
                    "error": "Lovart still running after local wait timeout",
                    "used_model": used_model,
                })
            else:
                logger.warning(f"WARN [{idx}/{len(products)}] {product.id} failed")
                fail += 1
                reason = str(
                    detail_result.error
                    or raw_result.get("warning")
                    or raw_result.get("final_status")
                    or "Detail image generation failed"
                )
                update_status(
                    product_dir,
                    "failed",
                    reason=reason,
                    project_url=project_url,
                    detail_generation_complete=False,
                    partial_complete=partial_complete,
                    needs_manual_action=False,
                    lovart_still_running=False,
                )
                _record_failure(
                    product,
                    "failed",
                    reason,
                    project_url,
                    used_model=used_model,
                )
                summary_rows.append({
                    "product_id": product.id,
                    "product_name": product.name_cn,
                    "status": "failed",
                    "project_url": project_url,
                    "gemini_chars": gemini_chars,
                    "artifact_count": artifact_count,
                    "duration_seconds": round(time.time() - started, 2),
                    "error": reason,
                    "used_model": used_model,
                    "partial_complete": partial_complete,
                    "detail_completed_count": completed_count,
                    "detail_failed_indexes": list(detail_result.failed_indexes),
                })
        except Exception as exc:
            status = read_status(product_dir)
            project_url = _project_url_from_status(status)
            update_status(product_dir, "failed", reason=str(exc), project_url=project_url)
            logger.error(f"FAIL [{idx}/{len(products)}] {product.id}: {exc}")
            fail += 1
            _record_failure(
                product,
                "failed",
                str(exc),
                project_url,
                used_model=str(status.get("used_model") or ""),
            )
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
        from utils import organize_output_folders
        organize_output_folders()

    write_run_summary(run_dir, summary_rows)
    return success, fail, skipped, still_running


def _read_run_summary(run_dir: str | Path) -> list[dict]:
    try:
        value = json.loads((Path(run_dir) / "summary.json").read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (OSError, ValueError, TypeError):
        return []


def _wait_before_failed_retry(delay: float) -> bool:
    deadline = time.monotonic() + delay
    while not _shutdown_requested:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(remaining, 0.25))
    return False


def _process_products(
    products,
    gemini,
    lovart,
    logger,
    run_dir,
    resume=True,
    prompt_settings=None,
    image_registry=None,
    routing=None,
    failed_retry_policy=None,
):
    policy = failed_retry_policy
    if policy is None:
        policy = getattr(lovart, "failed_retry_policy", FailedRetryPolicy(mode="off"))
    if not isinstance(policy, FailedRetryPolicy) or not policy.enabled:
        return _process_products_once(
            products,
            gemini,
            lovart,
            logger,
            run_dir,
            resume=resume,
            prompt_settings=prompt_settings,
            image_registry=image_registry,
            routing=routing,
        )

    product_by_id = {product.id: product for product in products}
    ordered_ids = list(product_by_id)
    merged_rows: dict[str, dict] = {}
    pending = list(products)

    completed_retry_rounds = 0
    while True:
        _process_products_once(
            pending,
            gemini,
            lovart,
            logger,
            run_dir,
            resume=resume if completed_retry_rounds == 0 else True,
            prompt_settings=prompt_settings,
            image_registry=image_registry,
            routing=routing,
        )
        round_rows = _read_run_summary(run_dir)
        for row in round_rows:
            product_id = str(row.get("product_id") or "")
            if product_id:
                merged_rows[product_id] = row

        retry_ids = []
        for row in round_rows:
            category = classify_retry_failure(row)
            product_id = str(row.get("product_id") or "")
            if category in policy.error_types and product_id in product_by_id:
                retry_ids.append(product_id)

        finite_limit_reached = not policy.infinite and completed_retry_rounds >= policy.rounds
        if not retry_ids or finite_limit_reached or _shutdown_requested:
            break

        pending = [product_by_id[product_id] for product_id in dict.fromkeys(retry_ids)]
        completed_retry_rounds += 1
        round_limit = "∞" if policy.infinite else str(policy.rounds)
        logger.warning(
            f"Retry queue: {len(pending)} product(s) will run again after all current tasks "
            f"(round {completed_retry_rounds}/{round_limit})"
        )
        if _is_ui_mode():
            print(
                f"[UI_PROGRESS] retry round {completed_retry_rounds}/{round_limit} | "
                f"{len(pending)} product(s)",
                flush=True,
            )
        if policy.delay and not _wait_before_failed_retry(policy.delay):
            break

    final_rows = [merged_rows[product_id] for product_id in ordered_ids if product_id in merged_rows]
    write_run_summary(run_dir, final_rows)
    success = sum(row.get("status") == "success" for row in final_rows)
    skipped = sum(row.get("status") == "skipped" for row in final_rows)
    still_running = sum(row.get("status") == "lovart_still_running" for row in final_rows)
    fail = len(final_rows) - success - skipped - still_running
    return success, fail, skipped, still_running


def _build_gemini_api(config, logger, prompt_settings=None):
    api_cfg = config.get("gemini_api", {})
    api_key = env_or_config(api_cfg, "api_key", "GEMINI_API_KEY")
    model = api_cfg.get("model", "gemini-2.5-flash-lite")
    base_url = api_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        sys.exit(1)
    logger.info(f"Using Gemini API: {model}")
    return GeminiAPI(
        api_key=api_key,
        model=model,
        base_url=base_url,
        logger=logger,
        prompt_settings=prompt_settings if prompt_settings is not None else get_prompt_settings(config),
    )


def _build_nvidia_api(config, logger, prompt_settings=None):
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
        prompt_settings=prompt_settings if prompt_settings is not None else get_prompt_settings(config),
    )


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


def _run_browser_flow(
    config,
    products,
    lovart,
    logger,
    run_dir,
    resume=True,
    wait_for_ready=True,
    prompt_settings=None,
    config_path: str | Path = Path("config.yaml"),
    image_registry=None,
    routing=None,
):
    paths = login_runtime_paths(config_path)
    owner = acquire_login_helper_owner(paths)
    if owner is None:
        logger.warning("Gemini 浏览器账户目录正在使用中，未启动正式任务。")
        raise GeminiPageNotReadyError(
            "Gemini 浏览器账户目录正在使用中，请等待当前浏览器任务结束后再试。"
        )
    try:
        return _run_owned_browser_flow(
            config,
            products,
            lovart,
            logger,
            run_dir,
            resume=resume,
            wait_for_ready=wait_for_ready,
            prompt_settings=prompt_settings,
            config_path=config_path,
            image_registry=image_registry,
            routing=routing,
        )
    finally:
        release_login_helper_owner(paths, owner)


def _run_owned_browser_flow(
    config,
    products,
    lovart,
    logger,
    run_dir,
    resume=True,
    wait_for_ready=True,
    prompt_settings=None,
    config_path: str | Path = Path("config.yaml"),
    image_registry=None,
    routing=None,
):
    browser_cfg = config["browser"]
    interactive_console = wait_for_ready and not _is_ui_mode()
    chrome_exe = _resolve_browser_executable_for_run(
        browser_cfg, interactive=interactive_console
    )
    if chrome_exe:
        logger.info(f"Using browser executable: {chrome_exe}")
    else:
        logger.warning("Chrome/Edge executable not found; using Playwright bundled Chromium")

    with sync_playwright() as pw:
        logger.info("Launching browser for Gemini")
        launch_options = build_browser_launch_options(config, config_path=Path(config_path))
        if chrome_exe:
            launch_options["executable_path"] = chrome_exe
        context = pw.chromium.launch_persistent_context(**launch_options)
        try:
            page = context.pages[0] if context.pages else context.new_page()
            policy = retry_policy_from_config(config)
            try:
                status = navigate_gemini_with_retry(
                    page, config["gemini"]["base_url"], policy, logger=logger
                )
            except GeminiPermanentTlsError:
                logger.warning("Gemini TLS 证书验证失败，未开始处理商品。")
                raise GeminiPermanentTlsError() from None
            except Exception as exc:
                kind = classify_network_error(exc)
                if kind is RetryKind.PERMANENT_TLS:
                    logger.warning("Gemini TLS 证书验证失败，未开始处理商品。")
                    raise GeminiPermanentTlsError() from None
                if kind is RetryKind.AUTH:
                    logger.warning("Gemini 身份验证或权限不足，未开始处理商品。")
                    raise GeminiAuthenticationError() from None
                if kind is RetryKind.NOT_FOUND:
                    logger.warning("Gemini 页面或资源不存在，未开始处理商品。")
                    raise GeminiResourceNotFoundError() from None
                logger.warning("Gemini 页面未准备完成，未开始处理商品。")
                raise GeminiPageNotReadyError() from None

            if status.state is GeminiPageState.WAITING_LOGIN:
                logger.warning("Gemini 未登录，未开始处理商品。")
                raise GeminiLoginRequiredError()
            if status.state is not GeminiPageState.READY or not status.ready:
                logger.warning("Gemini 页面未准备完成，未开始处理商品。")
                raise GeminiPageNotReadyError()

            logger.info("Gemini browser ready")
            if interactive_console:
                input("\nReady. Press Enter to start...")
            gemini = GeminiBot(page, config, logger, run_dir=run_dir)
            return _process_products(
                products,
                gemini,
                lovart,
                logger,
                run_dir,
                resume=resume,
                prompt_settings=prompt_settings,
                image_registry=image_registry,
                routing=routing,
                failed_retry_policy=FailedRetryPolicy.from_config(config.get("lovart")),
            )
        finally:
            context.close()


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

    config = load_config(args.config, load_environment=not args.dry_run)
    prompt_settings = get_prompt_settings(config)
    routing = routing_from_config(config, prompt_settings)
    routing = _routing_with_cli_overrides(routing, args)
    uses_lovart = PROVIDER_LOVART in {
        routing.support_provider,
        routing.detail_provider,
    }
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
            print(f"[UI_PRODUCT] {json.dumps({'id': product.id, 'name': product.name_cn, 'image': img})}")

    if args.dry_run:
        success, fail, skipped, still_running = _dry_run_products(products, logger, run_dir)
        print(f"\nDRY-RUN DONE - Parsed: {skipped}, Total: {len(products)}")
        print(f"Run summary: {run_dir}")
        logger.info(f"Dry-run complete. Parsed={skipped}")
        return

    if uses_lovart:
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

    prompt_source = _choose_prompt_source(config, args)
    if uses_lovart:
        fast_mode = _resolve_lovart_mode(args.lovart)
        _choose_lovart_tool_options(config, args)
        config.setdefault("_runtime", {})["lovart_fast_mode"] = fast_mode

    image_registry = _build_image_provider_registry(config, logger)
    lovart = None
    if uses_lovart:
        lovart = image_registry.get(PROVIDER_LOVART).bot
        logger.info(f"Lovart mode: {'fast' if fast_mode else 'unlimited'}")

    signal.signal(signal.SIGINT, _on_sigint)

    if prompt_source == "gemini_api":
        gemini = _build_gemini_api(config, logger, prompt_settings=prompt_settings)
        if (
            args.prompt_source == "ask" or uses_lovart and args.lovart == "ask"
        ) and not _is_ui_mode():
            input("\nReady. Press Enter to start...")
        success, fail, skipped, still_running = _process_products(
            products,
            gemini,
            lovart,
            logger,
            run_dir,
            resume=args.resume,
            prompt_settings=prompt_settings,
            image_registry=image_registry,
            routing=routing,
            failed_retry_policy=FailedRetryPolicy.from_config(config.get("lovart")),
        )
    elif prompt_source == "nvidia":
        prompt_client = _build_nvidia_api(config, logger, prompt_settings=prompt_settings)
        if (
            args.prompt_source == "ask" or uses_lovart and args.lovart == "ask"
        ) and not _is_ui_mode():
            input("\nReady. Press Enter to start...")
        success, fail, skipped, still_running = _process_products(
            products,
            prompt_client,
            lovart,
            logger,
            run_dir,
            resume=args.resume,
            prompt_settings=prompt_settings,
            image_registry=image_registry,
            routing=routing,
            failed_retry_policy=FailedRetryPolicy.from_config(config.get("lovart")),
        )
    else:
        success, fail, skipped, still_running = _run_browser_flow(
            config,
            products,
            lovart,
            logger,
            run_dir,
            resume=args.resume,
            wait_for_ready=(
                args.prompt_source == "ask" or uses_lovart and args.lovart == "ask"
            ),
            prompt_settings=prompt_settings,
            config_path=args.config,
            image_registry=image_registry,
            routing=routing,
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
