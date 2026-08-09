import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from image_generation import GenerationRouting
from image_providers import ImageProviderResult, SupportImageRequest
from tests.image_test_helpers import (
    RecordingImageProvider,
    RecordingRegistry,
    run_product_pipeline,
    write_truncated_png,
    write_valid_png,
)
from utils import is_product_completed, read_status, update_status


def test_cli_provider_overrides_are_parsed():
    from main import parse_args

    args = parse_args([
        "--support-provider", "openai_image",
        "--detail-provider", "lovart",
    ])

    assert args.support_provider == "openai_image"
    assert args.detail_provider == "lovart"


def test_cli_provider_overrides_take_precedence_before_selected_provider_validation(tmp_path):
    import main

    args = SimpleNamespace(
        generate_template=False,
        config="config.yaml",
        limit=None,
        dry_run=False,
        prompt_source="gemini_api",
        gemini=None,
        nvidia_model=None,
        lovart="ask",
        resume=True,
        lovart_image_model=None,
        lovart_model_selection=None,
        lovart_reasoning=None,
        support_provider="openai_image",
        detail_provider="openai_image",
    )
    product = SimpleNamespace(
        id="SKU-CLI",
        name_cn="Product",
        language="English",
        image_size="1:1",
        image_paths=["product.png"],
    )
    config = {
        "image_generation": {
            "support_provider": "lovart",
            "detail_provider": "lovart",
        }
    }

    with (
        patch("main.parse_args", return_value=args),
        patch("main.load_config", return_value=config),
        patch("main.get_prompt_settings", return_value={"detail_page_count": 3}),
        patch("setup_wizard.missing_or_placeholder_env_keys", side_effect=AssertionError("unselected Lovart validated")),
        patch("main.setup_logging", return_value=Mock()),
        patch("main.create_run_dir", return_value=tmp_path / "run"),
        patch("main.read_products", return_value=[product]),
        patch("main._choose_prompt_source", return_value="gemini_api"),
        patch("main._resolve_lovart_mode", side_effect=AssertionError("unselected Lovart configured")),
        patch("main._choose_lovart_tool_options", side_effect=AssertionError("unselected Lovart tools configured")),
        patch("main._build_image_provider_registry", return_value=Mock()),
        patch("main._build_gemini_api", return_value=Mock()),
        patch("main._process_products", return_value=(1, 0, 0, 0)) as process_products,
        patch("main.signal.signal"),
        patch("utils.organize_output_folders"),
    ):
        main.main([])

    routing = process_products.call_args.kwargs["routing"]
    assert routing == GenerationRouting("openai_image", "openai_image", 3)


@pytest.mark.parametrize(
    ("support_provider", "detail_provider"),
    [
        ("lovart", "lovart"),
        ("lovart", "openai_image"),
        ("openai_image", "lovart"),
        ("openai_image", "openai_image"),
    ],
)
def test_main_dry_run_needs_no_credentials_or_external_clients(
    tmp_path, support_provider, detail_provider
):
    import main

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "prompt_settings:\n  detail_page_count: 2\n",
        encoding="utf-8",
    )
    product = SimpleNamespace(
        id="SKU-DRY",
        name_cn="Dry run",
        language="English",
        image_size="1:1",
        image_paths=["product.png"],
    )
    fail = AssertionError
    argv = [
        "--config", str(config_path),
        "--dry-run",
        "--prompt-source", "gemini_browser",
        "--lovart", "fast",
        "--support-provider", support_provider,
        "--detail-provider", detail_provider,
    ]

    with (
        patch.dict(os.environ, {}, clear=True),
        patch("utils.load_dotenv", side_effect=fail("credential environment read")),
        patch("setup_wizard.missing_or_placeholder_env_keys", side_effect=fail("Lovart credential check")),
        patch("main.LovartBot", side_effect=fail("Lovart client construction")),
        patch("main.OpenAIImageAPI", side_effect=fail("GPT client construction")),
        patch("main._build_image_provider_registry", side_effect=fail("provider registry construction")),
        patch("main._build_gemini_api", side_effect=fail("Gemini client construction")),
        patch("main._build_nvidia_api", side_effect=fail("NVIDIA client construction")),
        patch("main._run_browser_flow", side_effect=fail("browser launch")),
        patch("main._choose_prompt_source", side_effect=fail("prompt provider setup")),
        patch("main.setup_logging", return_value=Mock()),
        patch("main.create_run_dir", return_value=tmp_path / "run"),
        patch("main.read_products", return_value=[product]),
        patch("main._dry_run_products", return_value=(0, 0, 1, 0)) as dry_run,
    ):
        main.main(argv)

    dry_run.assert_called_once()


def test_ui_detail_progress_reports_only_counts_and_failed_indexes(tmp_path, capsys):
    with patch.dict(os.environ, {"UI_MODE": "1"}):
        run_product_pipeline(
            tmp_path,
            "openai_image",
            "openai_image",
            detail_count=3,
            fail_indexes={2},
        )

    lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("[UI_DETAIL_PROGRESS]")
    ]
    payloads = [json.loads(line.split("]", 1)[1].strip()) for line in lines]
    assert payloads == [
        {"current": 1, "target": 3, "completed": 1, "failed": []},
        {"current": 2, "target": 3, "completed": 1, "failed": [2]},
    ]
    assert all(set(payload) == {"current", "target", "completed", "failed"} for payload in payloads)


@pytest.mark.parametrize(
    ("support", "detail"),
    [
        ("lovart", "lovart"),
        ("openai_image", "lovart"),
        ("lovart", "openai_image"),
        ("openai_image", "openai_image"),
    ],
)
def test_pipeline_routes_support_stage_independently_of_detail_choice(
    tmp_path, support, detail
):
    run = run_product_pipeline(tmp_path, support, detail)

    assert run.success == 1
    assert run.registry.get_calls == [support, detail]


def test_openai_detail_count_uses_snapshot_not_default_12(tmp_path):
    run = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=3
    )

    assert run.success == 1
    status = read_status(run.product_dir)
    assert status["detail_page_count_snapshot"] == 3
    assert status["detail_completed_count"] == 3
    assert status["detail_generation_complete"] is True
    assert status["partial_complete"] is False
    assert status["artifact_count"] == 3
    assert len(status["detail_images"]) == 3
    assert run.generated_indexes == (1, 2, 3)


def test_single_detail_screen_snapshot_prevents_extra_generation_on_resume(tmp_path):
    first = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=1
    )

    assert (first.success, first.fail, first.skipped) == (1, 0, 0)
    assert first.generated_indexes == (1,)
    assert read_status(first.product_dir)["detail_page_count_snapshot"] == 1

    second = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=9
    )

    assert (second.success, second.fail, second.skipped) == (0, 0, 1)
    assert second.generated_indexes == ()
    status = read_status(second.product_dir)
    assert status["detail_page_count_snapshot"] == 1
    assert status["detail_completed_count"] == 1
    assert len(status["detail_images"]) == 1
    assert status["detail_input_fingerprint"] == read_status(
        first.product_dir
    )["detail_input_fingerprint"]
    second.gemini.generate_prompt.assert_not_called()


def test_screen_count_mismatch_makes_no_paid_image_calls(tmp_path):
    api = Mock()

    run = run_product_pipeline(
        tmp_path,
        "openai_image",
        "openai_image",
        detail_count=4,
        prompt_screen_count=3,
        openai_api=api,
    )

    assert (run.success, run.fail) == (0, 1)
    api.generate_edit.assert_not_called()
    status = read_status(run.product_dir)
    assert status["detail_page_count_snapshot"] == 4
    assert status.get("detail_completed_count", 0) == 0


def test_partial_detail_failure_keeps_completed_images_and_resumes_only_missing(tmp_path):
    first = run_product_pipeline(
        tmp_path,
        "openai_image",
        "openai_image",
        detail_count=3,
        fail_indexes={2},
    )

    assert (first.success, first.fail) == (0, 1)
    first_status = read_status(first.product_dir)
    assert first_status["partial_complete"] is True
    assert first_status["detail_generation_complete"] is False
    assert first_status["detail_completed_count"] == 1
    assert first_status["detail_failed_indexes"] == [2]
    assert first_status["artifact_count"] == 1
    assert len(first_status["detail_images"]) == 1
    assert first.append_result.call_args.kwargs["used_model"] == "gpt-image-2"

    second = run_product_pipeline(
        tmp_path,
        "openai_image",
        "openai_image",
        detail_count=9,
        fail_indexes=set(),
    )

    assert second.generated_indexes == (2, 3)
    second_status = read_status(second.product_dir)
    assert second_status["detail_page_count_snapshot"] == 3
    assert second_status["detail_generation_complete"] is True
    assert second_status["partial_complete"] is False
    assert second_status["detail_completed_count"] == 3
    assert second_status["detail_failed_indexes"] == []
    assert second_status["artifact_count"] == 3
    assert second_status["failed"] is False
    assert second_status["reason"] == ""
    assert [Path(path).stem for path in second_status["detail_images"]] == ["01", "02", "03"]

    skipped = run_product_pipeline(
        tmp_path,
        "openai_image",
        "openai_image",
        detail_count=7,
    )
    assert (skipped.success, skipped.skipped) == (0, 1)
    assert skipped.append_result.call_args.kwargs["used_model"] == "gpt-image-2"
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary[0]["used_model"] == "gpt-image-2"


def test_detail_fingerprint_change_from_support_content_regenerates_prompt_and_set(
    tmp_path,
):
    first = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=2
    )
    first_status = read_status(first.product_dir)
    first_fingerprint = first_status["detail_input_fingerprint"]
    white_path = Path(first_status["white_bg_local_path"])
    Image.new("RGB", (2, 2), (0, 0, 255)).save(white_path, format="PNG")

    second = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=2
    )

    assert (second.success, second.skipped) == (1, 0)
    assert second.generated_indexes == (1, 2)
    second.gemini.generate_prompt.assert_called_once()
    assert read_status(second.product_dir)["detail_input_fingerprint"] != first_fingerprint


def test_support_provider_switch_invalidates_detail_prompt_and_gpt_checkpoints(tmp_path):
    first = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=2
    )
    first_fingerprint = read_status(first.product_dir)["detail_input_fingerprint"]

    switched = run_product_pipeline(
        tmp_path, "lovart", "openai_image", detail_count=2
    )

    assert (switched.success, switched.skipped) == (1, 0)
    assert switched.generated_indexes == (1, 2)
    switched.gemini.generate_prompt.assert_called_once()
    assert read_status(switched.product_dir)["detail_input_fingerprint"] != first_fingerprint


def test_snapshot_is_written_before_support_failure_and_survives_setting_change(tmp_path):
    failed = run_product_pipeline(
        tmp_path,
        "openai_image",
        "openai_image",
        detail_count=3,
        fail_support_steps={"white_bg"},
    )

    assert (failed.success, failed.fail) == (0, 1)
    assert read_status(failed.product_dir)["detail_page_count_snapshot"] == 3
    failed.gemini.generate_prompt.assert_not_called()

    resumed = run_product_pipeline(
        tmp_path,
        "openai_image",
        "openai_image",
        detail_count=9,
    )

    assert resumed.generated_indexes == (1, 2, 3)
    assert read_status(resumed.product_dir)["detail_page_count_snapshot"] == 3


def test_no_resume_regenerates_the_configured_gpt_detail_set(tmp_path):
    first = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=2
    )
    assert first.generated_indexes == (1, 2)

    rerun = run_product_pipeline(
        tmp_path,
        "openai_image",
        "openai_image",
        detail_count=2,
        resume=False,
    )

    assert (rerun.success, rerun.skipped) == (1, 0)
    assert rerun.generated_indexes == (1, 2)
    rerun.gemini.generate_prompt.assert_called_once()


@pytest.mark.parametrize(
    ("changed_setting", "changed_value"),
    [
        ("openai_model", "gpt-image-next"),
        ("openai_resolution", "4K"),
    ],
)
def test_openai_detail_execution_setting_change_invalidates_paid_outputs(
    tmp_path,
    changed_setting,
    changed_value,
):
    first = run_product_pipeline(
        tmp_path,
        "openai_image",
        "openai_image",
        detail_count=2,
    )
    first_fingerprint = read_status(first.product_dir)["detail_input_fingerprint"]

    changed = run_product_pipeline(
        tmp_path,
        "openai_image",
        "openai_image",
        detail_count=2,
        **{changed_setting: changed_value},
    )

    assert (changed.success, changed.skipped) == (1, 0)
    assert changed.generated_indexes == (1, 2)
    changed.gemini.generate_prompt.assert_called_once()
    assert read_status(changed.product_dir)["detail_input_fingerprint"] != first_fingerprint


def test_lovart_detail_model_setting_change_invalidates_prompt_and_completion(tmp_path):
    def lovart_with_model(model):
        bot = Mock()
        bot.tool_config = {
            "image_model": model,
            "image_models": [model],
            "model_selection": "force",
            "prefer_models": None,
            "include_tools": [f"generate_image_{model}"],
            "mode": "thinking",
            "tool_names": [f"generate_image_{model}"],
        }
        bot._fast_mode = True
        bot.create_and_generate.return_value = {
            "generation_succeeded": True,
            "project_id": "detail-project",
            "used_model": model,
        }
        return bot

    first_bot = lovart_with_model("nano_banana_2")
    first = run_product_pipeline(
        tmp_path,
        "openai_image",
        "lovart",
        detail_count=2,
        lovart=first_bot,
    )
    first_fingerprint = read_status(first.product_dir)["detail_input_fingerprint"]

    changed_bot = lovart_with_model("nano_banana_pro")
    changed = run_product_pipeline(
        tmp_path,
        "openai_image",
        "lovart",
        detail_count=2,
        lovart=changed_bot,
    )

    assert (changed.success, changed.skipped) == (1, 0)
    changed_bot.create_and_generate.assert_called_once()
    changed.gemini.generate_prompt.assert_called_once()
    assert read_status(changed.product_dir)["detail_input_fingerprint"] != first_fingerprint


def test_lovart_unlimited_model_order_change_invalidates_detail_reuse(tmp_path):
    def lovart_with_unlimited_models(models):
        bot = Mock()
        bot.tool_config = {
            "image_model": "auto",
            "image_models": [],
            "model_selection": "prefer",
            "prefer_models": None,
            "include_tools": None,
            "mode": None,
            "tool_names": [],
        }
        bot._fast_mode = False
        bot._configured_unlimited_models = tuple(models)
        bot.create_and_generate.return_value = {
            "generation_succeeded": True,
            "project_id": "detail-project",
            "used_model": models[0],
        }
        return bot

    first_bot = lovart_with_unlimited_models(
        ["nano_banana_2", "gpt_image_2"]
    )
    first = run_product_pipeline(
        tmp_path,
        "openai_image",
        "lovart",
        detail_count=2,
        lovart=first_bot,
    )
    first_fingerprint = read_status(first.product_dir)["detail_input_fingerprint"]

    reordered_bot = lovart_with_unlimited_models(
        ["gpt_image_2", "nano_banana_2"]
    )
    reordered = run_product_pipeline(
        tmp_path,
        "openai_image",
        "lovart",
        detail_count=2,
        lovart=reordered_bot,
    )

    assert (reordered.success, reordered.skipped) == (1, 0)
    reordered_bot.create_and_generate.assert_called_once()
    reordered.gemini.generate_prompt.assert_called_once()
    assert read_status(reordered.product_dir)["detail_input_fingerprint"] != first_fingerprint


def test_lovart_fast_mode_ignores_configured_unlimited_model_order(tmp_path):
    def fast_lovart_with_unlimited_models(models):
        bot = Mock()
        bot.tool_config = {
            "image_model": "nano_banana_pro",
            "image_models": ["nano_banana_pro"],
            "model_selection": "force",
            "prefer_models": None,
            "include_tools": ["generate_image_nano_banana_pro"],
            "mode": "thinking",
            "tool_names": ["generate_image_nano_banana_pro"],
        }
        bot._fast_mode = True
        bot._configured_unlimited_models = tuple(models)
        bot.create_and_generate.return_value = {
            "generation_succeeded": True,
            "project_id": "detail-project",
            "used_model": "nano_banana_pro",
        }
        return bot

    first_bot = fast_lovart_with_unlimited_models(
        ["nano_banana_2", "gpt_image_2"]
    )
    first = run_product_pipeline(
        tmp_path,
        "openai_image",
        "lovart",
        detail_count=2,
        lovart=first_bot,
    )
    first_fingerprint = read_status(first.product_dir)["detail_input_fingerprint"]

    reordered_bot = fast_lovart_with_unlimited_models(
        ["gpt_image_2", "nano_banana_2"]
    )
    reordered = run_product_pipeline(
        tmp_path,
        "openai_image",
        "lovart",
        detail_count=2,
        lovart=reordered_bot,
    )

    assert (reordered.success, reordered.skipped) == (0, 1)
    reordered_bot.create_and_generate.assert_not_called()
    reordered.gemini.generate_prompt.assert_not_called()
    assert read_status(reordered.product_dir)["detail_input_fingerprint"] == first_fingerprint


def test_deleted_detail_prompt_reuses_only_identically_regenerated_screen_prompts(tmp_path):
    first = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=2
    )
    (first.product_dir / "detail_prompt.txt").unlink()

    identical = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=2
    )

    assert (identical.success, identical.skipped) == (1, 0)
    identical.gemini.generate_prompt.assert_called_once()
    assert identical.generated_indexes == ()


def test_deleted_detail_prompt_with_changed_plan_regenerates_paid_set(tmp_path):
    first = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=2
    )
    (first.product_dir / "detail_prompt.txt").unlink()

    changed = run_product_pipeline(
        tmp_path,
        "openai_image",
        "openai_image",
        detail_count=2,
        screen_label="Changed screen",
    )

    assert (changed.success, changed.skipped) == (1, 0)
    changed.gemini.generate_prompt.assert_called_once()
    assert changed.generated_indexes == (1, 2)


def test_edited_detail_prompt_content_invalidates_matching_upstream_checkpoints(tmp_path):
    first = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=2
    )
    prompt_path = first.product_dir / "detail_prompt.txt"
    edited = prompt_path.read_text(encoding="utf-8").replace("Screen 2", "Edited screen 2")
    prompt_path.write_text(edited, encoding="utf-8")

    resumed = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=2
    )

    assert (resumed.success, resumed.skipped) == (1, 0)
    resumed.gemini.generate_prompt.assert_not_called()
    assert resumed.generated_indexes == (2,)


def test_completed_openai_detail_set_regenerates_only_corrupt_screen(tmp_path):
    first = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=3
    )
    corrupt_path = Path(read_status(first.product_dir)["detail_images"][1])
    write_truncated_png(corrupt_path)

    second = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=8
    )

    assert (second.success, second.skipped) == (1, 0)
    assert second.generated_indexes == (2,)
    status = read_status(second.product_dir)
    assert status["detail_page_count_snapshot"] == 3
    assert status["detail_completed_count"] == 3
    assert status["detail_generation_complete"] is True


@pytest.mark.parametrize(
    ("raw_result", "expected_counts", "expected_status"),
    [
        (
            {
                "generation_succeeded": False,
                "final_status": "pending_confirmation",
                "project_id": "detail-project",
                "warning": "confirmation required",
            },
            (0, 1, 0, 0),
            "needs_manual_action",
        ),
        (
            {
                "generation_succeeded": False,
                "final_status": "timeout",
                "project_id": "detail-project",
            },
            (0, 0, 0, 1),
            "lovart_still_running",
        ),
    ],
)
def test_lovart_detail_keeps_prompt_and_terminal_state_compatibility(
    tmp_path, raw_result, expected_counts, expected_status
):
    lovart = Mock()
    lovart.create_and_generate.return_value = raw_result

    run = run_product_pipeline(
        tmp_path, "openai_image", "lovart", detail_count=2, lovart=lovart
    )

    assert (run.success, run.fail, run.skipped, run.still_running) == expected_counts
    assert (run.product_dir / "detail_prompt.txt").exists()
    assert (run.product_dir / "lovart_prompt.txt").exists()
    assert lovart.create_and_generate.call_args.kwargs["prompt"] == (
        run.product_dir / "lovart_prompt.txt"
    ).read_text(encoding="utf-8")
    status = read_status(run.product_dir)
    assert status[expected_status] is True
    assert status["project_url"].endswith("projectId=detail-project")


def test_openai_detail_writes_only_provider_neutral_prompt_file_and_summary_model(tmp_path):
    run = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=2
    )

    assert (run.product_dir / "detail_prompt.txt").exists()
    assert not (run.product_dir / "lovart_prompt.txt").exists()
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary[0]["artifact_count"] == 2
    assert summary[0]["used_model"] == "gpt-image-2"


def test_openai_detail_success_clears_stale_lovart_project_link(tmp_path):
    product_dir = tmp_path / "products" / "SKU-ROUTING"
    update_status(
        product_dir,
        "failed",
        project_id="stale-project",
        project_url="https://www.lovart.ai/canvas?projectId=stale-project",
    )

    run = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=2
    )

    status = read_status(run.product_dir)
    assert status["project_id"] == ""
    assert status["project_url"] == ""


def test_lovart_support_openai_detail_preserves_project_for_switch_back(tmp_path):
    first = run_product_pipeline(
        tmp_path, "lovart", "openai_image", detail_count=2
    )

    first_status = read_status(first.product_dir)
    assert first.success == 1
    assert first_status["project_id"] == "project-routing"
    assert first_status["project_url"].endswith("projectId=project-routing")
    assert first_status["lovart_white_bg_local_path"]
    assert first_status["lovart_scene_local_path"]

    switched_lovart = Mock()
    switched_lovart.validate_project.return_value = True
    switched_lovart.create_project.side_effect = AssertionError(
        "valid support project was replaced"
    )
    switched_lovart.create_and_generate.return_value = {
        "generation_succeeded": True,
        "project_id": "project-routing",
        "used_model": "nano_banana_2",
    }
    switched = run_product_pipeline(
        tmp_path,
        "lovart",
        "lovart",
        detail_count=2,
        lovart=switched_lovart,
    )

    assert switched.success == 1
    switched_lovart.create_project.assert_not_called()
    assert switched_lovart.create_and_generate.call_args.kwargs["project_id"] == "project-routing"


def test_legacy_lovart_completion_is_migrated_with_provider_and_snapshot(tmp_path):
    product_dir = tmp_path / "products" / "SKU-ROUTING"
    white = write_valid_png(product_dir / "lovart_steps" / "white_bg" / "white.png")
    scene = write_valid_png(product_dir / "lovart_steps" / "scene" / "scene.png")
    update_status(
        product_dir,
        "lovart_done",
        project_id="legacy-project",
        lovart_white_bg_local_path=white,
        lovart_scene_local_path=scene,
        lovart_final_images=[white, scene],
        used_model="nano_banana_2",
    )

    run = run_product_pipeline(tmp_path, "lovart", "lovart", detail_count=4)

    assert (run.success, run.skipped) == (0, 1)
    status = read_status(product_dir)
    assert status["detail_provider"] == "lovart"
    assert status["detail_page_count_snapshot"] == 4
    assert status["detail_generation_complete"] is True
    assert run.append_result.call_args.kwargs["used_model"] == "nano_banana_2"


def test_lovart_openai_lovart_switch_does_not_reuse_stale_lovart_done(tmp_path):
    product_dir = tmp_path / "products" / "SKU-ROUTING"
    white = write_valid_png(product_dir / "lovart_steps" / "white_bg" / "white.png")
    scene = write_valid_png(product_dir / "lovart_steps" / "scene" / "scene.png")
    update_status(
        product_dir,
        "lovart_done",
        project_id="legacy-project",
        lovart_white_bg_local_path=white,
        lovart_scene_local_path=scene,
        lovart_final_images=[white, scene],
    )

    openai = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=2
    )
    assert openai.success == 1
    assert read_status(product_dir)["lovart_done"] is False

    lovart = run_product_pipeline(tmp_path, "lovart", "lovart", detail_count=9)

    assert (lovart.success, lovart.skipped) == (1, 0)
    lovart_bot = lovart.registry.providers["lovart"].bot
    lovart_bot.create_and_generate.assert_called_once()
    status = read_status(product_dir)
    assert status["detail_provider"] == "lovart"
    assert status["detail_page_count_snapshot"] == 2


def test_provider_change_to_pending_lovart_clears_openai_artifacts_and_reports_model(tmp_path):
    first = run_product_pipeline(
        tmp_path, "openai_image", "openai_image", detail_count=2
    )
    assert first.success == 1

    lovart = Mock()
    lovart.create_and_generate.return_value = {
        "generation_succeeded": False,
        "final_status": "pending_confirmation",
        "project_id": "lovart-project",
        "used_model": "nano_banana_2",
        "warning": "confirmation required",
    }
    pending = run_product_pipeline(
        tmp_path,
        "openai_image",
        "lovart",
        detail_count=8,
        lovart=lovart,
    )

    assert (pending.success, pending.fail) == (0, 1)
    status = read_status(pending.product_dir)
    assert status["detail_completed_count"] == 0
    assert status["detail_images"] == []
    assert status["artifact_count"] == 0
    assert status["partial_complete"] is False
    assert status["used_model"] == "nano_banana_2"
    assert pending.append_result.call_args.kwargs["used_model"] == "nano_banana_2"


def test_openai_support_never_validates_or_creates_lovart_project(tmp_path):
    lovart = Mock()
    lovart.create_project.side_effect = AssertionError("unexpected Lovart project")
    lovart.validate_project.side_effect = AssertionError("unexpected Lovart validation")
    lovart.create_and_generate.return_value = {
        "generation_succeeded": True,
        "project_id": "detail-project",
    }

    run = run_product_pipeline(
        tmp_path, "openai_image", "lovart", lovart=lovart
    )

    assert run.success == 1
    lovart.create_project.assert_not_called()
    lovart.validate_project.assert_not_called()
    status = read_status(run.product_dir)
    assert status["white_bg_local_path"]
    assert status["scene_local_path"]
    assert "lovart_project_created" not in status
    assert "lovart_final_images_ready" not in status
    assert "lovart_white_bg_local_path" not in status
    assert "lovart_scene_local_path" not in status


def test_completed_lovart_product_switching_to_openai_support_is_reprocessed(tmp_path):
    product_dir = tmp_path / "products" / "SKU-ROUTING"
    old_white = write_valid_png(product_dir / "lovart_steps" / "white_bg" / "old.png")
    old_scene = write_valid_png(product_dir / "lovart_steps" / "scene" / "old.png")
    update_status(
        product_dir,
        "lovart_done",
        project_id="old-project",
        white_bg_local_path=old_white,
        white_bg_provider="lovart",
        scene_local_path=old_scene,
        scene_provider="lovart",
        lovart_white_bg_local_path=old_white,
        lovart_scene_local_path=old_scene,
        lovart_final_images=[old_white, old_scene],
    )

    run = run_product_pipeline(tmp_path, "openai_image", "lovart")

    assert (run.success, run.skipped) == (1, 0)
    status = read_status(product_dir)
    assert status["white_bg_provider"] == "openai_image"
    assert status["scene_provider"] == "openai_image"


def test_completed_lovart_product_with_valid_legacy_support_is_skipped(tmp_path):
    product_dir = tmp_path / "products" / "SKU-ROUTING"
    white = write_valid_png(product_dir / "lovart_steps" / "white_bg" / "white.png")
    scene = write_valid_png(product_dir / "lovart_steps" / "scene" / "scene.png")
    update_status(
        product_dir,
        "lovart_done",
        project_id="project-1",
        lovart_white_bg_local_path=white,
        lovart_scene_local_path=scene,
        lovart_final_images=[white, scene],
    )

    run = run_product_pipeline(tmp_path, "lovart", "lovart")

    assert (run.success, run.skipped) == (0, 1)


def test_completed_lovart_product_with_invalid_project_is_reprocessed(tmp_path):
    import main

    product_dir = tmp_path / "products" / "SKU-INVALID-COMPLETED"
    product_image = write_valid_png(product_dir / "product.png")
    old_white = write_valid_png(product_dir / "lovart_steps" / "white_bg" / "old.png")
    old_scene = write_valid_png(product_dir / "lovart_steps" / "scene" / "old.png")
    update_status(
        product_dir,
        "lovart_done",
        project_id="invalid-project",
        lovart_white_bg_local_path=old_white,
        lovart_scene_local_path=old_scene,
        lovart_final_images=[old_white, old_scene],
    )
    new_white = write_valid_png(product_dir / "new-white.png")
    new_scene = write_valid_png(product_dir / "new-scene.png")
    product = SimpleNamespace(
        id="SKU-INVALID-COMPLETED",
        name_cn="Product",
        language="English",
        selling_points="",
        image_size="1:1",
        image_paths=[product_image],
        reference_images_are_product=False,
    )
    gemini = Mock()
    gemini.generate_prompt.return_value = (
        "[[SCREEN 01]]\nHero\n[[/SCREEN 01]]\n"
        "[[SCREEN 02]]\nFeature\n[[/SCREEN 02]]"
    )
    bot = Mock()
    bot.validate_project.return_value = False
    bot.create_project.return_value = "new-project"
    bot.create_support_image.side_effect = [
        {"local_path": new_white},
        {"local_path": new_scene},
    ]
    bot.create_and_generate.return_value = {
        "generation_succeeded": True,
        "project_id": "new-project",
    }

    with (
        patch("main.product_output_dir", return_value=product_dir),
        patch("main._backfill_result_project_urls", return_value=0),
        patch("main.append_result"),
        patch("utils.organize_output_folders"),
    ):
        success, fail, skipped, still_running = main._process_products_once(
            [product],
            gemini,
            bot,
            Mock(),
            tmp_path / "run",
            prompt_settings={"detail_page_count": 2},
        )

    assert (success, fail, skipped, still_running) == (1, 0, 0, 0)
    bot.validate_project.assert_called()
    bot.create_project.assert_called_once_with(product.id, product.name_cn)


def test_building_all_openai_registry_does_not_construct_lovart():
    from main import _build_image_provider_registry

    config = {
        "openai_image": {"model": "gpt-image-2"},
        "lovart": {"api_key": "must-not-be-read"},
    }
    with patch.dict(os.environ, {"OPENAI_IMAGE_API_KEY": "test-key"}, clear=True), patch("main.LovartBot") as lovart_bot:
        registry = _build_image_provider_registry(config, Mock())
        provider = registry.get("openai_image")

    assert provider is not None
    lovart_bot.assert_not_called()


def test_openai_registry_ignores_legacy_yaml_key_and_reports_missing_env_key(capsys):
    from main import _build_image_provider_registry
    from openai_image_api import OpenAIImageAPIError

    legacy_secret = "legacy-" + "sentinel-secret"
    config = {
        "openai_image": {
            "api_key": legacy_secret,
            "base_url": "https://hapiopen.cc/v1",
            "model": "gpt-image-2",
        }
    }

    with patch.dict(os.environ, {}, clear=True):
        registry = _build_image_provider_registry(config, Mock())
        with pytest.raises(OpenAIImageAPIError, match="密钥") as caught:
            registry.get("openai_image")

    captured = capsys.readouterr()
    combined_output = captured.out + captured.err + str(caught.value)
    assert legacy_secret not in combined_output


def test_provider_switch_immediate_gpt_401_reports_current_configured_model(tmp_path):
    from openai_image_api import OpenAIImageAPIError

    product_dir = tmp_path / "products" / "SKU-ROUTING"
    update_status(
        product_dir,
        "failed",
        detail_provider="lovart",
        used_model="nano_banana_2",
    )

    api = Mock()
    api.config = SimpleNamespace(
        base_url="https://images.example/v1",
        model="gpt-image-current",
        resolution="1K",
    )
    api.generate_edit.side_effect = OpenAIImageAPIError(
        "http_401", "GPT Image request was rejected."
    )
    run = run_product_pipeline(
        tmp_path,
        "openai_image",
        "openai_image",
        detail_count=2,
        openai_api=api,
    )

    assert (run.success, run.fail) == (0, 1)
    assert read_status(product_dir)["used_model"] == "gpt-image-current"
    assert run.append_result.call_args.kwargs["used_model"] == "gpt-image-current"


@pytest.mark.parametrize(("fail_indexes", "expected_counts"), [
    (set(), (1, 0)),
    ({2}, (0, 1)),
])
def test_gpt_success_and_partial_failure_retain_configured_model(
    tmp_path, fail_indexes, expected_counts
):
    run = run_product_pipeline(
        tmp_path,
        "openai_image",
        "openai_image",
        detail_count=2,
        fail_indexes=fail_indexes,
        openai_model="gpt-image-current",
    )

    assert (run.success, run.fail) == expected_counts
    assert read_status(run.product_dir)["used_model"] == "gpt-image-current"
    assert run.append_result.call_args.kwargs["used_model"] == "gpt-image-current"


def test_all_openai_main_does_not_enter_lovart_setup(tmp_path):
    import main

    args = SimpleNamespace(
        generate_template=False,
        config="config.yaml",
        limit=None,
        dry_run=False,
        prompt_source="gemini_api",
        gemini=None,
        nvidia_model=None,
        lovart="ask",
        resume=True,
        lovart_image_model=None,
        lovart_model_selection=None,
        lovart_reasoning=None,
    )
    product = SimpleNamespace(
        id="SKU-1",
        name_cn="Product",
        language="English",
        image_size="1:1",
        image_paths=["product.png"],
    )
    config = {
        "image_generation": {
            "support_provider": "openai_image",
            "detail_provider": "openai_image",
        }
    }
    logger = Mock()

    with (
        patch("main.parse_args", return_value=args),
        patch(
            "setup_wizard.missing_or_placeholder_env_keys",
            side_effect=AssertionError("Lovart credentials read"),
        ),
        patch("main.load_config", return_value=config),
        patch("main.get_prompt_settings", return_value={"detail_page_count": 2}),
        patch("main.setup_logging", return_value=logger),
        patch("main.create_run_dir", return_value=tmp_path / "run"),
        patch("main.read_products", return_value=[product]),
        patch("main._choose_prompt_source", return_value="gemini_api"),
        patch("main._resolve_lovart_mode", side_effect=AssertionError("Lovart mode setup")),
        patch("main._choose_lovart_tool_options", side_effect=AssertionError("Lovart tools setup")),
        patch("main.LovartBot", side_effect=AssertionError("Lovart construction")),
        patch("main._build_gemini_api", return_value=Mock()),
        patch("main._process_products", return_value=(1, 0, 0, 0)) as process_products,
        patch("main.signal.signal"),
        patch("utils.organize_output_folders"),
    ):
        main.main([])

    assert process_products.call_args.args[2] is None
    assert process_products.call_args.kwargs["routing"].support_provider == "openai_image"


def test_failed_white_support_stops_before_scene_generation(tmp_path):
    from main import _generate_support_images

    provider = Mock()
    provider.generate_support_image.return_value = ImageProviderResult(
        succeeded=False,
        error="white support failed",
    )
    product = Mock(
        id="SKU-FAIL",
        name_cn="Failed support",
        language="English",
        selling_points="",
        image_size="1:1",
        image_paths=["product.png"],
    )

    with pytest.raises(RuntimeError, match="white support failed"):
        _generate_support_images(
            product,
            tmp_path,
            provider,
            {},
            {},
        )

    assert provider.generate_support_image.call_count == 1


def test_failed_support_stops_before_prompt_and_detail_stages(tmp_path):
    import main

    product_dir = tmp_path / "products" / "SKU-FAIL"
    product_image = write_valid_png(product_dir / "product.png")
    product = SimpleNamespace(
        id="SKU-FAIL",
        name_cn="Product",
        language="English",
        selling_points="",
        image_size="1:1",
        image_paths=[product_image],
        reference_images_are_product=False,
    )
    support = RecordingImageProvider("openai_image")
    support.generate_support_image = Mock(
        return_value=ImageProviderResult(succeeded=False, error="support failed")
    )
    registry = RecordingRegistry(RecordingImageProvider("lovart"), support)
    gemini = Mock()
    lovart = Mock()
    lovart.create_and_generate.side_effect = AssertionError("detail stage reached")

    with (
        patch("main.product_output_dir", return_value=product_dir),
        patch("main._backfill_result_project_urls", return_value=0),
        patch("main.append_result"),
        patch("utils.organize_output_folders"),
    ):
        result = main._process_products_once(
            [product],
            gemini,
            lovart,
            Mock(),
            tmp_path / "run",
            image_registry=registry,
            routing=GenerationRouting("openai_image", "lovart", 2),
        )

    assert result == (0, 1, 0, 0)
    assert registry.get_calls == ["openai_image"]
    gemini.generate_prompt.assert_not_called()
    lovart.create_and_generate.assert_not_called()


def test_lovart_provider_restarts_invalid_resume_project(tmp_path):
    from image_providers import LovartImageProvider

    update_status(
        tmp_path,
        "failed",
        project_id="old-project",
        project_url="https://www.lovart.ai/canvas?projectId=old-project",
        lovart_done=True,
    )
    bot = Mock()
    bot.validate_project.return_value = False
    bot.create_project.return_value = "new-project"
    provider = LovartImageProvider(bot, logger=Mock())
    product = Mock(id="SKU-1", name_cn="Product")

    restart = provider.prepare_support_images(
        product,
        tmp_path,
        read_status(tmp_path),
    )

    assert restart is True
    bot.validate_project.assert_called_once_with("old-project")
    bot.create_project.assert_called_once_with("SKU-1", "Product")
    assert read_status(tmp_path)["project_id"] == "new-project"
    assert is_product_completed(tmp_path) is False


@pytest.mark.parametrize(
    "failed_result",
    [
        {"generation_succeeded": False, "final_status": "timeout"},
        None,
    ],
)
def test_failed_lovart_restart_does_not_reuse_old_support_path(tmp_path, failed_result):
    from image_providers import LovartImageProvider

    old_white = write_valid_png(tmp_path / "lovart_steps" / "white_bg" / "old.png")
    update_status(
        tmp_path,
        "failed",
        project_id="old-project",
        lovart_final_images=[old_white],
    )
    bot = Mock()
    bot.validate_project.return_value = False
    bot.create_project.return_value = "new-project"
    bot.create_support_image.return_value = failed_result
    provider = LovartImageProvider(bot, logger=Mock())
    product = Mock(id="SKU-1", name_cn="Product")
    provider.prepare_support_images(product, tmp_path, read_status(tmp_path))

    result = provider.generate_support_image(
        SupportImageRequest(
            product_id="SKU-1",
            product_dir=tmp_path,
            step_name="white_bg",
            prompt="white",
            image_paths=("product.png",),
        )
    )

    assert result.succeeded is False
    assert result.local_paths == ()


def test_openai_support_does_not_resume_lovart_only_artifacts(tmp_path):
    from main import _generate_support_images

    old_white = write_valid_png(tmp_path / "lovart_steps" / "white_bg" / "old.png")
    old_scene = write_valid_png(tmp_path / "lovart_steps" / "scene" / "old.png")
    product_image = write_valid_png(tmp_path / "product.png")
    update_status(
        tmp_path,
        "failed",
        white_bg_local_path=old_white,
        scene_local_path=old_scene,
        lovart_final_images=[old_white, old_scene],
        lovart_white_bg_local_path=old_white,
        lovart_scene_local_path=old_scene,
    )
    product = Mock(
        id="SKU-SWITCH",
        name_cn="Product",
        language="English",
        selling_points="",
        image_size="1:1",
        image_paths=[product_image],
    )
    provider = RecordingImageProvider("openai_image")

    white, scene = _generate_support_images(
        product, tmp_path, provider, {}, read_status(tmp_path)
    )

    assert provider.support_steps == ["white_bg", "scene"]
    assert white != old_white
    assert scene != old_scene


def test_lovart_restart_reuses_only_new_white_after_scene_timeout(tmp_path):
    from main import _SupportImageGenerationError, _generate_support_images
    from image_providers import LovartImageProvider

    old_white = write_valid_png(tmp_path / "lovart_steps" / "white_bg" / "old.png")
    old_scene = write_valid_png(tmp_path / "lovart_steps" / "scene" / "old.png")
    product_image = write_valid_png(tmp_path / "product.png")
    update_status(
        tmp_path,
        "failed",
        project_id="old-project",
        lovart_done=True,
        white_bg_local_path=old_white,
        scene_local_path=old_scene,
        lovart_final_images=[old_white, old_scene],
        lovart_white_bg_local_path=old_white,
        lovart_scene_local_path=old_scene,
    )
    product = Mock(
        id="SKU-RESTART",
        name_cn="Product",
        language="English",
        selling_points="",
        image_size="1:1",
        image_paths=[product_image],
    )
    new_white = write_valid_png(tmp_path / "new-white.png")
    first_bot = Mock()
    first_bot.validate_project.return_value = False
    first_bot.create_project.return_value = "new-project"
    first_bot.create_support_image.side_effect = [
        {"local_path": new_white},
        {"generation_succeeded": False, "final_status": "timeout"},
    ]

    with pytest.raises(_SupportImageGenerationError):
        _generate_support_images(
            product,
            tmp_path,
            LovartImageProvider(first_bot, logger=Mock()),
            {},
            read_status(tmp_path),
        )
    assert is_product_completed(tmp_path) is False

    new_scene = write_valid_png(tmp_path / "new-scene.png")
    second_bot = Mock()
    second_bot.validate_project.return_value = True
    second_bot.create_support_image.return_value = {"local_path": new_scene}
    second_provider = LovartImageProvider(second_bot, logger=Mock())

    white, scene = _generate_support_images(
        product, tmp_path, second_provider, {}, read_status(tmp_path)
    )

    assert (white, scene) == (new_white, new_scene)
    assert second_bot.create_support_image.call_count == 1
    assert second_bot.create_support_image.call_args.kwargs["step_name"] == "scene"
    assert read_status(tmp_path)["lovart_support_resume_invalidated"] is False


@pytest.mark.parametrize(
    ("resume_source", "invalid_kind"),
    [
        ("generic", "zero_byte"),
        ("generic", "corrupt"),
        ("generic", "truncated"),
        ("generic", "directory"),
        ("lovart_field", "corrupt"),
        ("lovart_final", "truncated"),
        ("lovart_directory", "corrupt"),
    ],
)
def test_lovart_support_resume_regenerates_invalid_images(
    tmp_path, resume_source, invalid_kind
):
    from main import _generate_support_images
    from image_providers import LovartImageProvider

    if resume_source == "lovart_directory":
        invalid_path = tmp_path / "lovart_steps" / "white_bg" / "old.png"
    else:
        invalid_path = tmp_path / "invalid.png"
    if invalid_kind == "directory":
        invalid_path.mkdir(parents=True)
    elif invalid_kind == "zero_byte":
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_path.write_bytes(b"")
    elif invalid_kind == "truncated":
        write_truncated_png(invalid_path)
    else:
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_path.write_text("not an image", encoding="utf-8")

    status_fields = {"project_id": "project-1"}
    if resume_source == "generic":
        status_fields.update(
            white_bg_local_path=str(invalid_path),
            white_bg_provider="lovart",
        )
    elif resume_source == "lovart_field":
        status_fields["lovart_white_bg_local_path"] = str(invalid_path)
    elif resume_source == "lovart_final":
        status_fields["lovart_final_images"] = [str(invalid_path)]
    update_status(tmp_path, "failed", **status_fields)

    product_image = write_valid_png(tmp_path / "product.png")
    new_white = write_valid_png(tmp_path / "new-white.png")
    new_scene = write_valid_png(tmp_path / "new-scene.png")
    product = Mock(
        id="SKU-INVALID-RESUME",
        name_cn="Product",
        language="English",
        selling_points="",
        image_size="1:1",
        image_paths=[product_image],
    )
    bot = Mock()
    bot.validate_project.return_value = True
    bot.create_support_image.side_effect = [
        {"local_path": new_white},
        {"local_path": new_scene},
    ]

    white, scene = _generate_support_images(
        product,
        tmp_path,
        LovartImageProvider(bot, logger=Mock()),
        {},
        read_status(tmp_path),
    )

    assert (white, scene) == (new_white, new_scene)
    assert [
        call.kwargs["step_name"] for call in bot.create_support_image.call_args_list
    ] == ["white_bg", "scene"]
