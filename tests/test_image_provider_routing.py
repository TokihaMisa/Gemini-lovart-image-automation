from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

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


@pytest.mark.parametrize(
    ("support", "detail", "expected_support"),
    [
        ("lovart", "lovart", "lovart"),
        ("openai_image", "lovart", "openai_image"),
        ("lovart", "openai_image", "lovart"),
        ("openai_image", "openai_image", "openai_image"),
    ],
)
def test_pipeline_routes_support_stage_independently_of_detail_choice(
    tmp_path, support, detail, expected_support
):
    run = run_product_pipeline(tmp_path, support, detail)

    assert run.success == 1
    assert run.registry.get_calls == [expected_support]


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
    gemini.generate_prompt.return_value = "generated prompt"
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
        )

    assert (success, fail, skipped, still_running) == (1, 0, 0, 0)
    bot.validate_project.assert_called()
    bot.create_project.assert_called_once_with(product.id, product.name_cn)


def test_building_all_openai_registry_does_not_construct_lovart():
    from main import _build_image_provider_registry

    config = {
        "openai_image": {"api_key": "test-key"},
        "lovart": {"api_key": "must-not-be-read"},
    }
    with patch("main.LovartBot") as lovart_bot:
        registry = _build_image_provider_registry(config, Mock())
        provider = registry.get("openai_image")

    assert provider is not None
    lovart_bot.assert_not_called()


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
