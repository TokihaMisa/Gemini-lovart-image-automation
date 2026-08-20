from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image

from image_generation import DetailScreen
from openai_image_api import GeneratedImage, ImageTaskSnapshot
from tests.image_test_helpers import write_truncated_png, write_valid_png


def completed_image_task() -> ImageTaskSnapshot:
    return ImageTaskSnapshot(
        task_id="test-task",
        state="success",
        is_final=True,
        task_created_at=0.0,
        result_url="https://cdn.example/result.png",
        result_type="image",
    )


def test_registry_does_not_build_lovart_for_all_openai_run():
    from image_providers import LazyImageProviderRegistry

    lovart_factory = Mock(side_effect=AssertionError("Lovart must stay lazy"))
    openai_factory = Mock(return_value=Mock())
    registry = LazyImageProviderRegistry(lovart_factory, openai_factory)

    assert registry.get("openai_image") is openai_factory.return_value
    lovart_factory.assert_not_called()


def test_registry_reuses_the_same_openai_provider_instance():
    from image_providers import LazyImageProviderRegistry

    constructed = []

    def openai_factory():
        instance = object()
        constructed.append(instance)
        return instance

    registry = LazyImageProviderRegistry(
        lambda: (_ for _ in ()).throw(AssertionError("Lovart must stay lazy")), openai_factory
    )

    assert registry.get("openai_image") is registry.get("openai_image")
    assert len(constructed) == 1


def test_openai_detail_set_skips_valid_completed_indexes(tmp_path: Path):
    from image_providers import (
        DetailSetRequest,
        OpenAIImageProvider,
        detail_screen_prompt_hash,
        record_detail_checkpoint,
    )

    screens = (DetailScreen(1, "hero"), DetailScreen(2, "feature"))
    first_path = write_valid_png(tmp_path / "gpt_image" / "detail" / "01.png")
    record_detail_checkpoint(
        tmp_path,
        1,
        "done",
        first_path,
        prompt_hash=detail_screen_prompt_hash(screens[0], 2, "1:1"),
    )
    api = Mock()
    second_path = str(tmp_path / "gpt_image" / "detail" / "02.png")

    def generate_second(*, output_path, **_kwargs):
        path = write_valid_png(Path(output_path))
        return GeneratedImage(path, "gpt-image-2", completed_image_task())

    api.generate_edit.side_effect = generate_second
    provider = OpenAIImageProvider(api, logger=Mock())
    request = DetailSetRequest(
        product_id="P1",
        product_dir=tmp_path,
        screens=screens,
        image_paths=(write_valid_png(tmp_path / "reference.png"),),
        image_size="1:1",
        target_count=2,
    )

    result = provider.generate_detail_set(request)

    assert result.completed_count == 2
    assert result.local_paths == (first_path, second_path)
    assert api.generate_edit.call_count == 1


def test_openai_detail_resume_rebases_checkpoint_after_product_directory_moves(
    tmp_path: Path,
):
    from types import SimpleNamespace

    from image_providers import (
        DetailSetRequest,
        OpenAIImageProvider,
        detail_screen_prompt_hash,
        record_detail_checkpoint,
    )
    from utils import read_status

    product_dir = tmp_path / "3_处理中" / "SKU-MOVED"
    screen = DetailScreen(1, "hero")
    canonical = write_valid_png(product_dir / "gpt_image" / "detail" / "01.png")
    prompt_hash = detail_screen_prompt_hash(screen, 1, "1:1")
    record_detail_checkpoint(
        product_dir,
        1,
        "done",
        str(tmp_path / "SKU-MOVED" / "gpt_image" / "detail" / "01.png"),
        input_fingerprint="same-inputs",
        prompt_hash=prompt_hash,
    )
    api = Mock()
    api.config = SimpleNamespace(model="gpt-image-2")
    provider = OpenAIImageProvider(api)

    result = provider.generate_detail_set(DetailSetRequest(
        product_id="SKU-MOVED",
        product_dir=product_dir,
        screens=(screen,),
        image_paths=(write_valid_png(product_dir / "reference.png"),),
        image_size="1:1",
        target_count=1,
        input_fingerprint="same-inputs",
    ))

    assert result.succeeded is True
    assert result.local_paths == (canonical,)
    api.generate_edit.assert_not_called()
    checkpoint = read_status(product_dir)["detail_checkpoints"]["1"]
    assert checkpoint["local_path"] == canonical


def test_completed_indexes_ignore_checkpoint_with_invalid_image(tmp_path: Path):
    from image_providers import read_completed_detail_indexes, record_detail_checkpoint

    invalid = tmp_path / "gpt_image" / "detail" / "01.png"
    invalid.parent.mkdir(parents=True)
    invalid.write_text("not an image", encoding="utf-8")
    record_detail_checkpoint(tmp_path, 1, "done", str(invalid))

    assert read_completed_detail_indexes(tmp_path, expected_count=1) == set()


def test_openai_detail_set_regenerates_a_header_valid_truncated_checkpoint(tmp_path: Path):
    from image_providers import DetailSetRequest, OpenAIImageProvider, record_detail_checkpoint

    corrupted_path = write_truncated_png(tmp_path / "gpt_image" / "detail" / "01.png")
    with Image.open(corrupted_path) as header:
        assert header.size == (1, 1)
    record_detail_checkpoint(tmp_path, 1, "done", corrupted_path)
    replacement_path = write_valid_png(tmp_path / "replacement.png")
    api = Mock()
    api.generate_edit.return_value = GeneratedImage(
        replacement_path, "gpt-image-2", completed_image_task()
    )
    request = DetailSetRequest(
        product_id="P1",
        product_dir=tmp_path,
        screens=(DetailScreen(1, "hero"),),
        image_paths=(write_valid_png(tmp_path / "reference.png"),),
        image_size="1:1",
        target_count=1,
    )

    result = OpenAIImageProvider(api).generate_detail_set(request)

    assert result.succeeded is True
    assert result.local_paths == (replacement_path,)
    assert api.generate_edit.call_count == 1


def test_openai_detail_set_keeps_paid_success_when_later_screen_fails(tmp_path: Path):
    from image_providers import DetailSetRequest, OpenAIImageProvider, read_completed_detail_indexes

    first_path = str(tmp_path / "gpt_image" / "detail" / "01.png")
    api = Mock()

    def generate_until_failure(*, output_path, **_kwargs):
        if Path(output_path).stem == "01":
            path = write_valid_png(Path(output_path))
            return GeneratedImage(path, "gpt-image-2", completed_image_task())
        raise RuntimeError("temporary upstream failure")

    api.generate_edit.side_effect = generate_until_failure
    provider = OpenAIImageProvider(api, logger=Mock())
    request = DetailSetRequest(
        product_id="P1",
        product_dir=tmp_path,
        screens=(DetailScreen(1, "hero"), DetailScreen(2, "feature")),
        image_paths=(write_valid_png(tmp_path / "reference.png"),),
        image_size="1:1",
        target_count=2,
    )

    result = provider.generate_detail_set(request)

    assert result.completed_count == 1
    assert result.failed_indexes == (2,)
    assert result.partial_complete is True
    assert read_completed_detail_indexes(tmp_path, expected_count=2) == {1}


def test_openai_detail_set_stops_after_first_exhausted_screen_and_resumes_in_order(
    tmp_path: Path,
):
    from image_providers import DetailSetRequest, OpenAIImageProvider

    reference = write_valid_png(tmp_path / "reference.png")
    first_path = str(tmp_path / "gpt_image" / "detail" / "01.png")
    first_api = Mock()

    def generate_until_exhaustion(*, output_path, **_kwargs):
        index = int(Path(output_path).stem)
        if index == 1:
            path = write_valid_png(Path(output_path))
            return GeneratedImage(path, "gpt-image-2", completed_image_task())
        if index == 2:
            raise RuntimeError("screen 2 exhausted retries")
        raise AssertionError("screen 3 must not be called in the failed run")

    first_api.generate_edit.side_effect = generate_until_exhaustion
    request = DetailSetRequest(
        product_id="P1",
        product_dir=tmp_path,
        screens=(
            DetailScreen(1, "hero"),
            DetailScreen(2, "feature"),
            DetailScreen(3, "proof"),
        ),
        image_paths=(reference,),
        image_size="1:1",
        target_count=3,
        input_fingerprint="inputs-v1",
        resume=True,
    )

    first = OpenAIImageProvider(first_api).generate_detail_set(request)

    assert first_api.generate_edit.call_count == 2
    assert first.completed_count == 1
    assert first.failed_indexes == (2,)

    resumed_api = Mock()

    def finish_missing(*, output_path, **_kwargs):
        path = write_valid_png(Path(output_path))
        return GeneratedImage(path, "gpt-image-2", completed_image_task())

    resumed_api.generate_edit.side_effect = finish_missing
    resumed = OpenAIImageProvider(resumed_api).generate_detail_set(request)

    assert [
        Path(call.kwargs["output_path"]).stem
        for call in resumed_api.generate_edit.call_args_list
    ] == ["02", "03"]
    assert resumed.succeeded is True
    assert resumed.completed_count == 3


def test_openai_detail_set_no_resume_replaces_all_prior_checkpoints(tmp_path: Path):
    from image_providers import DetailSetRequest, OpenAIImageProvider, record_detail_checkpoint

    first_path = write_valid_png(tmp_path / "gpt_image" / "detail" / "01.png")
    second_path = write_valid_png(tmp_path / "gpt_image" / "detail" / "02.png")
    record_detail_checkpoint(
        tmp_path,
        1,
        "done",
        first_path,
        attempts=4,
        input_fingerprint="inputs-v1",
    )
    record_detail_checkpoint(
        tmp_path,
        2,
        "done",
        second_path,
        attempts=3,
        input_fingerprint="inputs-v1",
    )
    api = Mock()

    def regenerate(*, output_path, **_kwargs):
        path = write_valid_png(Path(output_path))
        return GeneratedImage(path, "gpt-image-2", completed_image_task())

    api.generate_edit.side_effect = regenerate
    request = DetailSetRequest(
        product_id="P1",
        product_dir=tmp_path,
        screens=(DetailScreen(1, "hero"), DetailScreen(2, "feature")),
        image_paths=(write_valid_png(tmp_path / "reference.png"),),
        image_size="1:1",
        target_count=2,
        input_fingerprint="inputs-v1",
        resume=False,
    )

    result = OpenAIImageProvider(api).generate_detail_set(request)

    assert api.generate_edit.call_count == 2
    assert result.succeeded is True
    checkpoints = __import__("utils").read_status(tmp_path)["detail_checkpoints"]
    assert checkpoints["1"]["attempts"] == 1
    assert checkpoints["2"]["attempts"] == 1


def test_openai_detail_set_reconciles_only_matching_running_canonical_output(
    tmp_path: Path,
):
    from image_providers import (
        DetailSetRequest,
        OpenAIImageProvider,
        detail_screen_prompt_hash,
        record_detail_checkpoint,
    )
    from utils import read_status

    canonical = write_valid_png(tmp_path / "gpt_image" / "detail" / "01.png")
    screen = DetailScreen(1, "hero")
    record_detail_checkpoint(
        tmp_path,
        1,
        "running",
        attempts=1,
        input_fingerprint="inputs-v1",
        prompt_hash=detail_screen_prompt_hash(screen, 1, "1:1"),
    )
    api = Mock()
    request = DetailSetRequest(
        product_id="P1",
        product_dir=tmp_path,
        screens=(screen,),
        image_paths=(write_valid_png(tmp_path / "reference.png"),),
        image_size="1:1",
        target_count=1,
        input_fingerprint="inputs-v1",
        resume=True,
    )

    result = OpenAIImageProvider(api).generate_detail_set(request)

    api.generate_edit.assert_not_called()
    assert result.succeeded is True
    assert result.local_paths == (canonical,)
    checkpoint = read_status(tmp_path)["detail_checkpoints"]["1"]
    assert checkpoint["state"] == "done"
    assert checkpoint["input_fingerprint"] == "inputs-v1"

    other_product = tmp_path / "other-product"
    other_canonical = write_valid_png(other_product / "gpt_image" / "detail" / "01.png")
    wrong_screen = DetailScreen(1, "hero")
    record_detail_checkpoint(
        other_product,
        1,
        "running",
        other_canonical,
        attempts=1,
        input_fingerprint="wrong-inputs",
        prompt_hash=detail_screen_prompt_hash(wrong_screen, 1, "1:1"),
    )
    wrong_api = Mock()
    wrong_api.generate_edit.return_value = GeneratedImage(
        other_canonical, "gpt-image-2", completed_image_task()
    )
    wrong_request = DetailSetRequest(
        product_id="P2",
        product_dir=other_product,
        screens=(wrong_screen,),
        image_paths=(write_valid_png(other_product / "reference.png"),),
        image_size="1:1",
        target_count=1,
        input_fingerprint="inputs-v2",
        resume=True,
    )

    OpenAIImageProvider(wrong_api).generate_detail_set(wrong_request)

    wrong_api.generate_edit.assert_called_once()


@pytest.mark.parametrize("mismatch_kind", ["input_fingerprint", "prompt_hash"])
def test_new_running_checkpoint_cannot_reconcile_stale_canonical_after_crash(
    tmp_path: Path,
    mismatch_kind: str,
):
    from image_providers import (
        DetailSetRequest,
        OpenAIImageProvider,
        detail_screen_prompt_hash,
        record_detail_checkpoint,
    )
    from utils import read_status

    screen = DetailScreen(1, "current paid prompt")
    current_fingerprint = "current-inputs"
    current_prompt_hash = detail_screen_prompt_hash(screen, 1, "1:1")
    prior_fingerprint = (
        "stale-inputs" if mismatch_kind == "input_fingerprint" else current_fingerprint
    )
    prior_prompt_hash = (
        detail_screen_prompt_hash(DetailScreen(1, "stale paid prompt"), 1, "1:1")
        if mismatch_kind == "prompt_hash"
        else current_prompt_hash
    )
    canonical = Path(
        write_valid_png(tmp_path / "gpt_image" / "detail" / "01.png")
    )
    record_detail_checkpoint(
        tmp_path,
        1,
        "running",
        str(canonical),
        attempts=1,
        input_fingerprint=prior_fingerprint,
        prompt_hash=prior_prompt_hash,
    )
    request = DetailSetRequest(
        product_id="P1",
        product_dir=tmp_path,
        screens=(screen,),
        image_paths=(write_valid_png(tmp_path / "reference.png"),),
        image_size="1:1",
        target_count=1,
        input_fingerprint=current_fingerprint,
        resume=True,
    )
    crashing_api = Mock()
    crashing_api.generate_edit.side_effect = KeyboardInterrupt("crash before API output")

    with pytest.raises(KeyboardInterrupt, match="crash before API output"):
        OpenAIImageProvider(crashing_api).generate_detail_set(request)

    assert not canonical.exists()
    running = read_status(tmp_path)["detail_checkpoints"]["1"]
    assert running["state"] == "running"
    assert running["input_fingerprint"] == current_fingerprint
    assert running["prompt_hash"] == current_prompt_hash

    resumed_api = Mock()

    def save_current(*, output_path, **_kwargs):
        path = write_valid_png(Path(output_path))
        return GeneratedImage(path, "gpt-image-2", completed_image_task())

    resumed_api.generate_edit.side_effect = save_current

    resumed = OpenAIImageProvider(resumed_api).generate_detail_set(request)

    resumed_api.generate_edit.assert_called_once()
    assert resumed.succeeded is True


def test_openai_detail_set_regenerates_legacy_done_checkpoint_without_prompt_hash(
    tmp_path: Path,
):
    from image_providers import DetailSetRequest, OpenAIImageProvider, record_detail_checkpoint

    canonical = write_valid_png(tmp_path / "gpt_image" / "detail" / "01.png")
    record_detail_checkpoint(
        tmp_path,
        1,
        "done",
        canonical,
        input_fingerprint="inputs-v1",
    )
    api = Mock()
    api.generate_edit.return_value = GeneratedImage(
        canonical, "gpt-image-2", completed_image_task()
    )
    request = DetailSetRequest(
        product_id="P1",
        product_dir=tmp_path,
        screens=(DetailScreen(1, "hero"),),
        image_paths=(write_valid_png(tmp_path / "reference.png"),),
        image_size="1:1",
        target_count=1,
        input_fingerprint="inputs-v1",
        resume=True,
    )

    OpenAIImageProvider(api).generate_detail_set(request)

    api.generate_edit.assert_called_once()


def test_openai_detail_set_reuses_only_identical_final_screen_prompt(tmp_path: Path):
    from image_providers import (
        DetailSetRequest,
        OpenAIImageProvider,
        detail_screen_prompt_hash,
        record_detail_checkpoint,
    )

    old_screen = DetailScreen(1, "exact paid prompt")
    canonical = write_valid_png(tmp_path / "gpt_image" / "detail" / "01.png")
    record_detail_checkpoint(
        tmp_path,
        1,
        "done",
        canonical,
        input_fingerprint="inputs-v1",
        prompt_hash=detail_screen_prompt_hash(old_screen, 1, "1:1"),
    )
    reference = write_valid_png(tmp_path / "reference.png")
    identical_api = Mock()
    identical_request = DetailSetRequest(
        product_id="P1",
        product_dir=tmp_path,
        screens=(old_screen,),
        image_paths=(reference,),
        image_size="1:1",
        target_count=1,
        input_fingerprint="inputs-v1",
        resume=True,
    )

    identical = OpenAIImageProvider(identical_api).generate_detail_set(identical_request)

    identical_api.generate_edit.assert_not_called()
    assert identical.succeeded is True

    changed_api = Mock()
    changed_api.generate_edit.return_value = GeneratedImage(
        canonical, "gpt-image-2", completed_image_task()
    )
    changed_request = DetailSetRequest(
        product_id="P1",
        product_dir=tmp_path,
        screens=(DetailScreen(1, "changed paid prompt"),),
        image_paths=(reference,),
        image_size="1:1",
        target_count=1,
        input_fingerprint="inputs-v1",
        resume=True,
    )

    OpenAIImageProvider(changed_api).generate_detail_set(changed_request)

    changed_api.generate_edit.assert_called_once()


def test_detail_execution_settings_are_explicit_and_never_include_openai_key():
    import json

    from image_providers import OpenAIImageProvider
    from openai_image_api import OpenAIImageAPI, OpenAIImageAPIConfig

    config = OpenAIImageAPIConfig(
        api_key="top-secret-key",
        base_url="https://images.example/",
        model="gpt-image-custom",
        resolution="4K",
    )

    settings = OpenAIImageProvider(OpenAIImageAPI(config)).detail_execution_settings()

    assert settings == {
        "base_url": "https://images.example",
        "model": "gpt-image-custom",
        "resolution": "4K",
        "merge_reference_images": False,
    }
    assert "top-secret-key" not in json.dumps(settings)


def test_lovart_detail_execution_settings_match_selected_tool_and_mode():
    from image_providers import LovartImageProvider

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
    bot._configured_unlimited_models = ("nano_banana_2", "gpt_image_2")

    settings = LovartImageProvider(bot).detail_execution_settings()

    assert settings == {
        "image_model": "nano_banana_pro",
        "image_models": ["nano_banana_pro"],
        "model_selection": "force",
        "prefer_models": None,
        "include_tools": ["generate_image_nano_banana_pro"],
        "mode": "thinking",
        "tool_names": ["generate_image_nano_banana_pro"],
        "run_mode": "fast",
        "configured_unlimited_models_selected": False,
    }


def test_lovart_adapter_preserves_pending_confirmation_result(tmp_path: Path):
    from image_providers import LovartImageProvider, SupportImageRequest

    raw_result = {
        "generation_succeeded": False,
        "final_status": "pending_confirmation",
        "warning": "Lovart requires confirmation",
    }
    bot = Mock()
    bot.create_support_image.return_value = raw_result
    provider = LovartImageProvider(bot)

    result = provider.generate_support_image(
        SupportImageRequest(
            product_id="P1",
            product_dir=tmp_path,
            step_name="white_bg",
            prompt="plain white background",
            image_paths=("product.png",),
        )
    )

    assert result.succeeded is False
    assert result.error == "Lovart requires confirmation"
    assert result.raw_result is raw_result
    assert bot.create_support_image.call_args.kwargs == {
        "product_id": "P1",
        "step_name": "white_bg",
        "prompt": "plain white background",
        "image_paths": ["product.png"],
    }


def test_lovart_detail_result_never_falls_back_to_uploaded_support_images(tmp_path: Path):
    from image_providers import DetailSetRequest, LovartImageProvider
    from utils import update_status

    white = write_valid_png(tmp_path / "support" / "white.png")
    scene = write_valid_png(tmp_path / "support" / "scene.png")
    update_status(tmp_path, "lovart_final_images_ready", lovart_final_images=[white, scene])
    bot = Mock()
    bot.create_and_generate.return_value = {
        "generation_succeeded": True,
        "project_id": "project-1",
        "used_model": "nano_banana_2",
    }
    request = DetailSetRequest(
        product_id="P1",
        product_dir=tmp_path,
        screens=(DetailScreen(1, "hero"), DetailScreen(2, "feature")),
        image_paths=(white, scene),
        image_size="1:1",
        target_count=2,
    )

    result = LovartImageProvider(bot).generate_detail_set(request)

    assert result.succeeded is True
    assert result.local_paths == ()
    assert result.artifact_count == 0


def test_lovart_detail_adapter_returns_only_downloaded_detail_artifact_paths(tmp_path: Path):
    from image_providers import DetailSetRequest, LovartImageProvider

    detail = write_valid_png(tmp_path / "lovart" / "detail.png")
    bot = Mock()
    bot.create_and_generate.return_value = {
        "generation_succeeded": True,
        "project_id": "project-1",
        "used_model": "nano_banana_2",
        "artifact_count": 1,
        "downloaded": [
            {"type": "image", "local_path": detail},
            {"type": "video", "local_path": str(tmp_path / "preview.mp4")},
        ],
    }
    request = DetailSetRequest(
        product_id="P1",
        product_dir=tmp_path,
        screens=(DetailScreen(1, "hero"),),
        image_paths=(write_valid_png(tmp_path / "support.png"),),
        image_size="1:1",
        target_count=1,
    )

    result = LovartImageProvider(bot).generate_detail_set(request)

    assert result.local_paths == (detail,)
    assert result.artifact_count == 1
