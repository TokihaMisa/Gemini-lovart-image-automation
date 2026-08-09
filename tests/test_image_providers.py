from pathlib import Path
from unittest.mock import Mock

from PIL import Image

from image_generation import DetailScreen
from openai_image_api import GeneratedImage
from tests.image_test_helpers import write_truncated_png, write_valid_png


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
    from image_providers import DetailSetRequest, OpenAIImageProvider, record_detail_checkpoint

    first_path = write_valid_png(tmp_path / "gpt_image" / "detail" / "01.png")
    record_detail_checkpoint(tmp_path, 1, "done", first_path)
    api = Mock()
    second_path = write_valid_png(tmp_path / "gpt_image" / "detail" / "02.png")
    api.generate_edit.return_value = GeneratedImage(second_path, "gpt-image-2")
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

    assert result.completed_count == 2
    assert result.local_paths == (first_path, second_path)
    assert api.generate_edit.call_count == 1


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
    api.generate_edit.return_value = GeneratedImage(replacement_path, "gpt-image-2")
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

    first_path = write_valid_png(tmp_path / "gpt_image" / "detail" / "01.png")
    api = Mock()
    api.generate_edit.side_effect = [
        GeneratedImage(first_path, "gpt-image-2"),
        RuntimeError("temporary upstream failure"),
    ]
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
