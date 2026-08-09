from pathlib import Path
from unittest.mock import Mock

from image_generation import DetailScreen
from openai_image_api import GeneratedImage
from tests.image_test_helpers import write_valid_png


def test_registry_does_not_build_lovart_for_all_openai_run():
    from image_providers import LazyImageProviderRegistry

    lovart_factory = Mock(side_effect=AssertionError("Lovart must stay lazy"))
    openai_factory = Mock(return_value=Mock())
    registry = LazyImageProviderRegistry(lovart_factory, openai_factory)

    assert registry.get("openai_image") is openai_factory.return_value
    lovart_factory.assert_not_called()


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
