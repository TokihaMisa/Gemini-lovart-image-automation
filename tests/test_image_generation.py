from pathlib import Path

import pytest

from image_generation import (
    DetailScreenParseError,
    PROVIDER_LOVART,
    PROVIDER_OPENAI_IMAGE,
    compose_detail_image_prompt,
    ensure_detail_page_count_snapshot,
    normalize_image_provider,
    routing_from_config,
    split_detail_screens,
)
from utils import read_status


def test_routing_uses_dynamic_detail_count():
    routing = routing_from_config(
        {"image_generation": {"support_provider": "openai_image", "detail_provider": "lovart"}},
        {"detail_page_count": 8},
    )
    assert routing.support_provider == PROVIDER_OPENAI_IMAGE
    assert routing.detail_provider == PROVIDER_LOVART
    assert routing.detail_page_count == 8


def test_routing_defaults_to_lovart_for_both_stages():
    routing = routing_from_config({}, {"detail_page_count": 4})
    assert routing.support_provider == PROVIDER_LOVART
    assert routing.detail_provider == PROVIDER_LOVART


def test_provider_normalization_keeps_supported_provider_names():
    assert normalize_image_provider(" OPENAI_IMAGE ") == PROVIDER_OPENAI_IMAGE


def test_design_prompt_requires_stable_screen_markers():
    from utils import build_design_prompt

    prompt = build_design_prompt("杯子", "英语", "保温", prompt_settings={"detail_page_count": 3})
    assert "[[SCREEN 01]]" in prompt
    assert "[[/SCREEN 01]]" in prompt
    assert "[[SCREEN 03]]" in prompt


def test_parser_accepts_markers_and_rejects_wrong_count():
    text = "[[SCREEN 01]]\nHero\n[[/SCREEN 01]]\n[[SCREEN 02]]\nFeature\n[[/SCREEN 02]]"
    assert [item.index for item in split_detail_screens(text, 2)] == [1, 2]
    with pytest.raises(DetailScreenParseError):
        split_detail_screens(text, 3)


def test_parser_rejects_duplicate_or_nonsequential_markers():
    duplicate = "[[SCREEN 01]]\nOne\n[[/SCREEN 01]]\n[[SCREEN 01]]\nAgain\n[[/SCREEN 01]]"
    skipped = "[[SCREEN 01]]\nOne\n[[/SCREEN 01]]\n[[SCREEN 03]]\nThree\n[[/SCREEN 03]]"

    with pytest.raises(DetailScreenParseError):
        split_detail_screens(duplicate, 2)
    with pytest.raises(DetailScreenParseError):
        split_detail_screens(skipped, 2)


def test_parser_rejects_an_unclosed_marker_even_when_complete_screens_match_count():
    text = "[[SCREEN 01]]\nOne\n[[/SCREEN 01]]\n[[SCREEN 02]]\nUnclosed"

    with pytest.raises(DetailScreenParseError):
        split_detail_screens(text, 1)


def test_parser_accepts_legacy_sequential_screen_headings():
    text = "屏 01\nHero\n\nSCREEN 02\nFeature"
    screens = split_detail_screens(text, 2)
    assert [(screen.index, screen.prompt) for screen in screens] == [(1, "Hero"), (2, "Feature")]


def test_snapshot_survives_later_global_setting_change(tmp_path: Path):
    assert ensure_detail_page_count_snapshot(tmp_path, 6) == 6
    assert ensure_detail_page_count_snapshot(tmp_path, 9) == 6
    assert read_status(tmp_path)["detail_page_count_snapshot"] == 6


def test_compose_detail_image_prompt_keeps_one_screen_and_product_constraints():
    prompt = compose_detail_image_prompt(
        screen=split_detail_screens("[[SCREEN 01]]\nHero with thermal claim\n[[/SCREEN 01]]", 1)[0],
        product_name_cn="杯子",
        language="英语",
        selling_points="保温 12 小时",
        image_note="参考图是实际产品。\n",
        image_size="4:5",
        prompt_settings={"detail_page_count": 1},
    )

    assert "Hero with thermal claim" in prompt
    assert "杯子" in prompt
    assert "英语" in prompt
    assert "保温 12 小时" in prompt
    assert "4:5" in prompt
    assert "第 1 屏" in prompt
