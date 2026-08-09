from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from prompt_settings import normalize_prompt_settings
from utils import read_status, update_status


PROVIDER_LOVART = "lovart"
PROVIDER_OPENAI_IMAGE = "openai_image"
_SUPPORTED_PROVIDERS = {PROVIDER_LOVART, PROVIDER_OPENAI_IMAGE}


@dataclass(frozen=True)
class GenerationRouting:
    support_provider: str
    detail_provider: str
    detail_page_count: int


@dataclass(frozen=True)
class DetailScreen:
    index: int
    prompt: str


class DetailScreenParseError(ValueError):
    """Raised when a generated detail-page plan cannot be split safely."""


def routing_from_config(
    config: Mapping[str, object] | None,
    prompt_settings: Mapping[str, object] | None,
) -> GenerationRouting:
    image_generation = (config or {}).get("image_generation", {})
    if not isinstance(image_generation, Mapping):
        image_generation = {}
    support_provider = normalize_image_provider(image_generation.get("support_provider"))
    detail_provider = normalize_image_provider(image_generation.get("detail_provider"))
    settings = normalize_prompt_settings(prompt_settings)
    return GenerationRouting(
        support_provider=support_provider,
        detail_provider=detail_provider,
        detail_page_count=settings["detail_page_count"],
    )


def normalize_image_provider(value: object) -> str:
    provider = str(value or PROVIDER_LOVART).strip().lower()
    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported image generation provider: {provider}")
    return provider


def split_detail_screens(text: str, expected_count: int) -> list[DetailScreen]:
    expected_count = normalize_prompt_settings({"detail_page_count": expected_count})["detail_page_count"]
    source = str(text or "")
    if "[[SCREEN" in source or "[[/SCREEN" in source:
        screens = _parse_marked_screens(source)
    else:
        screens = _parse_legacy_screens(source)
    _validate_screen_indexes(screens, expected_count)
    return screens


def _parse_marked_screens(text: str) -> list[DetailScreen]:
    pattern = re.compile(
        r"^\s*\[\[SCREEN\s+(\d{2})\]\]\s*\r?\n(.*?)^\s*\[\[/SCREEN\s+(\d{2})\]\]\s*$",
        re.MULTILINE | re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise DetailScreenParseError("Detail screens must use complete [[SCREEN NN]] markers")
    marker_tokens = re.findall(r"\[\[/?SCREEN\b[^\]]*\]\]", text)
    if len(marker_tokens) != len(matches) * 2:
        raise DetailScreenParseError("Every detail screen marker must have a matching closing marker")
    screens = []
    for match in matches:
        opening_index = int(match.group(1))
        closing_index = int(match.group(3))
        if opening_index != closing_index:
            raise DetailScreenParseError("Detail screen opening and closing marker indexes must match")
        prompt = match.group(2).strip()
        if not prompt:
            raise DetailScreenParseError(f"Detail screen {opening_index} is empty")
        screens.append(DetailScreen(index=opening_index, prompt=prompt))
    return screens


def _parse_legacy_screens(text: str) -> list[DetailScreen]:
    heading = re.compile(r"^\s*(?:屏\s*|SCREEN\s+)(\d{1,2})\s*[:：]?\s*$", re.IGNORECASE | re.MULTILINE)
    matches = list(heading.finditer(text))
    if not matches:
        raise DetailScreenParseError("Detail screens must use [[SCREEN NN]] markers or sequential screen headings")
    screens = []
    for position, match in enumerate(matches):
        next_start = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        prompt = text[match.end():next_start].strip()
        index = int(match.group(1))
        if not prompt:
            raise DetailScreenParseError(f"Detail screen {index} is empty")
        screens.append(DetailScreen(index=index, prompt=prompt))
    return screens


def _validate_screen_indexes(screens: list[DetailScreen], expected_count: int) -> None:
    expected_indexes = list(range(1, expected_count + 1))
    indexes = [screen.index for screen in screens]
    if indexes != expected_indexes:
        raise DetailScreenParseError(
            f"Expected exactly one sequential detail screen for indexes {expected_indexes}, got {indexes}"
        )


def ensure_detail_page_count_snapshot(
    product_dir: str | Path,
    configured_count: int,
    *,
    replace_existing: bool = False,
) -> int:
    status = read_status(product_dir)
    existing = status.get("detail_page_count_snapshot")
    if existing is not None and not replace_existing:
        return normalize_prompt_settings({"detail_page_count": existing})["detail_page_count"]
    count = normalize_prompt_settings({"detail_page_count": configured_count})["detail_page_count"]
    update_status(product_dir, "detail_target_snapshotted", detail_page_count_snapshot=count)
    return count


def build_detail_input_fingerprint(
    *,
    support_provider: str,
    detail_provider: str,
    product_id: object,
    product_name_cn: object,
    language: object,
    selling_points: object,
    image_size: object,
    reference_images_are_product: object,
    prompt_settings: Mapping[str, object] | None,
    target_count: int,
    image_inputs: Mapping[str, Sequence[str | Path]],
) -> str:
    """Hash every deterministic input that can alter the detail prompt or result."""
    settings = normalize_prompt_settings(prompt_settings)
    settings["detail_page_count"] = int(target_count)
    hashed_images = {
        str(role): [_hash_detail_input_file(path) for path in paths]
        for role, paths in sorted(image_inputs.items(), key=lambda item: str(item[0]))
    }
    payload = {
        "schema": 1,
        "providers": {
            "support": normalize_image_provider(support_provider),
            "detail": normalize_image_provider(detail_provider),
        },
        "product": {
            "id": str(product_id or ""),
            "name_cn": str(product_name_cn or ""),
            "language": str(language or ""),
            "selling_points": str(selling_points or ""),
            "image_size": str(image_size or ""),
            "reference_images_are_product": bool(reference_images_are_product),
        },
        "prompt_settings": settings,
        "target_count": int(target_count),
        "image_inputs": hashed_images,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _hash_detail_input_file(path_value: str | Path) -> dict[str, object]:
    path = Path(path_value)
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        return {"state": "unreadable"}
    return {"sha256": digest.hexdigest(), "size": size}


def compose_detail_image_prompt(
    screen: DetailScreen,
    product_name_cn: str,
    language: str,
    selling_points: str,
    image_note: str,
    image_size: str,
    prompt_settings: Mapping[str, object] | None,
) -> str:
    settings = normalize_prompt_settings(prompt_settings)
    output_language = str(language or "").strip() or str(settings["default_language"])
    image_size_instruction = f"图片尺寸/比例：{image_size.strip()}\n" if image_size.strip() else ""
    return (
        f"请生成电商详情页第 {screen.index} 屏的一张最终图片。\n"
        f"产品：{product_name_cn}\n"
        f"图片语言：{output_language}\n"
        f"{image_size_instruction}"
        f"{image_note.strip()}\n"
        "产品主体必须严格贴近参考图片，不得改变真实形态、颜色、材质、比例或结构。\n"
        f"产品信息/卖点：{selling_points}\n\n"
        "本屏设计提示词：\n"
        f"{screen.prompt.strip()}\n"
    )
