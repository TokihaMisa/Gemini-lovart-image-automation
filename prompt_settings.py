from __future__ import annotations

from copy import deepcopy
from typing import Mapping


DEFAULT_PROMPT_SETTINGS = {
    "detail_page_count": 12,
    "design_style": "温馨感、高级感",
    "required_sections": ["主标题", "副标题", "信息布局", "排版形式"],
    "image_quality": "1K",
    "logo_policy": "不出现 Logo",
    "copy_style": "适合跨境电商，具体、不空泛",
    "copy_detail_level": "详细",
    "product_fidelity": "严格还原",
    "white_background_requirements": "白底、超清摄影、突出高级感，产品造型与原图一致",
    "scene_requirements": "重新设计场景，产品特征与原图保持一致，超清摄影",
    "allow_questions": False,
    "default_language": "巴西葡萄牙语",
    "missing_image_size_policy": "不使用默认固定图片比例",
    "extra_requirements": "",
}

LOCKED_PROMPT_RULES = (
    "所有提示词生成模型只输出文字设计提示词，可交给 Lovart 使用，不直接生成图片。",
    "Excel 已提供的商品名、语言、图片尺寸/比例、卖点和参考图属性优先，软件设置不得覆盖。",
    "商品图片角色、上传顺序和参考图属性由程序与 Excel 决定。",
    "不得改变商品真实形态，不得虚构不存在的部件、颜色或结构。",
    "Lovart 付费确认与安全规则不可由提示词设置修改。",
    "最终图片只能在 Lovart 阶段生成；额外要求与本规则冲突时忽略额外要求。",
)

_TEXT_LIMITS = {
    "design_style": 500,
    "image_quality": 100,
    "logo_policy": 300,
    "copy_style": 500,
    "copy_detail_level": 100,
    "product_fidelity": 500,
    "white_background_requirements": 2000,
    "scene_requirements": 2000,
    "default_language": 100,
    "missing_image_size_policy": 500,
    "extra_requirements": 5000,
}


class PromptSettingsError(ValueError):
    pass


def _clean_text(name: str, value: object) -> str:
    text = str(value or "").strip()
    limit = _TEXT_LIMITS[name]
    if len(text) > limit:
        raise PromptSettingsError(f"{name} 不能超过 {limit} 个字符")
    return text


def normalize_prompt_settings(raw: Mapping[str, object] | None) -> dict[str, object]:
    source = dict(raw or {})
    result = deepcopy(DEFAULT_PROMPT_SETTINGS)
    result.update({key: value for key, value in source.items() if key in result})

    try:
        page_count = int(result["detail_page_count"])
    except (TypeError, ValueError) as exc:
        raise PromptSettingsError("detail_page_count 必须是 1-50 的整数") from exc
    if not 1 <= page_count <= 50:
        raise PromptSettingsError("detail_page_count 必须是 1-50 的整数")
    result["detail_page_count"] = page_count

    sections = result.get("required_sections", [])
    if isinstance(sections, str):
        sections = sections.replace("，", ",").split(",")
    if not isinstance(sections, (list, tuple)):
        raise PromptSettingsError("required_sections 必须是文本列表")
    cleaned_sections = []
    for item in sections:
        text = str(item or "").strip()
        if text and text not in cleaned_sections:
            cleaned_sections.append(text)
    if not cleaned_sections:
        raise PromptSettingsError("required_sections 至少需要一项")
    if sum(len(item) for item in cleaned_sections) > 1000:
        raise PromptSettingsError("required_sections 内容过长")
    result["required_sections"] = cleaned_sections

    for name in _TEXT_LIMITS:
        result[name] = _clean_text(name, result.get(name))
    result["allow_questions"] = bool(result.get("allow_questions", False))
    return result


def get_prompt_settings(config: Mapping[str, object]) -> dict[str, object]:
    raw = config.get("prompt_settings", {}) if isinstance(config, Mapping) else {}
    return normalize_prompt_settings(raw if isinstance(raw, Mapping) else {})


def merge_prompt_settings(config: Mapping[str, object], raw: Mapping[str, object]) -> dict[str, object]:
    updated = deepcopy(dict(config))
    updated["prompt_settings"] = normalize_prompt_settings(raw)
    return updated


def locked_rules_text() -> str:
    return "\n".join(f"- {rule}" for rule in LOCKED_PROMPT_RULES)


def effective_rules_preview(settings: Mapping[str, object]) -> str:
    normalized = normalize_prompt_settings(settings)
    sections = "、".join(normalized["required_sections"])
    editable = (
        f"详情页屏数：{normalized['detail_page_count']}（一屏一张成品图）\n"
        f"整体风格：{normalized['design_style']}\n"
        f"每屏内容：{sections}\n"
        f"图片画质：{normalized['image_quality']}\n"
        f"Logo 规则：{normalized['logo_policy']}\n"
        f"文案要求：{normalized['copy_style']}；{normalized['copy_detail_level']}\n"
        f"产品还原：{normalized['product_fidelity']}\n"
        f"允许反问：{'是' if normalized['allow_questions'] else '否'}"
    )
    return f"【可编辑长期参数】\n{editable}\n\n【锁定规则（不可编辑）】\n{locked_rules_text()}"
