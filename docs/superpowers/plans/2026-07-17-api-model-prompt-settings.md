# API Model Discovery and Prompt Settings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Gemini/NVIDIA API diagnostics and dynamic model selection, plus persistent structured prompt settings whose locked rules and Excel precedence apply consistently to all prompt-generation paths.

**Architecture:** Add `prompt_settings.py` as the single source of defaults, validation, precedence, and locked-rule previews; add `model_provider.py` as the provider-neutral discovery and multimodal-test layer. Keep `utils.py` prompt builders as the pipeline entry points, inject normalized settings into Gemini API, Gemini browser, NVIDIA, and Lovart stages, and expose pure callback helpers from `webui.py` so the Gradio behavior can be unit tested without launching a server.

**Tech Stack:** Python 3.12+, standard-library `urllib`, dataclasses, PyYAML 6, Gradio 6, unittest/pytest, Playwright (existing browser flow).

## Global Constraints

- Do not introduce OpenSpec or another specification dependency.
- Preserve existing uncommitted changes in `gemini_bot.py` and `tests/test_medium_priority.py`; inspect their diff before editing and merge around them.
- API keys remain in `.env`; never write or echo them to `config.yaml`, logs, status files, exceptions, or model catalog state.
- Configuration precedence is locked rules > Excel product fields > persistent UI settings > program defaults.
- All prompt-generation providers output text prompts only; only Lovart generates final images.
- Excel product name, language, image size/ratio, selling points, image roles, upload order, and reference-image identity cannot be overridden by UI settings.
- Dynamic provider model catalogs are runtime state only; persist only the selected Gemini and NVIDIA model IDs.
- A detail page count means one finished detail image per screen, not multiple design variants.
- Browser mode must remain usable without any API discovery request.
- Keep backward compatibility with `nvidia_api.model_choice` plus `nvidia_api.models` while preferring the new direct `nvidia_api.model` field.
- Do not bump `version.py`, `version.json`, package, upload, push, or publish in this implementation plan; ask separately after verification if the user wants an OTA release.

---

## File Structure

- Create `prompt_settings.py`: defaults, validation, locked rules, previews, and immutable config merge.
- Create `model_provider.py`: provider-neutral model metadata, discovery, filtering, diagnostics, and minimal multimodal tests.
- Modify `utils.py`: consume normalized settings in white-background, scene, design, and Lovart prompt builders.
- Modify `gemini_api.py`: store normalized settings and use them for design prompts.
- Modify `nvidia_api.py`: support direct model IDs, store normalized settings, and use them for design prompts.
- Modify `gemini_bot.py`: store normalized settings and use them for browser design prompts without disturbing current browser fixes.
- Modify `main.py`: resolve settings once, pass them through all image/prompt stages, and construct API clients with settings.
- Modify `webui.py`: atomic config writes, API/model callbacks, model selection, prompt-settings tab, and locked previews.
- Modify `config.example.yaml`: document new prompt settings and direct model fields.
- Modify `README.md`: explain API detection, model testing, persistent settings, and precedence.
- Create `tests/test_prompt_settings.py`: validation, defaults, precedence, locked rules, and prompt composition.
- Create `tests/test_model_provider.py`: discovery, filtering, pagination, status mapping, secret redaction, and multimodal test payloads.
- Create `tests/test_webui_model_settings.py`: callback behavior, persistence, source/model switching, reset semantics, and atomic saves.
- Modify focused existing tests only where constructor signatures or legacy behavior require coverage.

---

### Task 1: Prompt-settings domain and backward-compatible configuration

**Files:**
- Create: `prompt_settings.py`
- Create: `tests/test_prompt_settings.py`

**Interfaces:**
- Produces: `DEFAULT_PROMPT_SETTINGS: dict[str, object]`
- Produces: `LOCKED_PROMPT_RULES: tuple[str, ...]`
- Produces: `PromptSettingsError(ValueError)`
- Produces: `normalize_prompt_settings(raw: Mapping[str, object] | None) -> dict[str, object]`
- Produces: `get_prompt_settings(config: Mapping[str, object]) -> dict[str, object]`
- Produces: `merge_prompt_settings(config: Mapping[str, object], raw: Mapping[str, object]) -> dict[str, object]`
- Produces: `locked_rules_text() -> str`
- Produces: `effective_rules_preview(settings: Mapping[str, object]) -> str`

- [ ] **Step 1: Write failing default, validation, merge, and locked-rule tests**

Create `tests/test_prompt_settings.py` with these tests:

```python
import unittest

from prompt_settings import (
    DEFAULT_PROMPT_SETTINGS,
    LOCKED_PROMPT_RULES,
    PromptSettingsError,
    effective_rules_preview,
    get_prompt_settings,
    locked_rules_text,
    merge_prompt_settings,
    normalize_prompt_settings,
)


class PromptSettingsTests(unittest.TestCase):
    def test_missing_config_uses_independent_defaults(self):
        first = get_prompt_settings({})
        second = get_prompt_settings({})
        self.assertEqual(first["detail_page_count"], 12)
        first["required_sections"].append("changed")
        self.assertNotIn("changed", second["required_sections"])

    def test_normalization_strips_text_and_deduplicates_sections(self):
        settings = normalize_prompt_settings({
            "detail_page_count": "18",
            "design_style": "  极简、高级  ",
            "required_sections": ["主标题", "", "主标题", "规格表"],
            "allow_questions": True,
        })
        self.assertEqual(settings["detail_page_count"], 18)
        self.assertEqual(settings["design_style"], "极简、高级")
        self.assertEqual(settings["required_sections"], ["主标题", "规格表"])
        self.assertTrue(settings["allow_questions"])

    def test_page_count_outside_one_to_fifty_is_rejected(self):
        for value in (0, 51, "not-an-int"):
            with self.subTest(value=value), self.assertRaises(PromptSettingsError):
                normalize_prompt_settings({"detail_page_count": value})

    def test_oversized_extra_requirements_are_rejected(self):
        with self.assertRaises(PromptSettingsError):
            normalize_prompt_settings({"extra_requirements": "x" * 5001})

    def test_merge_returns_new_config_and_preserves_other_sections(self):
        original = {"excel": {"path": "data/products.xlsx"}, "prompt_settings": {"detail_page_count": 9}}
        updated = merge_prompt_settings(original, {"detail_page_count": 16})
        self.assertEqual(updated["prompt_settings"]["detail_page_count"], 16)
        self.assertEqual(updated["excel"], original["excel"])
        self.assertEqual(original["prompt_settings"]["detail_page_count"], 9)

    def test_locked_rules_cover_every_provider_and_excel_precedence(self):
        text = locked_rules_text()
        self.assertIn("所有提示词生成模型", text)
        self.assertIn("只输出文字", text)
        self.assertIn("Excel", text)
        self.assertIn("Lovart", text)
        self.assertGreaterEqual(len(LOCKED_PROMPT_RULES), 6)

    def test_preview_contains_normalized_values_and_locked_rules(self):
        preview = effective_rules_preview({"detail_page_count": 15, "design_style": "自然光"})
        self.assertIn("15", preview)
        self.assertIn("自然光", preview)
        self.assertIn("只输出文字", preview)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```powershell
uv run python -m pytest tests/test_prompt_settings.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'prompt_settings'`.

- [ ] **Step 3: Implement the complete settings domain**

Create `prompt_settings.py` with:

```python
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
    "所有提示词生成模型只输出可交给 Lovart 的文字设计提示词，不直接生成图片。",
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
```

- [ ] **Step 4: Run the focused tests**

Run:

```powershell
uv run python -m pytest tests/test_prompt_settings.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the domain layer**

```powershell
git add prompt_settings.py tests/test_prompt_settings.py
git commit -m "feat: add persistent prompt settings domain"
```

---

### Task 2: Apply prompt settings consistently across every generation path

**Files:**
- Modify: `utils.py:341-460`
- Modify: `gemini_api.py:16-45`
- Modify: `nvidia_api.py:18-55`
- Modify: `gemini_bot.py:17-75`
- Modify: `main.py:420-715, 838-865, 947-985, 1060-1095`
- Modify: `tests/test_prompt_settings.py`
- Modify: `tests/test_nvidia_api.py`
- Modify carefully: `tests/test_medium_priority.py`

**Interfaces:**
- Consumes: Task 1 `get_prompt_settings`, `normalize_prompt_settings`, `locked_rules_text`
- Produces: `build_white_background_prompt(image_size="", prompt_settings=None) -> str`
- Produces: `build_scene_prompt(image_size="", prompt_settings=None) -> str`
- Produces: `build_design_prompt(product_name_cn, language, selling_points, image_size="", prompt_settings=None) -> str`
- Produces: `build_lovart_prompt(product_name_cn, language, selling_points, generated_prompt, image_note="", image_size="", prompt_settings=None) -> str`
- Produces: `GeminiAPI(api_key, model="gemini-2.5-flash-lite", logger=None, prompt_settings=None)`, `NvidiaAPI(api_key, model, base_url=DEFAULT_NVIDIA_BASE_URL, logger=None, send_images=True, prompt_settings=None)` and `GeminiBot.prompt_settings`
- Produces: `_process_products(products, gemini, lovart, logger, run_dir, resume=True, prompt_settings=None)` with one normalized settings object shared by all stages

- [ ] **Step 1: Inspect and preserve the dirty browser changes before editing**

Run:

```powershell
git diff -- gemini_bot.py tests/test_medium_priority.py
```

Expected: review the existing user changes and record the touched methods. Do not restore, overwrite, or reformat unrelated lines.

- [ ] **Step 2: Add failing composition and provider-consistency tests**

Append to `tests/test_prompt_settings.py`:

```python
from utils import (
    build_design_prompt,
    build_lovart_prompt,
    build_scene_prompt,
    build_white_background_prompt,
)


class PromptCompositionTests(unittest.TestCase):
    def setUp(self):
        self.settings = normalize_prompt_settings({
            "detail_page_count": 16,
            "design_style": "极简自然光",
            "required_sections": ["主标题", "规格表"],
            "image_quality": "4K",
            "logo_policy": "不添加新 Logo",
            "copy_style": "简洁可信",
            "copy_detail_level": "充分展开",
            "product_fidelity": "严格保持外观",
            "white_background_requirements": "纯白背景并精修",
            "scene_requirements": "家庭使用场景",
            "allow_questions": False,
            "default_language": "英文",
            "missing_image_size_policy": "不固定比例",
            "extra_requirements": "避免使用夸张促销词",
        })

    def test_design_prompt_combines_settings_and_excel_values(self):
        prompt = build_design_prompt(
            "咖啡机", "德语", "15 bar 压力", image_size="4:5", prompt_settings=self.settings
        )
        for expected in ("16屏", "极简自然光", "主标题", "规格表", "4K", "德语", "4:5", "15 bar 压力"):
            self.assertIn(expected, prompt)
        self.assertNotIn("默认语言：英文", prompt)
        self.assertIn("避免使用夸张促销词", prompt)
        self.assertIn("只输出", prompt)

    def test_excel_empty_values_use_configured_fallbacks(self):
        prompt = build_design_prompt("咖啡机", "", "卖点", image_size="", prompt_settings=self.settings)
        self.assertIn("英文", prompt)
        self.assertIn("不固定比例", prompt)

    def test_support_prompts_use_stage_settings_and_excel_size(self):
        white = build_white_background_prompt("3:4", self.settings)
        scene = build_scene_prompt("3:4", self.settings)
        self.assertIn("纯白背景并精修", white)
        self.assertIn("家庭使用场景", scene)
        self.assertIn("4K", white)
        self.assertIn("3:4", white)
        self.assertNotIn("不固定比例", white)

    def test_lovart_prompt_repeats_page_count_and_locked_rules(self):
        prompt = build_lovart_prompt(
            "咖啡机", "德语", "卖点", "模型生成的逐屏方案",
            image_size="4:5", prompt_settings=self.settings,
        )
        self.assertIn("16", prompt)
        self.assertIn("一屏一张", prompt)
        self.assertIn("模型生成的逐屏方案", prompt)
        self.assertIn("最终图片只能在 Lovart 阶段生成", prompt)
```

Add to `tests/test_nvidia_api.py`:

```python
def test_resolve_nvidia_model_prefers_direct_model_id(self):
    cfg = {
        "model": "nvidia/new-vision-model",
        "model_choice": "kimi",
        "models": {"kimi": "moonshotai/kimi-k2.5"},
    }
    self.assertEqual(resolve_nvidia_model(cfg), "nvidia/new-vision-model")
```

- [ ] **Step 3: Run the new tests and confirm failures**

Run:

```powershell
uv run python -m pytest tests/test_prompt_settings.py tests/test_nvidia_api.py -v
```

Expected: failures show prompt builders do not accept `prompt_settings`, configurable values are absent, and direct NVIDIA model IDs are ignored.

- [ ] **Step 4: Refactor prompt builders around normalized settings**

In `utils.py`, import:

```python
from prompt_settings import locked_rules_text, normalize_prompt_settings
```

Replace the fixed support/design prompt block with functions following this exact structure:

```python
def _effective_image_size_instruction(image_size: str, settings: dict) -> str:
    cleaned = str(image_size or "").strip()
    if cleaned:
        return f"图片尺寸/比例: {cleaned}\n"
    fallback = str(settings["missing_image_size_policy"] or "").strip()
    return f"图片尺寸/比例: {fallback}\n" if fallback else ""


def build_white_background_prompt(image_size: str = "", prompt_settings=None) -> str:
    settings = normalize_prompt_settings(prompt_settings)
    return (
        f"{settings['white_background_requirements']}\n"
        f"图片画质: {settings['image_quality']}\n"
        f"{_effective_image_size_instruction(image_size, settings)}"
    )


def build_scene_prompt(image_size: str = "", prompt_settings=None) -> str:
    settings = normalize_prompt_settings(prompt_settings)
    return (
        f"{settings['scene_requirements']}\n"
        f"图片画质: {settings['image_quality']}\n"
        f"{_effective_image_size_instruction(image_size, settings)}"
    )


def build_design_prompt(product_name_cn, language, selling_points, image_size="", prompt_settings=None) -> str:
    settings = normalize_prompt_settings(prompt_settings)
    output_language = str(language or "").strip() or str(settings["default_language"])
    sections = "、".join(settings["required_sections"])
    question_rule = "允许在信息确实不足时提出一个必要问题。" if settings["allow_questions"] else "请不要反问，直接根据现有信息生成最优提示词。"
    extra = str(settings["extra_requirements"] or "").strip()
    extra_block = f"\n额外要求：\n{extra}\n" if extra else ""
    return (
        f"上传图片是我的{product_name_cn}产品。\n"
        "【角色设定】你是一名资深电商设计师，擅长平面设计、信息层级和文字排版。\n"
        f"请设计一套包含{settings['detail_page_count']}屏的电商详情页；一屏对应一张详情成品图，不是多套设计版本。\n"
        "你当前只负责输出可交给 Lovart 的逐屏文字设计提示词，不要直接生成图片。\n"
        f"整体风格：{settings['design_style']}\n"
        f"每屏必须包含：{sections}\n"
        f"图片画质：{settings['image_quality']}\n"
        f"Logo 规则：{settings['logo_policy']}\n"
        f"文案要求：{settings['copy_style']}；详细程度：{settings['copy_detail_level']}\n"
        f"产品还原：{settings['product_fidelity']}\n"
        f"{_effective_image_size_instruction(image_size, settings)}"
        f"图片语言：{output_language}\n"
        f"{question_rule}\n"
        f"产品信息/卖点：\n{selling_points}\n"
        f"{extra_block}\n"
        f"【锁定规则】\n{locked_rules_text()}\n"
    )
```

Update `build_lovart_prompt` to accept `prompt_settings=None`, normalize it, use the Excel language when present or the configured default when empty, include the page count, one-screen/one-image meaning, configurable fields and `locked_rules_text()` before appending `generated_prompt`.

- [ ] **Step 5: Inject one normalized settings object through clients and main**

Make these exact compatibility-preserving changes:

```python
# gemini_api.py
from prompt_settings import normalize_prompt_settings

def __init__(self, api_key, model="gemini-2.5-flash-lite", logger=None, prompt_settings=None):
    self.api_key = api_key
    self.model = model
    self.logger = logger
    self.prompt_settings = normalize_prompt_settings(prompt_settings)

# generate_prompt build call
build_design_prompt(
    product_name_cn,
    language,
    selling_points,
    image_size=image_size,
    prompt_settings=self.prompt_settings,
)
```

```python
# nvidia_api.py
from prompt_settings import normalize_prompt_settings

def resolve_nvidia_model(cfg: dict) -> str:
    direct = str(cfg.get("model", "") or "").strip()
    if direct:
        return direct
    choice = str(cfg.get("model_choice", "kimi") or "kimi").strip().lower()
    model = cfg.get("models", {}).get(choice)
    if not model:
        raise ValueError(f"Unknown NVIDIA model choice '{choice}'. Configure nvidia_api.model.")
    return str(model)
```

Add `prompt_settings=None` to `NvidiaAPI.__init__`, store `normalize_prompt_settings(prompt_settings)`, and pass it to `build_design_prompt`.

```python
# gemini_bot.py
from prompt_settings import get_prompt_settings

# in __init__
self.prompt_settings = get_prompt_settings(config)

# in generate_prompt
prompt = build_design_prompt(
    product_name_cn, language, selling_points,
    image_size=image_size,
    prompt_settings=self.prompt_settings,
)
```

In `main.py`, import `get_prompt_settings`, add `prompt_settings=None` to `_process_products`, normalize once at its start, and pass it to `build_white_background_prompt`, `build_scene_prompt`, and `build_lovart_prompt`. In `main()`, compute `prompt_settings = get_prompt_settings(config)` and pass it to all `_process_products` calls. Pass it into both API client constructors from `_build_gemini_api` and `_build_nvidia_api`; browser `GeminiBot` reads it from `config`.

- [ ] **Step 6: Run focused prompt and client tests**

Run:

```powershell
uv run python -m pytest tests/test_prompt_settings.py tests/test_high_priority.py tests/test_nvidia_api.py tests/test_medium_priority.py -v
```

Expected: all tests pass, including the pre-existing dirty browser tests.

- [ ] **Step 7: Commit the pipeline integration without unrelated dirty hunks**

Review first:

```powershell
git diff -- utils.py gemini_api.py nvidia_api.py gemini_bot.py main.py tests/test_prompt_settings.py tests/test_nvidia_api.py tests/test_medium_priority.py
```

Stage only implementation hunks belonging to this task. Preserve existing user changes and do not claim them as newly authored.

```powershell
git add utils.py gemini_api.py nvidia_api.py main.py tests/test_prompt_settings.py tests/test_nvidia_api.py
git add -p gemini_bot.py tests/test_medium_priority.py
git commit -m "feat: apply prompt settings across generation pipeline"
```

---

### Task 3: Provider-neutral model discovery and filtering

**Files:**
- Create: `model_provider.py`
- Create: `tests/test_model_provider.py`

**Interfaces:**
- Produces: `DiscoveredModel`
- Produces: `ModelProviderError(code: str, user_message: str, status_code: int | None)`
- Produces: `discover_models(provider: str, api_key: str, base_url: str, timeout: float = 20) -> list[DiscoveredModel]`
- Produces: `model_choice_labels(models: list[DiscoveredModel]) -> list[tuple[str, str]]`

- [ ] **Step 1: Write failing Gemini/NVIDIA discovery tests**

Create `tests/test_model_provider.py` with this discovery test foundation:

```python
import io
import json
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from model_provider import (
    DiscoveredModel,
    ModelProviderError,
    discover_models,
    model_choice_labels,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class ModelDiscoveryTests(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_gemini_discovery_paginates_and_keeps_generate_content_models(self, urlopen):
        urlopen.side_effect = [
            FakeResponse({
                "models": [{
                    "name": "models/gemini-3.5-flash",
                    "displayName": "Gemini 3.5 Flash",
                    "supportedGenerationMethods": ["generateContent"],
                    "thinking": True,
                }],
                "nextPageToken": "page-2",
            }),
            FakeResponse({
                "models": [{
                    "name": "models/gemini-3.5-pro",
                    "displayName": "Gemini 3.5 Pro",
                    "supportedGenerationMethods": ["generateContent"],
                    "thinking": True,
                }]
            }),
        ]
        models = discover_models("gemini", "key", "https://google.test/v1beta")
        self.assertEqual([m.model_id for m in models], ["gemini-3.5-flash", "gemini-3.5-pro"])
        self.assertIn("pageToken=page-2", urlopen.call_args_list[1].args[0].full_url)

    @patch("urllib.request.urlopen")
    def test_gemini_discovery_filters_non_prompt_models(self, urlopen):
        urlopen.return_value = FakeResponse({"models": [
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/imagen-4", "supportedGenerationMethods": ["predict"]},
            {"name": "models/gemini-live", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/veo-3", "supportedGenerationMethods": ["predictLongRunning"]},
            {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash", "supportedGenerationMethods": ["generateContent"]},
        ]})
        models = discover_models("gemini", "key", "https://google.test/v1beta")
        self.assertEqual([m.model_id for m in models], ["gemini-2.5-flash"])

    @patch("urllib.request.urlopen")
    def test_nvidia_discovery_sends_bearer_auth_and_filters_non_chat_models(self, urlopen):
        urlopen.return_value = FakeResponse({"data": [
            {"id": "nvidia/nv-embed-v1"},
            {"id": "black-forest-labs/flux.1"},
            {"id": "moonshotai/kimi-k2.5"},
        ]})
        models = discover_models("nvidia", "super-secret-key", "https://nvidia.test/v1")
        self.assertEqual([m.model_id for m in models], ["moonshotai/kimi-k2.5"])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer super-secret-key")

    @patch("urllib.request.urlopen")
    def test_discovery_rejects_missing_key_before_network_call(self, urlopen):
        with self.assertRaises(ModelProviderError) as ctx:
            discover_models("gemini", "", "https://google.test/v1beta")
        self.assertEqual(ctx.exception.code, "missing_key")
        urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_http_401_maps_to_secret_free_invalid_key_message(self, urlopen):
        urlopen.side_effect = HTTPError(
            "https://google.test", 401, "Unauthorized super-secret-key", {}, io.BytesIO(b"secret")
        )
        with self.assertRaises(ModelProviderError) as ctx:
            discover_models("gemini", "super-secret-key", "https://google.test/v1beta")
        self.assertEqual(ctx.exception.code, "authentication")
        self.assertNotIn("super-secret-key", ctx.exception.user_message)

    @patch("urllib.request.urlopen")
    def test_http_429_maps_to_rate_limit_without_secret(self, urlopen):
        urlopen.side_effect = HTTPError(
            "https://nvidia.test", 429, "Too Many Requests", {}, io.BytesIO(b"super-secret-key")
        )
        with self.assertRaises(ModelProviderError) as ctx:
            discover_models("nvidia", "super-secret-key", "https://nvidia.test/v1")
        self.assertEqual(ctx.exception.code, "rate_limit")
        self.assertNotIn("super-secret-key", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_empty_compatible_list_returns_empty_list(self, urlopen):
        urlopen.return_value = FakeResponse({"data": [{"id": "nvidia/nv-embed-v1"}]})
        self.assertEqual(discover_models("nvidia", "key", "https://nvidia.test/v1"), [])

    def test_model_choice_labels_show_thinking_and_image_status(self):
        models = [DiscoveredModel(
            provider="gemini",
            model_id="gemini-3.5-flash",
            display_name="Gemini 3.5 Flash",
            supports_generation=True,
            supports_thinking=True,
            image_input_status="unknown",
            recommendation="recommended",
        )]
        labels = model_choice_labels(models)
        self.assertEqual(labels[0][1], "gemini-3.5-flash")
        self.assertIn("Thinking", labels[0][0])
        self.assertIn("图片未验证", labels[0][0])
```

- [ ] **Step 2: Run discovery tests and confirm failure**

Run:

```powershell
uv run python -m pytest tests/test_model_provider.py -k "discovery or labels or http" -v
```

Expected: import fails because `model_provider.py` does not exist.

- [ ] **Step 3: Implement discovery, pagination, filters, and error mapping**

Create `model_provider.py` with immutable dataclasses:

```python
@dataclass(frozen=True)
class DiscoveredModel:
    provider: str
    model_id: str
    display_name: str
    supports_generation: bool
    supports_thinking: bool | None
    image_input_status: str
    recommendation: str


class ModelProviderError(RuntimeError):
    def __init__(self, code, user_message, status_code=None):
        super().__init__(user_message)
        self.code = code
        self.user_message = user_message
        self.status_code = status_code
```

Implement:

- `_request_json(url, api_key, provider, method="GET", payload=None, timeout=20)` with Gemini key query authentication and NVIDIA bearer authentication.
- `_map_http_error(provider, exc)` mapping `401/403`, `404`, `429`, timeout/DNS/TLS/network failures to fixed Chinese messages that never include request URLs containing keys or raw response bodies.
- `_discover_gemini` loop over `nextPageToken`, retaining only `generateContent` and excluding IDs containing `embedding`, `imagen`, `veo`, `live`, `tts`, `speech`, or `audio`.
- `_discover_nvidia` parsing `data`, excluding IDs containing `embed`, `rerank`, `retrieval`, `tts`, `speech`, `audio`, `flux`, `stable-diffusion`, `imagen`, or `veo`.
- Recommendation `recommended` for Gemini Flash/Pro text models and the legacy configured Kimi family; otherwise `available`.
- Gemini image status `reported` only when explicit response metadata says image input is supported; otherwise `unknown`. NVIDIA discovery always starts `unknown`.
- `model_choice_labels` returning Gradio-compatible `(label, model_id)` tuples with stable labels such as `Gemini 3.5 Flash · Thinking · 图片未验证`.

- [ ] **Step 4: Run the discovery tests**

Run:

```powershell
uv run python -m pytest tests/test_model_provider.py -k "discovery or labels or http" -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit model discovery**

```powershell
git add model_provider.py tests/test_model_provider.py
git commit -m "feat: discover Gemini and NVIDIA models"
```

---

### Task 4: Minimal multimodal model compatibility test

**Files:**
- Modify: `model_provider.py`
- Modify: `tests/test_model_provider.py`

**Interfaces:**
- Consumes: Task 3 request and error mapping
- Produces: `ModelTestResult(ok: bool, message: str, latency_ms: int)`
- Produces: `test_selected_model(provider: str, api_key: str, base_url: str, model_id: str, timeout: float = 30) -> ModelTestResult`

- [ ] **Step 1: Add failing payload and response tests**

Extend the imports with `ModelTestResult` and `test_selected_model`, then add:

```python
class ModelCompatibilityTests(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_gemini_model_test_sends_inline_png_and_small_output_limit(self, urlopen):
        urlopen.return_value = FakeResponse({"candidates": [{"content": {"parts": [{"text": "OK"}]}}]})
        result = test_selected_model("gemini", "key", "https://google.test/v1beta", "gemini-2.5-flash")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        parts = payload["contents"][0]["parts"]
        self.assertTrue(result.ok)
        self.assertTrue(request.full_url.startswith("https://google.test/v1beta/models/gemini-2.5-flash:generateContent"))
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 32)
        self.assertEqual(parts[1]["inline_data"]["mime_type"], "image/png")

    @patch("urllib.request.urlopen")
    def test_nvidia_model_test_sends_data_url_and_small_output_limit(self, urlopen):
        urlopen.return_value = FakeResponse({"choices": [{"message": {"content": "OK"}}]})
        result = test_selected_model("nvidia", "key", "https://nvidia.test/v1", "moonshotai/kimi-k2.5")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        content = payload["messages"][0]["content"]
        self.assertTrue(result.ok)
        self.assertEqual(request.full_url, "https://nvidia.test/v1/chat/completions")
        self.assertEqual(payload["max_tokens"], 32)
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    @patch("urllib.request.urlopen")
    def test_model_test_requires_nonempty_text(self, urlopen):
        urlopen.return_value = FakeResponse({"choices": [{"message": {"content": ""}}]})
        with self.assertRaises(ModelProviderError) as ctx:
            test_selected_model("nvidia", "key", "https://nvidia.test/v1", "model")
        self.assertEqual(ctx.exception.code, "empty_response")

    @patch("urllib.request.urlopen")
    def test_model_test_reports_latency_and_success_message(self, urlopen):
        urlopen.return_value = FakeResponse({"choices": [{"message": {"content": "OK"}}]})
        result = test_selected_model("nvidia", "key", "https://nvidia.test/v1", "model")
        self.assertIsInstance(result, ModelTestResult)
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertIn("支持图片输入", result.message)

    @patch("urllib.request.urlopen")
    def test_model_test_error_does_not_echo_api_key(self, urlopen):
        urlopen.side_effect = HTTPError(
            "https://nvidia.test", 403, "super-secret-key", {}, io.BytesIO(b"super-secret-key")
        )
        with self.assertRaises(ModelProviderError) as ctx:
            test_selected_model("nvidia", "super-secret-key", "https://nvidia.test/v1", "model")
        self.assertNotIn("super-secret-key", ctx.exception.user_message)
```

The Gemini request targets `{base_url}/models/{model_id}:generateContent`; the NVIDIA request targets `{base_url}/chat/completions`.

- [ ] **Step 2: Run and confirm the new tests fail**

```powershell
uv run python -m pytest tests/test_model_provider.py -k "model_test" -v
```

Expected: failures because `test_selected_model` and `ModelTestResult` do not exist.

- [ ] **Step 3: Implement the minimal compatibility probe**

Add to `model_provider.py`:

```python
@dataclass(frozen=True)
class ModelTestResult:
    ok: bool
    message: str
    latency_ms: int


_TEST_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "/x8AAusB9Y9Z4WQAAAAASUVORK5CYII="
)
```

Implement provider-specific payload builders and extractors. Use `time.perf_counter()` around one request, require non-empty returned text, and return `ModelTestResult(True, "API 与所选模型可用，支持图片输入并返回文字", elapsed_ms)`. Convert an empty response to `ModelProviderError("empty_response", "模型接受了请求，但没有返回可用文字")`. Do not retry the button test automatically, so one click creates at most one billable request.

- [ ] **Step 4: Run the full provider suite**

```powershell
uv run python -m pytest tests/test_model_provider.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit compatibility testing**

```powershell
git add model_provider.py tests/test_model_provider.py
git commit -m "feat: test selected models with multimodal probe"
```

---

### Task 5: WebUI API diagnostics, runtime catalogs, and persistent model selection

**Files:**
- Modify: `webui.py:1-150, 705-855`
- Create: `tests/test_webui_model_settings.py`

**Interfaces:**
- Consumes: `discover_models`, `model_choice_labels`, `test_selected_model`
- Produces: `refresh_provider_models(provider, api_key, base_url, current_model) -> tuple[str, list[tuple[str, str]], str, list[dict]]`
- Produces: `test_provider_model(provider, api_key, base_url, model_id) -> str`
- Produces: `resolve_model_dropdown(prompt_source, gemini_catalog, nvidia_catalog, config) -> tuple[list[tuple[str, str]], str, bool]`
- Produces: `persist_selected_model(config, prompt_source, model_id) -> dict`
- Modifies: `run_process(excel_file, custom_output_dir, prompt_source, prompt_model, lovart_mode, lovart_image_model, gemini_key, nvidia_key, lovart_access, lovart_secret)` to persist the selected model before spawning `main.py`

- [ ] **Step 1: Write failing pure callback and persistence tests**

Create `tests/test_webui_model_settings.py` with pure-function tests; do not call `build_ui()` or launch Gradio:

```python
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_provider import DiscoveredModel, ModelProviderError, ModelTestResult
from webui import (
    persist_selected_model,
    refresh_provider_models,
    resolve_model_dropdown,
    save_config,
    test_provider_model,
)


def gemini_model(model_id="gemini-2.5-flash"):
    return DiscoveredModel(
        provider="gemini", model_id=model_id, display_name=model_id,
        supports_generation=True, supports_thinking=True,
        image_input_status="unknown", recommendation="recommended",
    )


class WebUIModelSettingsTests(unittest.TestCase):
    @patch("webui.discover_models")
    def test_refresh_returns_choices_and_preserves_current_model_when_present(self, discover):
        discover.return_value = [gemini_model("gemini-a"), gemini_model("gemini-b")]
        status, choices, selected, catalog = refresh_provider_models(
            "gemini", "key", "https://google.test/v1beta", "gemini-b"
        )
        self.assertIn("成功", status)
        self.assertEqual(selected, "gemini-b")
        self.assertEqual([value for _, value in choices], ["gemini-a", "gemini-b"])
        self.assertEqual(catalog[1]["model_id"], "gemini-b")

    @patch("webui.discover_models")
    def test_refresh_failure_returns_current_model_without_clearing_it(self, discover):
        discover.side_effect = ModelProviderError("network", "网络连接失败")
        status, choices, selected, catalog = refresh_provider_models(
            "gemini", "key", "https://google.test/v1beta", "saved-model"
        )
        self.assertIn("网络连接失败", status)
        self.assertEqual(choices, [("saved-model", "saved-model")])
        self.assertEqual(selected, "saved-model")
        self.assertEqual(catalog, [])

    def test_browser_source_returns_read_only_page_managed_model(self):
        choices, selected, interactive = resolve_model_dropdown(
            "gemini_browser", [], [], {"gemini_api": {}, "nvidia_api": {}}
        )
        self.assertEqual(choices, [("由浏览器页面选择", "由浏览器页面选择")])
        self.assertEqual(selected, "由浏览器页面选择")
        self.assertFalse(interactive)

    def test_source_switch_restores_each_saved_provider_model(self):
        config = {
            "gemini_api": {"model": "gemini-saved"},
            "nvidia_api": {"model": "nvidia-saved"},
        }
        gemini = [gemini_model("gemini-saved").__dict__]
        nvidia = [{**gemini_model("nvidia-saved").__dict__, "provider": "nvidia"}]
        self.assertEqual(resolve_model_dropdown("gemini_api", gemini, nvidia, config)[1], "gemini-saved")
        self.assertEqual(resolve_model_dropdown("nvidia", gemini, nvidia, config)[1], "nvidia-saved")

    def test_persist_selected_model_writes_gemini_direct_model(self):
        updated = persist_selected_model({}, "gemini_api", "gemini-3.5-flash")
        self.assertEqual(updated["gemini_api"]["model"], "gemini-3.5-flash")

    def test_persist_selected_model_writes_nvidia_direct_model(self):
        updated = persist_selected_model({}, "nvidia", "moonshotai/kimi-k2.5")
        self.assertEqual(updated["nvidia_api"]["model"], "moonshotai/kimi-k2.5")

    @patch("webui.os.replace", side_effect=OSError("replace failed"))
    def test_atomic_save_failure_preserves_original_config(self, _replace):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("original: true\n", encoding="utf-8")
            with self.assertRaises(OSError):
                save_config({"changed": True}, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "original: true\n")
            self.assertFalse((Path(tmp) / ".config.yaml.tmp").exists())

    @patch("webui.test_selected_model")
    def test_test_provider_model_returns_usage_notice_and_result(self, test_model):
        test_model.return_value = ModelTestResult(True, "模型可用", 42)
        status = test_provider_model("gemini", "key", "https://google.test/v1beta", "gemini-model")
        self.assertIn("模型可用", status)
        self.assertIn("42", status)
        self.assertIn("API 用量", status)
```

- [ ] **Step 2: Run and confirm callback tests fail**

```powershell
uv run python -m pytest tests/test_webui_model_settings.py -k "refresh or source or persist or atomic or provider" -v
```

Expected: imports fail because the helpers do not exist.

- [ ] **Step 3: Make config writes atomic and add pure model callbacks**

Replace `webui.save_config` with an atomic implementation:

```python
def save_config(config_data: dict, path: str | Path = "config.yaml"):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp")
    try:
        text = yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False)
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()
```

Add the four pure helpers named in the Interfaces block. `refresh_provider_models` catches only `ModelProviderError` and returns a red status plus `[(current_model, current_model)]` when discovery fails. It never returns or logs the key. `persist_selected_model` deep-copies the config and writes either `gemini_api.model` or `nvidia_api.model`; browser source leaves both unchanged.

- [ ] **Step 4: Add Gradio API/model controls and event wiring**

In `build_ui()`:

- Add `gr.State` objects for Gemini and NVIDIA runtime catalogs.
- Add a `prompt_model` dropdown next to `prompt_source`.
- Rename the credentials tab to `API 与模型`.
- Add editable Gemini/NVIDIA base URL fields loaded from config.
- Add one refresh button, one test button, one model dropdown, and one status Markdown per provider.
- Keep Lovart credentials in the same tab.
- Wire refresh buttons to update status, provider dropdown, workspace dropdown, and provider catalog state.
- Wire source changes through `resolve_model_dropdown`; browser mode sets `interactive=False` and value `由浏览器页面选择`.
- Wire provider model selections back to the workspace model when that provider is active.
- Display `测试可能产生极少量 API 用量。` immediately above both test buttons.

Update `run_process` to accept `prompt_model` after `prompt_source`, call `persist_selected_model`, and save the updated config before spawning the CLI. Do not pass a new CLI flag; `main.py` reads the saved direct model field.

- [ ] **Step 5: Run callback and existing WebUI-adjacent tests**

```powershell
uv run python -m pytest tests/test_webui_model_settings.py tests/test_nvidia_api.py tests/test_setup_wizard.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit WebUI model diagnostics**

```powershell
git add webui.py tests/test_webui_model_settings.py
git commit -m "feat: add API diagnostics and model selection UI"
```

---

### Task 6: Persistent prompt-settings UI and read-only effective-rule preview

**Files:**
- Modify: `webui.py`
- Modify: `tests/test_webui_model_settings.py`

**Interfaces:**
- Consumes: `DEFAULT_PROMPT_SETTINGS`, `effective_rules_preview`, `get_prompt_settings`, `locked_rules_text`, `merge_prompt_settings`
- Produces: `prompt_settings_to_form(config) -> tuple`
- Produces: `form_to_prompt_settings(detail_page_count, design_style, required_sections, image_quality, logo_policy, copy_style, copy_detail_level, product_fidelity, white_background_requirements, scene_requirements, allow_questions, default_language, missing_image_size_policy, extra_requirements) -> dict[str, object]`
- Produces: `save_prompt_settings_from_form(*values, config_path="config.yaml") -> tuple[str, str]`
- Produces: `reset_prompt_settings_form() -> tuple`

- [ ] **Step 1: Add failing form round-trip, save, reset, and read-only tests**

Extend imports with `yaml`, `DEFAULT_PROMPT_SETTINGS`, and the four prompt form helpers, then append:

```python
    def _form_values(self, page_count=14):
        return (
            page_count, "自然高级", ["主标题", "规格表"], "2K", "不新增 Logo",
            "具体可信", "详细", "严格还原", "纯白背景精修", "家庭场景",
            False, "英文", "不固定比例", "避免夸张促销词",
        )

    def test_prompt_settings_form_round_trip_preserves_all_fields(self):
        settings = form_to_prompt_settings(*self._form_values())
        config = {"prompt_settings": settings}
        form = prompt_settings_to_form(config)
        self.assertEqual(form[0], 14)
        self.assertEqual(form[2], ["主标题", "规格表"])
        self.assertEqual(form[-1], "避免夸张促销词")

    def test_save_prompt_settings_persists_normalized_values_and_returns_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("excel:\n  path: data/products.xlsx\n", encoding="utf-8")
            status, preview = save_prompt_settings_from_form(*self._form_values(), config_path=path)
            saved = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertIn("已保存", status)
        self.assertEqual(saved["prompt_settings"]["detail_page_count"], 14)
        self.assertEqual(saved["prompt_settings"]["required_sections"], ["主标题", "规格表"])
        self.assertFalse(saved["prompt_settings"]["allow_questions"])
        self.assertEqual(saved["prompt_settings"]["extra_requirements"], "避免夸张促销词")
        self.assertIn("只输出文字", preview)

    def test_invalid_page_count_does_not_modify_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("original: true\n", encoding="utf-8")
            before = path.read_bytes()
            status, _preview = save_prompt_settings_from_form(*self._form_values(99), config_path=path)
            self.assertEqual(path.read_bytes(), before)
        self.assertIn("❌", status)

    def test_reset_returns_defaults_without_writing_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("original: true\n", encoding="utf-8")
            before = path.read_bytes()
            values = reset_prompt_settings_form()
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual(values[0], DEFAULT_PROMPT_SETTINGS["detail_page_count"])
        self.assertIn("锁定规则", values[-1])

    def test_locked_preview_mentions_all_providers_excel_and_lovart(self):
        preview = reset_prompt_settings_form()[-1]
        self.assertIn("所有提示词生成模型", preview)
        self.assertIn("Excel", preview)
        self.assertIn("Lovart", preview)
        self.assertIn("不可编辑", preview)
```

- [ ] **Step 2: Run and confirm prompt form tests fail**

```powershell
uv run python -m pytest tests/test_webui_model_settings.py -k "prompt_settings or reset or locked" -v
```

Expected: failures because prompt form helpers do not exist.

- [ ] **Step 3: Implement pure form conversion and save helpers**

Add `PROMPT_FORM_FIELDS` in the exact order from the design spec and implement:

```python
def form_to_prompt_settings(
    detail_page_count, design_style, required_sections, image_quality,
    logo_policy, copy_style, copy_detail_level, product_fidelity,
    white_background_requirements, scene_requirements, allow_questions,
    default_language, missing_image_size_policy, extra_requirements,
):
    return normalize_prompt_settings({
        "detail_page_count": detail_page_count,
        "design_style": design_style,
        "required_sections": required_sections,
        "image_quality": image_quality,
        "logo_policy": logo_policy,
        "copy_style": copy_style,
        "copy_detail_level": copy_detail_level,
        "product_fidelity": product_fidelity,
        "white_background_requirements": white_background_requirements,
        "scene_requirements": scene_requirements,
        "allow_questions": allow_questions,
        "default_language": default_language,
        "missing_image_size_policy": missing_image_size_policy,
        "extra_requirements": extra_requirements,
    })
```

`save_prompt_settings_from_form` must load the current file, build and validate all form values, call `merge_prompt_settings`, atomically save once, and return `(“✅ 提示词设置已保存”, effective_rules_preview(settings))`. On validation failure it returns `(f“❌ {exc}”, existing_preview)` without writing. `reset_prompt_settings_form` returns default form values plus the default preview and does not touch disk.

- [ ] **Step 4: Build the Prompt Settings tab**

Add a `提示词设置` tab before OTA containing:

- `gr.Number` for page count with precision `0`.
- Textboxes for style, quality, Logo, copy style/detail, fidelity, language fallback, and missing-size policy.
- `gr.CheckboxGroup` for required sections, with the saved list as value.
- Multi-line textboxes for white-background, scene, and extra requirements.
- Checkbox for allowing questions.
- “保存设置” and “恢复默认值” buttons.
- A read-only `gr.Textbox(lines=18, interactive=False)` titled `当前最终生效规则预览`.
- A visible Markdown notice that Excel product fields override these defaults.

Wire save and reset through the pure helpers. Ensure the locked-rule preview is visible on initial load and cannot be edited.

- [ ] **Step 5: Run all WebUI settings tests**

```powershell
uv run python -m pytest tests/test_webui_model_settings.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the prompt settings UI**

```powershell
git add webui.py tests/test_webui_model_settings.py
git commit -m "feat: add persistent prompt settings UI"
```

---

### Task 7: Defaults, documentation, regression, and manual smoke verification

**Files:**
- Modify: `config.example.yaml`
- Modify: `webui.py` embedded default config
- Modify: `README.md`
- Modify: `tests/test_setup_wizard.py` to assert newly created configs contain `prompt_settings`, `gemini_api.model`, and `nvidia_api.model`

**Interfaces:**
- Consumes all prior tasks.
- Produces documented defaults that exactly match `DEFAULT_PROMPT_SETTINGS` and runtime model-field names.

- [ ] **Step 1: Add a failing defaults-consistency test**

Add to `tests/test_webui_model_settings.py`:

```python
def test_example_and_embedded_defaults_expose_prompt_settings_and_direct_models():
    example = Path("config.example.yaml").read_text(encoding="utf-8")
    webui = Path("webui.py").read_text(encoding="utf-8")
    for text in (example, webui):
        self.assertIn("prompt_settings:", text)
        self.assertIn("detail_page_count: 12", text)
        self.assertIn("model: gemini-2.5-flash-lite", text)
        self.assertIn("model: moonshotai/kimi-k2.5", text)
```

- [ ] **Step 2: Run and confirm the consistency test fails**

```powershell
uv run python -m pytest tests/test_webui_model_settings.py::WebUIModelSettingsTests::test_example_and_embedded_defaults_expose_prompt_settings_and_direct_models -v
```

Expected: failure because the defaults have not yet been added to both files.

- [ ] **Step 3: Update defaults and user documentation**

Add the complete `prompt_settings` block from the design spec to `config.example.yaml` and the `DEFAULT_CONFIG` text in `webui.py`. Add `gemini_api.base_url`, retain `gemini_api.model`, and add the direct `nvidia_api.model` while keeping legacy `model_choice/models` documented as compatibility-only.

Update `README.md` with exact user steps:

1. Save Gemini/NVIDIA keys.
2. Click “检测 API 并刷新模型”.
3. Select a model and optionally run the minimal multimodal test.
4. Configure persistent prompt settings.
5. Review the visible locked rules.
6. Start a task; Excel values override software defaults.

State that browser mode needs no API model discovery and that model tests may use a very small amount of API quota.

- [ ] **Step 4: Run focused and full automated verification**

Run:

```powershell
uv run python -m pytest tests/test_prompt_settings.py tests/test_model_provider.py tests/test_webui_model_settings.py -v
uv run python -m pytest -q
```

Expected: all focused tests and the complete existing suite pass.

- [ ] **Step 5: Run static and repository checks**

```powershell
uv run python -m compileall prompt_settings.py model_provider.py utils.py gemini_api.py nvidia_api.py gemini_bot.py main.py webui.py
git diff --check
git status --short
```

Expected: compilation succeeds; `git diff --check` reports no whitespace errors; status shows only intentional files plus the pre-existing user changes that were preserved.

- [ ] **Step 6: Perform a local WebUI smoke test without spending quota**

Run:

```powershell
uv run python webui.py
```

Verify in the browser:

- All four tabs load.
- Browser source shows a disabled “由浏览器页面选择” model.
- Gemini/NVIDIA source changes show the saved model.
- Prompt settings save, reload, and reset-as-form-only behavior work.
- Locked rules are visible and read-only.
- Missing keys produce a clear local error without a network request.
- Do not click a real model test unless the user explicitly authorizes API usage during verification.

- [ ] **Step 7: Commit defaults and documentation**

```powershell
git add config.example.yaml webui.py README.md tests/test_webui_model_settings.py tests/test_setup_wizard.py
git commit -m "docs: document model discovery and prompt settings"
```

---

## Final Verification Gate

Before claiming completion, invoke `superpowers:verification-before-completion` and verify:

```powershell
uv run python -m pytest -q
uv run python -m compileall prompt_settings.py model_provider.py utils.py gemini_api.py nvidia_api.py gemini_bot.py main.py webui.py
git diff --check
git status --short
```

Report exact test counts, any skipped tests, and the preserved pre-existing dirty files. Then ask whether the user wants `version.py` and `version.json` bumped and an OTA package built; do not release automatically.
