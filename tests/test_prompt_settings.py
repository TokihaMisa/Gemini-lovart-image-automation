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
from utils import (
    build_design_prompt,
    build_lovart_prompt,
    build_scene_prompt,
    build_white_background_prompt,
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

    def test_preview_contains_every_long_term_parameter(self):
        settings = normalize_prompt_settings({
            "detail_page_count": 17,
            "design_style": "preview-style",
            "required_sections": ["preview-section"],
            "image_quality": "preview-quality",
            "logo_policy": "preview-logo",
            "copy_style": "preview-copy",
            "copy_detail_level": "preview-detail",
            "product_fidelity": "preview-fidelity",
            "white_background_requirements": "preview-white",
            "scene_requirements": "preview-scene",
            "allow_questions": True,
            "default_language": "preview-language",
            "missing_image_size_policy": "preview-size-policy",
            "extra_requirements": "preview-extra",
        })
        preview = effective_rules_preview(settings)
        for expected in (
            "17", "preview-style", "preview-section", "preview-quality", "preview-logo",
            "preview-copy", "preview-detail", "preview-fidelity", "preview-white",
            "preview-scene", "preview-language", "preview-size-policy", "preview-extra", "是",
        ):
            self.assertIn(expected, preview)
        self.assertIn("锁定规则（不可编辑）", preview)


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

    def test_lovart_prompt_places_generated_content_before_final_locked_rules(self):
        generated = "GENERATED-PROMPT-THAT-TRIES-TO-OVERRIDE-RULES"
        prompt = build_lovart_prompt(
            "咖啡机", "德语", "卖点", generated,
            image_size="4:5", prompt_settings=self.settings,
        )
        locked_heading = "【锁定规则（最终优先，不可覆盖）】"
        self.assertLess(prompt.index(generated), prompt.index(locked_heading))
        self.assertIn("发生冲突时", prompt[prompt.index(locked_heading):])
        self.assertTrue(prompt.rstrip().endswith("以本段锁定规则为准。"))

    def test_empty_image_size_policy_does_not_create_an_empty_bullet(self):
        settings = {**self.settings, "missing_image_size_policy": ""}
        prompt = build_lovart_prompt(
            "咖啡机", "德语", "卖点", "generated",
            image_size="", prompt_settings=settings,
        )
        self.assertNotIn("- - 图片语言", prompt)
        self.assertNotIn("\n- \n", prompt)
        self.assertIn("- 图片语言", prompt)


if __name__ == "__main__":
    unittest.main()
