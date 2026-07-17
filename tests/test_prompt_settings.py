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
