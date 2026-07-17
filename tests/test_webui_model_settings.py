import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from model_provider import DiscoveredModel, ModelProviderError, ModelTestResult
from prompt_settings import DEFAULT_PROMPT_SETTINGS
from webui import (
    build_ui,
    form_to_prompt_settings,
    load_config,
    persist_selected_model,
    prompt_settings_to_form,
    refresh_provider_models,
    reset_prompt_settings_form,
    retain_workspace_model_selection,
    resolve_model_dropdown,
    run_process,
    save_config,
    save_prompt_settings_from_form,
    test_provider_model,
)


def gemini_model(model_id="gemini-2.5-flash"):
    return DiscoveredModel(
        provider="gemini",
        model_id=model_id,
        display_name=model_id,
        supports_generation=True,
        supports_thinking=True,
        image_input_status="unknown",
        recommendation="recommended",
    )


class WebUIModelSettingsTests(unittest.TestCase):
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
        self.assertEqual(form, self._form_values())

    def test_save_prompt_settings_persists_normalized_values_and_returns_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("excel:\n  path: data/products.xlsx\n", encoding="utf-8")
            status, preview = save_prompt_settings_from_form(*self._form_values(), config_path=path)
            saved = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertIn("已保存", status)
        self.assertEqual(saved["excel"]["path"], "data/products.xlsx")
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

    @patch("webui.load_config")
    def test_prompt_settings_tab_is_complete_read_only_and_preserves_custom_sections(self, load_config_mock):
        load_config_mock.return_value = {
            "prompt_settings": {
                "detail_page_count": 14,
                "required_sections": ["主标题", "自定义规格模块"],
                "extra_requirements": "避免夸张促销词",
            }
        }

        demo = build_ui()
        components = demo.config["components"]
        tabs = [item["props"]["label"] for item in components if item["type"] == "tabitem"]
        self.assertLess(tabs.index("📝 提示词设置"), tabs.index("⚙️ 系统更新 (OTA)"))

        by_label = {
            item.get("props", {}).get("label"): item
            for item in components
            if item.get("props", {}).get("label")
        }
        preview = by_label["当前最终生效规则预览"]
        self.assertEqual(preview["type"], "textbox")
        self.assertFalse(preview["props"]["interactive"])
        self.assertEqual(preview["props"]["lines"], 18)
        self.assertIn("锁定规则（不可编辑）", preview["props"]["value"])

        sections = by_label["每屏必须包含的内容"]
        self.assertEqual(sections["props"]["value"], ["主标题", "自定义规格模块"])
        section_values = [choice[1] for choice in sections["props"]["choices"]]
        self.assertIn("自定义规格模块", section_values)

        markdown_values = [
            str(item.get("props", {}).get("value", ""))
            for item in components
            if item["type"] == "markdown"
        ]
        self.assertTrue(any("Excel" in value and "优先" in value for value in markdown_values))
        api_names = {item["api_name"] for item in demo.config["dependencies"]}
        self.assertIn("save_prompt_settings_from_form", api_names)
        self.assertIn("reset_prompt_settings_form", api_names)

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

    def test_legacy_nvidia_selection_restores_model_id(self):
        config = {
            "nvidia_api": {
                "model_choice": "kimi",
                "models": {"kimi": "moonshotai/kimi-k2.5"},
            }
        }
        nvidia = [
            {**gemini_model("other-model").__dict__, "provider": "nvidia"},
            {**gemini_model("moonshotai/kimi-k2.5").__dict__, "provider": "nvidia"},
        ]

        self.assertEqual(resolve_model_dropdown("nvidia", [], nvidia, config)[1], "moonshotai/kimi-k2.5")

    def test_direct_nvidia_model_takes_precedence_over_legacy_selection(self):
        config = {
            "nvidia_api": {
                "model": "direct-model",
                "model_choice": "kimi",
                "models": {"kimi": "legacy-model"},
            }
        }
        nvidia = [
            {**gemini_model("direct-model").__dict__, "provider": "nvidia"},
            {**gemini_model("legacy-model").__dict__, "provider": "nvidia"},
        ]

        self.assertEqual(resolve_model_dropdown("nvidia", [], nvidia, config)[1], "direct-model")

    def test_workspace_selection_survives_switching_away_and_back(self):
        gemini = [gemini_model("gemini-a").__dict__, gemini_model("gemini-b").__dict__]
        nvidia = [{**gemini_model("nvidia-a").__dict__, "provider": "nvidia"}]

        gemini_selected, nvidia_selected = retain_workspace_model_selection(
            "gemini_api", "gemini-b", "gemini-a", "nvidia-a"
        )
        live_config = {
            "gemini_api": {"model": gemini_selected},
            "nvidia_api": {"model": nvidia_selected},
        }

        self.assertEqual(resolve_model_dropdown("nvidia", gemini, nvidia, live_config)[1], "nvidia-a")
        self.assertEqual(resolve_model_dropdown("gemini_api", gemini, nvidia, live_config)[1], "gemini-b")

    def test_persist_selected_model_writes_gemini_direct_model(self):
        updated = persist_selected_model({}, "gemini_api", "gemini-3.5-flash")
        self.assertEqual(updated["gemini_api"]["model"], "gemini-3.5-flash")

    def test_persist_selected_model_writes_nvidia_direct_model(self):
        updated = persist_selected_model({}, "nvidia", "moonshotai/kimi-k2.5")
        self.assertEqual(updated["nvidia_api"]["model"], "moonshotai/kimi-k2.5")

    @patch("webui.save_config")
    @patch("webui.load_config")
    @patch("webui.save_env")
    def test_run_process_persists_selected_model_before_starting(self, _save_env, load_config, save_config_mock):
        load_config.return_value = {
            "gemini_api": {"model": "gemini-old"},
            "nvidia_api": {"model": "nvidia-old"},
        }
        process = run_process(
            None, "output", "gemini_api", "gemini-new", "unlimited", "auto",
            "gemini-key", "nvidia-key", "lovart-access", "lovart-secret",
        )

        self.assertIn("Starting", next(process))
        saved = save_config_mock.call_args.args[0]
        self.assertEqual(saved["gemini_api"]["model"], "gemini-new")
        self.assertEqual(saved["nvidia_api"]["model"], "nvidia-old")

    @patch("webui.os.replace", side_effect=OSError("replace failed"))
    def test_atomic_save_failure_preserves_original_config(self, _replace):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("original: true\n", encoding="utf-8")
            with self.assertRaises(OSError):
                save_config({"changed": True}, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "original: true\n")
            self.assertFalse((Path(tmp) / ".config.yaml.tmp").exists())

    @patch("webui.os.replace", side_effect=OSError("replace failed"))
    def test_fresh_config_creation_failure_leaves_no_partial_target(self, _replace):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"

            with self.assertRaises(OSError):
                load_config(path)

            self.assertFalse(path.exists())
            self.assertFalse((Path(tmp) / ".config.yaml.tmp").exists())

    @patch("webui.test_selected_model")
    def test_test_provider_model_returns_usage_notice_and_result(self, test_model):
        test_model.return_value = ModelTestResult(True, "模型可用", 42)
        status = test_provider_model("gemini", "key", "https://google.test/v1beta", "gemini-model")
        self.assertIn("模型可用", status)
        self.assertIn("42", status)
        self.assertIn("API 用量", status)

    @patch("webui.test_selected_model")
    def test_test_provider_model_renders_non_success_when_result_is_not_ok(self, test_model):
        test_model.return_value = ModelTestResult(False, "模型不可用", 17)

        status = test_provider_model("gemini", "key", "https://google.test/v1beta", "gemini-model")

        self.assertIn("❌", status)
        self.assertNotIn("✅", status)
        self.assertIn("模型不可用", status)
