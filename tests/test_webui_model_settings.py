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
    persist_provider_settings,
    probe_provider_model,
    prompt_settings_to_form,
    refresh_provider_models,
    reset_prompt_settings_form,
    retain_workspace_model_selection,
    resolve_model_dropdown,
    run_process,
    save_api_settings,
    save_config,
    save_prompt_settings_from_form,
    test_provider_model,
    update_catalog_image_status,
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
    def test_example_and_embedded_defaults_expose_prompt_settings_and_direct_models(self):
        example = Path("config.example.yaml").read_text(encoding="utf-8")
        webui = Path("webui.py").read_text(encoding="utf-8")
        for text in (example, webui):
            self.assertIn("prompt_settings:", text)
            self.assertIn("detail_page_count: 12", text)
            self.assertIn("model: gemini-2.5-flash-lite", text)
            self.assertIn("model: moonshotai/kimi-k2.5", text)

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
        expected = form_to_prompt_settings(*self._form_values())
        self.assertEqual(saved["prompt_settings"], expected)
        self.assertEqual(len(saved["prompt_settings"]), 14)
        self.assertIn("只输出文字", preview)

    def test_custom_required_sections_save_and_reload_from_multiline_text(self):
        values = list(self._form_values())
        values[2] = "主标题\n自定义规格模块, 售后说明"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            status, _preview = save_prompt_settings_from_form(*values, config_path=path)
            reloaded = prompt_settings_to_form(load_config(path))
        self.assertIn("已保存", status)
        self.assertEqual(reloaded[2], ["主标题", "自定义规格模块", "售后说明"])

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
        self.assertEqual(sections["type"], "textbox")
        self.assertTrue(sections["props"]["interactive"])
        self.assertIn("主标题", sections["props"]["value"])
        self.assertIn("自定义规格模块", sections["props"]["value"])

        markdown_values = [
            str(item.get("props", {}).get("value", ""))
            for item in components
            if item["type"] == "markdown"
        ]
        self.assertTrue(any("Excel" in value and "优先" in value for value in markdown_values))
        api_names = {item["api_name"] for item in demo.config["dependencies"]}
        self.assertIn("save_prompt_settings_from_form", api_names)
        self.assertIn("reset_prompt_settings_form", api_names)

    @patch("webui.load_config")
    def test_api_save_probe_and_run_events_include_endpoint_model_and_catalog_controls(self, load_config_mock):
        load_config_mock.return_value = {
            "gemini_api": {
                "base_url": "https://gemini.test/v1beta", "model": "gemini-model"
            },
            "nvidia_api": {
                "base_url": "https://nvidia.test/v1", "model": "nvidia-model"
            },
        }
        demo = build_ui()
        component_labels = {
            item["id"]: item.get("props", {}).get("label")
            for item in demo.config["components"]
        }
        dependencies = {item["api_name"]: item for item in demo.config["dependencies"]}

        save_event = dependencies["save_api_settings"]
        save_labels = {component_labels[item] for item in save_event["inputs"]}
        self.assertIn("Gemini API 地址", save_labels)
        self.assertIn("Gemini 模型", save_labels)
        self.assertIn("NVIDIA API 地址", save_labels)
        self.assertIn("NVIDIA 模型", save_labels)

        for api_name, endpoint_label in (
            ("probe_gemini_model", "Gemini API 地址"),
            ("probe_nvidia_model", "NVIDIA API 地址"),
        ):
            event = dependencies[api_name]
            input_labels = {component_labels[item] for item in event["inputs"]}
            self.assertIn(endpoint_label, input_labels)
            self.assertIn("提示词引擎", input_labels)
            self.assertEqual(len(event["outputs"]), 4)

        run_event = dependencies["run_process"]
        run_labels = {component_labels[item] for item in run_event["inputs"]}
        self.assertIn("Gemini API 地址", run_labels)
        self.assertIn("NVIDIA API 地址", run_labels)

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

    @patch("webui.discover_models")
    def test_refresh_prefers_recommended_model_when_current_model_disappeared(self, discover):
        discover.return_value = [
            DiscoveredModel("gemini", "available-first", "Available", True, None, "unknown", "available"),
            DiscoveredModel("gemini", "recommended-second", "Recommended", True, None, "unknown", "recommended"),
        ]
        _status, _choices, selected, _catalog = refresh_provider_models(
            "gemini", "key", "https://google.test/v1beta", "removed-model"
        )
        self.assertEqual(selected, "recommended-second")

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

    def test_initial_resolution_prefers_recommended_model_when_saved_model_is_absent(self):
        catalog = [
            DiscoveredModel("gemini", "available-first", "Available", True, None, "unknown", "available").__dict__,
            DiscoveredModel("gemini", "recommended-second", "Recommended", True, None, "unknown", "recommended").__dict__,
        ]
        config = {"gemini_api": {"model": "removed-model"}, "nvidia_api": {}}
        choices, selected, interactive = resolve_model_dropdown("gemini_api", catalog, [], config)
        self.assertEqual(selected, "recommended-second")
        self.assertNotIn("removed-model", [value for _, value in choices])
        self.assertTrue(interactive)

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

    def test_provider_settings_validate_and_persist_base_urls_with_models(self):
        updated = persist_provider_settings(
            {"other": {"keep": True}},
            "https://gemini.proxy.test/v1beta/", "gemini-custom",
            "https://nvidia.proxy.test/v1/", "nvidia/custom",
        )
        self.assertEqual(updated["gemini_api"], {
            "base_url": "https://gemini.proxy.test/v1beta", "model": "gemini-custom"
        })
        self.assertEqual(updated["nvidia_api"], {
            "base_url": "https://nvidia.proxy.test/v1", "model": "nvidia/custom"
        })
        self.assertTrue(updated["other"]["keep"])

    @patch("webui.save_env")
    def test_save_api_settings_atomically_persists_both_provider_addresses_and_models(self, save_env_mock):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("other:\n  keep: true\n", encoding="utf-8")
            status = save_api_settings(
                "gemini-key", "nvidia-key", "lovart-access", "lovart-secret",
                "https://gemini.proxy.test/v1beta", "gemini-custom",
                "https://nvidia.proxy.test/v1", "nvidia/custom",
                config_path=path,
            )
            saved = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertIn("已保存", status)
        self.assertEqual(saved["gemini_api"]["base_url"], "https://gemini.proxy.test/v1beta")
        self.assertEqual(saved["gemini_api"]["model"], "gemini-custom")
        self.assertEqual(saved["nvidia_api"]["base_url"], "https://nvidia.proxy.test/v1")
        self.assertEqual(saved["nvidia_api"]["model"], "nvidia/custom")
        save_env_mock.assert_called_once()

    @patch("webui.save_env")
    def test_invalid_api_settings_do_not_modify_config_or_env(self, save_env_mock):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("original: true\n", encoding="utf-8")
            before = path.read_bytes()
            status = save_api_settings(
                "gemini-key", "nvidia-key", "lovart-access", "lovart-secret",
                "not-a-url", "gemini-custom",
                "https://nvidia.proxy.test/v1", "nvidia/custom",
                config_path=path,
            )
            self.assertEqual(path.read_bytes(), before)
        self.assertIn("地址", status)
        save_env_mock.assert_not_called()

    @patch("webui.save_config")
    @patch("webui.load_config")
    @patch("webui.save_env")
    def test_run_process_persists_selected_model_and_current_base_urls_before_starting(self, _save_env, load_config, save_config_mock):
        load_config.return_value = {
            "gemini_api": {"model": "gemini-old"},
            "nvidia_api": {"model": "nvidia-old"},
        }
        process = run_process(
            None, "output", "gemini_api", "gemini-new", "unlimited", "auto",
            "https://gemini.current.test/v1beta", "https://nvidia.current.test/v1",
            "gemini-key", "nvidia-key", "lovart-access", "lovart-secret",
        )

        self.assertIn("Starting", next(process))
        saved = save_config_mock.call_args.args[0]
        self.assertEqual(saved["gemini_api"]["model"], "gemini-new")
        self.assertEqual(saved["gemini_api"]["base_url"], "https://gemini.current.test/v1beta")
        self.assertEqual(saved["nvidia_api"]["model"], "nvidia-old")
        self.assertEqual(saved["nvidia_api"]["base_url"], "https://nvidia.current.test/v1")

    @patch("webui.save_config")
    @patch("webui.load_config")
    @patch("webui.save_env")
    def test_run_process_rejects_invalid_endpoint_without_writing_or_starting(
        self, save_env_mock, load_config, save_config_mock
    ):
        load_config.return_value = {
            "gemini_api": {"model": "gemini-model"},
            "nvidia_api": {"model": "nvidia-model"},
        }
        process = run_process(
            None, "output", "gemini_api", "gemini-model", "unlimited", "auto",
            "not-a-url", "https://nvidia.test/v1",
            "gemini-key", "nvidia-key", "lovart-access", "lovart-secret",
        )
        status = next(process)
        self.assertIn("API 地址", status)
        save_env_mock.assert_not_called()
        save_config_mock.assert_not_called()

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

    def test_malformed_yaml_prompt_save_returns_actionable_error_and_preserves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("prompt_settings: [unterminated", encoding="utf-8")
            before = path.read_bytes()
            status, preview = save_prompt_settings_from_form(*self._form_values(), config_path=path)
            self.assertEqual(path.read_bytes(), before)
        self.assertIn("读取", status)
        self.assertIn("config", status)
        self.assertIn("锁定规则", preview)

    @patch("webui.os.replace", side_effect=OSError("replace failed"))
    def test_prompt_save_write_failure_returns_error_and_preserves_original(self, _replace):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("original: true\n", encoding="utf-8")
            before = path.read_bytes()
            status, _preview = save_prompt_settings_from_form(*self._form_values(), config_path=path)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((Path(tmp) / ".config.yaml.tmp").exists())
        self.assertIn("保存失败", status)

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

    def test_catalog_status_update_is_pure_and_preserves_selection_data(self):
        original = [gemini_model("gemini-a").__dict__, gemini_model("gemini-b").__dict__]
        updated = update_catalog_image_status(original, "gemini-b", "verified")
        self.assertEqual(original[1]["image_input_status"], "unknown")
        self.assertEqual(updated[0]["image_input_status"], "unknown")
        self.assertEqual(updated[1]["image_input_status"], "verified")
        self.assertEqual(updated[1]["model_id"], "gemini-b")

    @patch("webui.test_selected_model")
    def test_probe_success_marks_catalog_verified_and_updates_label(self, test_model):
        test_model.return_value = ModelTestResult(True, "模型可用", 21)
        status, choices, selected, catalog = probe_provider_model(
            "gemini", "key", "https://google.test/v1beta", "gemini-a",
            [gemini_model("gemini-a").__dict__],
        )
        self.assertIn("模型可用", status)
        self.assertEqual(selected, "gemini-a")
        self.assertEqual(catalog[0]["image_input_status"], "verified")
        self.assertIn("图片已验证支持", choices[0][0])

    @patch("webui.test_selected_model")
    def test_probe_non_ok_result_marks_catalog_failed_and_preserves_selection(self, test_model):
        test_model.return_value = ModelTestResult(False, "模型不可用", 17)
        status, choices, selected, catalog = probe_provider_model(
            "gemini", "key", "https://google.test/v1beta", "gemini-a",
            [gemini_model("gemini-a").__dict__],
        )
        self.assertIn("模型不可用", status)
        self.assertEqual(selected, "gemini-a")
        self.assertEqual(catalog[0]["image_input_status"], "failed")
        self.assertIn("图片不支持", choices[0][0])

    @patch("webui.test_selected_model")
    def test_probe_provider_error_marks_catalog_failed_and_preserves_selection(self, test_model):
        test_model.side_effect = ModelProviderError("model_unavailable", "模型不存在或不可用")
        status, choices, selected, catalog = probe_provider_model(
            "nvidia", "key", "https://nvidia.test/v1", "nvidia-a",
            [{**gemini_model("nvidia-a").__dict__, "provider": "nvidia"}],
        )
        self.assertIn("模型不存在或不可用", status)
        self.assertEqual(selected, "nvidia-a")
        self.assertEqual(catalog[0]["image_input_status"], "failed")
        self.assertIn("图片不支持", choices[0][0])
