import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_provider import DiscoveredModel, ModelProviderError, ModelTestResult
from webui import (
    load_config,
    persist_selected_model,
    refresh_provider_models,
    retain_workspace_model_selection,
    resolve_model_dropdown,
    run_process,
    save_config,
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
