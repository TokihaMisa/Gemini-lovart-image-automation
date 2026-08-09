import inspect
import io
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
import webui

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
    save_env,
    save_failed_retry_settings,
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
    @patch("webui.OpenAIImageAPI")
    def test_paid_image_test_runs_only_when_handler_is_explicitly_called(self, api_cls):
        api_cls.return_value.test_edit.return_value.local_path = "test-output.png"

        message = webui.test_openai_image_edit(
            "test-key", "https://hapiopen.cc", "gpt-image-2", "1K"
        )

        self.assertIn("测试成功", message)
        api_cls.return_value.test_edit.assert_called_once()

    @patch("webui.OpenAIImageAPI")
    @patch("webui.load_config", return_value={})
    def test_ui_build_and_save_never_run_paid_image_test(self, _load_config, api_cls):
        demo = build_ui()
        self.assertIsNotNone(demo)

        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            env_path = Path(tmp) / ".env"
            save_api_settings(
                "", "", "", "", "",
                "https://gemini.test/v1beta", "gemini-model",
                "https://nvidia.test/v1", "nvidia-model",
                "https://hapiopen.cc/v1", "gpt-image-2", "1K",
                "openai_image", "openai_image",
                config_path=config_path,
                env_path=env_path,
            )

        api_cls.assert_not_called()

    @patch("webui.load_config", return_value={})
    def test_ui_exposes_gpt_image_settings_routes_and_explicit_charge_button(self, _load_config):
        demo = build_ui()
        components = demo.config["components"]
        labels = {
            item["id"]: item.get("props", {}).get("label") for item in components
        }
        by_label = {
            item.get("props", {}).get("label"): item
            for item in components
            if item.get("props", {}).get("label")
        }
        dependencies = {
            item["api_name"]: item for item in demo.config["dependencies"]
        }

        self.assertEqual(by_label["GPT Image API 密钥"]["props"]["type"], "password")
        self.assertEqual(by_label["GPT Image API 密钥"]["props"]["value"], "")
        self.assertTrue(by_label["GPT Image API 地址"]["props"]["value"].endswith("/v1"))
        self.assertEqual(by_label["GPT Image 模型"]["props"]["value"], "gpt-image-2")
        self.assertEqual(
            {
                choice[1] if isinstance(choice, (list, tuple)) else choice
                for choice in by_label["GPT Image 分辨率"]["props"]["choices"]
            },
            {"1K", "2K", "4K"},
        )
        self.assertFalse(by_label["清除已保存 GPT Image 密钥"]["props"]["value"])

        run_labels = {labels[item] for item in dependencies["run_process"]["inputs"]}
        self.assertIn("白底图和场景图来源", run_labels)
        self.assertIn("最终套图来源", run_labels)
        self.assertIn("清除已保存 GPT Image 密钥", run_labels)
        save_labels = {labels[item] for item in dependencies["save_api_settings"]["inputs"]}
        self.assertIn("清除已保存 GPT Image 密钥", save_labels)
        markdown_values = [
            str(item.get("props", {}).get("value", ""))
            for item in components
            if item["type"] == "markdown"
        ]
        self.assertTrue(any("GPT Image 密钥状态：" in value for value in markdown_values))

        paid_test = dependencies["test_openai_image_edit"]
        self.assertGreaterEqual(
            {labels[item] for item in paid_test["inputs"]},
            {"GPT Image API 密钥", "GPT Image API 地址", "GPT Image 模型", "GPT Image 分辨率"},
        )
        button_values = [
            str(item.get("props", {}).get("value", ""))
            for item in components
            if item["type"] == "button"
        ]
        self.assertTrue(any("可能产生一次图片费用" in value for value in button_values))
        self.assertIn(
            "清除已保存 GPT Image 密钥",
            {labels[item] for item in dependencies["save_api_settings"]["outputs"]},
        )
        self.assertIn(
            "清除已保存 GPT Image 密钥",
            {labels[item] for item in dependencies["run_process"]["outputs"]},
        )

    def test_save_ui_clear_flow_updates_indicator_resets_checkbox_and_allows_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            env_path = Path(tmp) / ".env"
            config_path.write_text("{}\n", encoding="utf-8")
            env_path.write_text("OPENAI_IMAGE_API_KEY=old-secret\n", encoding="utf-8")
            cleared = webui.save_api_settings_from_ui(
                "", "", "", "", "",
                "https://gemini.test/v1beta", "gemini-model",
                "https://nvidia.test/v1", "nvidia-model",
                "https://hapiopen.cc/v1", "gpt-image-2", "1K",
                "lovart", "lovart", True,
                config_path=config_path,
                env_path=env_path,
            )
            replaced = webui.save_api_settings_from_ui(
                "", "", "", "", "replacement-secret",
                "https://gemini.test/v1beta", "gemini-model",
                "https://nvidia.test/v1", "nvidia-model",
                "https://hapiopen.cc/v1", "gpt-image-2", "1K",
                "openai_image", "openai_image", cleared[2],
                config_path=config_path,
                env_path=env_path,
            )
            saved_env = env_path.read_text(encoding="utf-8")

        self.assertEqual(cleared, (
            webui.API_SETTINGS_SAVE_SUCCESS,
            "GPT Image 密钥状态：未保存",
            False,
        ))
        self.assertEqual(replaced[1:], ("GPT Image 密钥状态：已保存", False))
        self.assertNotIn("old-secret", saved_env)
        self.assertIn("OPENAI_IMAGE_API_KEY=replacement-secret", saved_env)
        self.assertNotIn("replacement-secret", str((cleared, replaced)))

    @patch("webui.load_config", return_value={})
    def test_build_ui_uses_gradio6_launch_options_without_constructor_warning(self, _load_config):
        launch_options_factory = getattr(webui, "gradio_launch_kwargs", None)
        self.assertIsNotNone(launch_options_factory)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            demo = build_ui()

        self.assertIsNotNone(demo)
        migration_warnings = [
            str(item.message)
            for item in caught
            if "moved from the Blocks constructor to the launch() method" in str(item.message)
        ]
        self.assertEqual(migration_warnings, [])

        launch_options = launch_options_factory()
        self.assertIn("css", launch_options)
        self.assertIn("gradient-text", launch_options["css"])
        self.assertIn("js", launch_options)
        self.assertIn("classList.add('dark')", launch_options["js"])

        app_source = Path("app.py").read_text(encoding="utf-8")
        webui_source = Path("webui.py").read_text(encoding="utf-8")
        self.assertIn("**gradio_launch_kwargs()", app_source)
        self.assertIn("**gradio_launch_kwargs()", webui_source)

    @patch("webui.load_config", return_value={})
    def test_build_ui_keeps_api_save_callback_arity_valid_before_gpt_image_controls_exist(self, _load_config):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_ui()

        arity_warnings = [
            str(item.message)
            for item in caught
            if "Expected at least" in str(item.message)
        ]
        self.assertEqual(arity_warnings, [])

    def test_example_and_embedded_defaults_expose_prompt_settings_and_direct_models(self):
        example = Path("config.example.yaml").read_text(encoding="utf-8")
        webui = Path("webui.py").read_text(encoding="utf-8")
        for text in (example, webui):
            self.assertIn("prompt_settings:", text)
            self.assertIn("detail_page_count: 12", text)
            self.assertIn("model: gemini-2.5-flash-lite", text)
            self.assertIn("model: moonshotai/kimi-k2.5", text)
            self.assertIn("allow_regular_chat_fallback: false", text)
            self.assertIn("failed_retry_mode: finite", text)
            self.assertIn("failed_retry_error_types:", text)
            self.assertIn("unlimited_models: []", text)

    def test_example_defaults_include_image_routing_and_openai_image(self):
        config = yaml.safe_load(Path("config.example.yaml").read_text(encoding="utf-8"))
        self.assertEqual(
            config["image_generation"],
            {"support_provider": "lovart", "detail_provider": "lovart"},
        )
        self.assertEqual(config["openai_image"]["base_url"], "https://hapiopen.cc/v1")
        self.assertEqual(config["openai_image"]["model"], "gpt-image-2")

    def test_blank_openai_image_key_preserves_existing_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OPENAI_IMAGE_API_KEY=existing\n", encoding="utf-8")

            save_env("", "", "", "", openai_image_key="", env_path=env_path)

            self.assertIn("OPENAI_IMAGE_API_KEY=existing", env_path.read_text(encoding="utf-8"))

    def test_clear_openai_image_key_removes_existing_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("# local credential\r\nOPENAI_IMAGE_API_KEY=existing\r\nOTHER=value\r\n", encoding="utf-8", newline="")

            save_env("", "", "", "", clear_openai_image_key=True, env_path=env_path)

            self.assertEqual(env_path.read_bytes(), b"# local credential\r\nOTHER=value\r\nGEMINI_API_KEY=\r\nNVIDIA_API_KEY=\r\nLOVART_ACCESS_KEY=\r\nLOVART_SECRET_KEY=\r\n")

    def test_explicit_openai_image_key_clear_wins_over_a_submitted_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OPENAI_IMAGE_API_KEY=existing\n", encoding="utf-8")

            save_env(
                "", "", "", "",
                openai_image_key="stale-value",
                clear_openai_image_key=True,
                env_path=env_path,
            )

            self.assertNotIn("OPENAI_IMAGE_API_KEY=", env_path.read_text(encoding="utf-8"))

    def test_openai_image_key_indicator_reports_saved_without_echoing_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OPENAI_IMAGE_API_KEY=super-secret\n", encoding="utf-8")

            indicator = webui.openai_image_key_status(env_path)

        self.assertIn("已保存", indicator)
        self.assertNotIn("super-secret", indicator)

    def test_persist_openai_image_settings_normalizes_config_without_mutating_input(self):
        original = {"other": {"keep": True}}

        updated = webui.persist_openai_image_settings(
            original,
            "https://hapiopen.cc/", " custom-image ", "2k",
            "openai_image", "lovart",
        )

        self.assertEqual(original, {"other": {"keep": True}})
        self.assertEqual(updated["openai_image"], {
            "base_url": "https://hapiopen.cc/v1",
            "model": "custom-image",
            "resolution": "2K",
        })
        self.assertEqual(updated["image_generation"], {
            "support_provider": "openai_image",
            "detail_provider": "lovart",
        })

    def test_persist_openai_image_settings_preserves_unrelated_image_routing_fields(self):
        original = {
            "image_generation": {
                "future_option": "keep-me",
                "support_provider": "lovart",
                "detail_provider": "lovart",
            },
        }

        updated = webui.persist_openai_image_settings(
            original,
            "https://hapiopen.cc/v1", "gpt-image-2", "1K",
            "openai_image", "openai_image",
        )

        self.assertEqual(original["image_generation"]["future_option"], "keep-me")
        self.assertEqual(updated["image_generation"], {
            "future_option": "keep-me",
            "support_provider": "openai_image",
            "detail_provider": "openai_image",
        })

    def test_save_api_settings_replaces_nonblank_openai_key_without_writing_it_to_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path, env_path = Path(tmp) / "config.yaml", Path(tmp) / ".env"
            config_path.write_text("other: keep\n", encoding="utf-8")
            env_path.write_text(
                "OPENAI_IMAGE_API_KEY=old-key\nUNRELATED_ENV=preserve\n",
                encoding="utf-8",
            )

            status = save_api_settings(
                "", "", "", "", "new-key",
                "https://gemini.test/v1beta", "gemini-test",
                "https://nvidia.test/v1", "nvidia-test",
                "https://hapiopen.cc/v1", "gpt-image-2", "1K",
                "openai_image", "openai_image",
                config_path=config_path, env_path=env_path,
            )

            saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            saved_env = env_path.read_text(encoding="utf-8")
            self.assertNotIn("OPENAI_IMAGE_API_KEY", yaml.safe_dump(saved_config))
            self.assertNotIn("new-key", yaml.safe_dump(saved_config))
            self.assertNotIn("OPENAI_IMAGE_API_KEY=old-key", saved_env)
            self.assertIn("OPENAI_IMAGE_API_KEY=new-key", saved_env)
            self.assertIn("UNRELATED_ENV=preserve", saved_env)
        self.assertNotIn("new-key", status)

    def test_save_api_settings_none_openai_key_preserves_existing_key_and_unrelated_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path, env_path = Path(tmp) / "config.yaml", Path(tmp) / ".env"
            config_path.write_text("other: keep\n", encoding="utf-8")
            env_path.write_text(
                "OPENAI_IMAGE_API_KEY=existing-key\nUNRELATED_ENV=preserve\n",
                encoding="utf-8",
            )

            status = save_api_settings(
                "", "", "", "", None,
                "https://gemini.test/v1beta", "gemini-test",
                "https://nvidia.test/v1", "nvidia-test",
                "https://hapiopen.cc/v1", "gpt-image-2", "1K",
                "openai_image", "openai_image",
                config_path=config_path, env_path=env_path,
            )

            saved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            saved_env = env_path.read_text(encoding="utf-8")
            self.assertNotIn("OPENAI_IMAGE_API_KEY", yaml.safe_dump(saved_config))
            self.assertNotIn("existing-key", yaml.safe_dump(saved_config))
            self.assertIn("OPENAI_IMAGE_API_KEY=existing-key", saved_env)
            self.assertIn("UNRELATED_ENV=preserve", saved_env)
        self.assertNotIn("existing-key", status)

    def test_transaction_rolls_back_both_files_when_config_temp_write_fails_without_leaking_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path, env_path = Path(tmp) / "config.yaml", Path(tmp) / ".env"
            config_path.write_bytes(b"original: config\r\n")
            env_path.write_bytes(b"UNRELATED_ENV=preserve\r\nOPENAI_IMAGE_API_KEY=old-key\r\n")
            original_config, original_env = config_path.read_bytes(), env_path.read_bytes()
            real_write_text = Path.write_text

            def fail_config_temp_write(path, *args, **kwargs):
                if path.name == ".config.yaml.tmp":
                    raise OSError("config temp write failed")
                return real_write_text(path, *args, **kwargs)

            with patch.object(webui.Path, "write_text", autospec=True, side_effect=fail_config_temp_write):
                status = save_api_settings(
                    "", "", "", "", "new-key",
                    "https://gemini.test/v1beta", "gemini-test",
                    "https://nvidia.test/v1", "nvidia-test",
                    "https://hapiopen.cc/v1", "gpt-image-2", "1K",
                    "openai_image", "openai_image",
                    config_path=config_path, env_path=env_path,
                )

            self.assertEqual(config_path.read_bytes(), original_config)
            self.assertEqual(env_path.read_bytes(), original_env)
        self.assertIn("config temp write failed", status)
        self.assertNotIn("new-key", status)
        self.assertNotIn("old-key", status)

    def test_transaction_rolls_back_both_files_when_config_replace_fails_without_leaking_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path, env_path = Path(tmp) / "config.yaml", Path(tmp) / ".env"
            config_path.write_bytes(b"original: config\r\n")
            env_path.write_bytes(b"UNRELATED_ENV=preserve\r\nOPENAI_IMAGE_API_KEY=old-key\r\n")
            original_config, original_env = config_path.read_bytes(), env_path.read_bytes()
            real_replace = os.replace

            def fail_config_replace(source, destination):
                if Path(source).name == ".config.yaml.tmp":
                    raise OSError("config replace failed")
                return real_replace(source, destination)

            with patch("webui.os.replace", side_effect=fail_config_replace):
                status = save_api_settings(
                    "", "", "", "", "new-key",
                    "https://gemini.test/v1beta", "gemini-test",
                    "https://nvidia.test/v1", "nvidia-test",
                    "https://hapiopen.cc/v1", "gpt-image-2", "1K",
                    "openai_image", "openai_image",
                    config_path=config_path, env_path=env_path,
                )

            self.assertEqual(config_path.read_bytes(), original_config)
            self.assertEqual(env_path.read_bytes(), original_env)
        self.assertIn("config replace failed", status)
        self.assertNotIn("new-key", status)
        self.assertNotIn("old-key", status)

    def test_transaction_rolls_back_both_files_when_env_temp_write_fails_without_leaking_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path, env_path = Path(tmp) / "config.yaml", Path(tmp) / ".env"
            config_path.write_bytes(b"original: config\r\n")
            env_path.write_bytes(b"UNRELATED_ENV=preserve\r\nOPENAI_IMAGE_API_KEY=old-key\r\n")
            original_config, original_env = config_path.read_bytes(), env_path.read_bytes()
            real_write_text = Path.write_text

            def fail_env_temp_write(path, *args, **kwargs):
                if path.name == "..env.tmp":
                    raise OSError("env temp write failed")
                return real_write_text(path, *args, **kwargs)

            with patch.object(webui.Path, "write_text", autospec=True, side_effect=fail_env_temp_write):
                status = save_api_settings(
                    "", "", "", "", "new-key",
                    "https://gemini.test/v1beta", "gemini-test",
                    "https://nvidia.test/v1", "nvidia-test",
                    "https://hapiopen.cc/v1", "gpt-image-2", "1K",
                    "openai_image", "openai_image",
                    config_path=config_path, env_path=env_path,
                )

            self.assertEqual(config_path.read_bytes(), original_config)
            self.assertEqual(env_path.read_bytes(), original_env)
        self.assertIn("env temp write failed", status)
        self.assertNotIn("new-key", status)
        self.assertNotIn("old-key", status)

    def test_transaction_rolls_back_config_and_openai_image_key_on_second_replace_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path, env_path = Path(tmp) / "config.yaml", Path(tmp) / ".env"
            config_path.write_text(
                "gemini_api:\n  base_url: https://gemini.test/v1beta\n  model: gemini-test\n"
                "nvidia_api:\n  base_url: https://nvidia.test/v1\n  model: nvidia-test\n",
                encoding="utf-8",
            )
            env_path.write_text("OPENAI_IMAGE_API_KEY=old-key\n", encoding="utf-8")
            original_config, original_env = config_path.read_bytes(), env_path.read_bytes()
            real_replace, call_count = os.replace, 0

            def fail_second_replace(source, destination):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("injected env replace failure")
                return real_replace(source, destination)

            with patch("webui.os.replace", side_effect=fail_second_replace):
                status = save_api_settings(
                    "", "", "", "", "new-key",
                    "https://gemini.test/v1beta", "gemini-test",
                    "https://nvidia.test/v1", "nvidia-test",
                    "https://hapiopen.cc/v1", "gpt-image-2", "1K",
                    "openai_image", "openai_image",
                    config_path=config_path, env_path=env_path,
                )

            self.assertIn("\u5931\u8d25", status)
            self.assertEqual(config_path.read_bytes(), original_config)
            self.assertEqual(env_path.read_bytes(), original_env)
        self.assertNotIn("new-key", status)
        self.assertNotIn("old-key", status)

    @patch("webui.AgentSkill")
    def test_detect_lovart_unlimited_models_returns_only_enabled_supported_models(self, skill_cls):
        skill_cls.return_value.query_mode.return_value = {
            "unlimited": True,
            "unlimited_enable": True,
            "unlimited_list": [
                {
                    "name": "Nano Banana",
                    "status": 1,
                    "alias_list": ["generate_image_nano_banana"],
                },
                {
                    "name": "Nano Banana Pro",
                    "status": 0,
                    "extraItem": "1K",
                    "alias_list": ["generate_image_nano_banana_pro"],
                },
                {
                    "name": "Unknown",
                    "status": 1,
                    "alias_list": ["generate_image_unknown"],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("lovart:\n  base_url: https://lovart.test\n", encoding="utf-8")
            status, catalog = webui.detect_lovart_unlimited_models(
                "access", "secret", config_path=path
            )

        self.assertIn("检测到 1 个", status)
        self.assertEqual([item["model"] for item in catalog], ["nano_banana"])
        skill_cls.assert_called_once()

    def test_save_lovart_unlimited_models_uses_user_order_and_preserves_config(self):
        catalog = [
            {
                "model": "nano_banana",
                "label": "Nano Banana",
                "restriction": "",
            },
            {
                "model": "nano_banana_pro",
                "label": "Nano Banana Pro",
                "restriction": "1K",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "lovart:\n  image_model: auto\nother:\n  keep: true\n",
                encoding="utf-8",
            )
            status = webui.save_lovart_unlimited_models(
                ["nano_banana", "nano_banana_pro"],
                ["nano_banana_pro", "nano_banana"],
                catalog,
                config_path=path,
            )
            saved = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertIn("已保存", status)
        self.assertEqual(
            saved["lovart"]["unlimited_models"],
            ["nano_banana_pro", "nano_banana"],
        )
        self.assertEqual(saved["lovart"]["image_model"], "auto")
        self.assertTrue(saved["other"]["keep"])

    def test_move_lovart_model_changes_only_requested_position(self):
        order = ["nano_banana", "nano_banana_2", "nano_banana_pro"]
        self.assertEqual(
            webui.move_lovart_model("nano_banana_pro", order, -1),
            ["nano_banana", "nano_banana_pro", "nano_banana_2"],
        )
        self.assertEqual(order, ["nano_banana", "nano_banana_2", "nano_banana_pro"])

    def test_save_failed_retry_settings_preserves_other_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "lovart:\n  image_model: auto\nother:\n  keep: true\n",
                encoding="utf-8",
            )
            status = save_failed_retry_settings(
                "infinite",
                7,
                4.5,
                ["network", "lovart_no_artifacts", "other"],
                config_path=path,
            )
            saved = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertIn("已保存", status)
        self.assertEqual(saved["lovart"]["failed_retry_mode"], "infinite")
        self.assertEqual(saved["lovart"]["failed_retry_rounds"], 7)
        self.assertEqual(saved["lovart"]["failed_retry_delay"], 4.5)
        self.assertEqual(
            saved["lovart"]["failed_retry_error_types"],
            ["network", "lovart_no_artifacts", "other"],
        )
        self.assertEqual(saved["lovart"]["image_model"], "auto")
        self.assertTrue(saved["other"]["keep"])

    def test_invalid_failed_retry_settings_do_not_modify_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("lovart:\n  image_model: auto\n", encoding="utf-8")
            before = path.read_bytes()
            status = save_failed_retry_settings(
                "finite", 0, 15, ["network"], config_path=path
            )
            self.assertEqual(path.read_bytes(), before)
        self.assertIn("❌", status)

    @patch("webui.load_config", return_value={})
    def test_failed_retry_controls_have_defaults_and_save_event(self, _load_config):
        demo = build_ui()
        components = demo.config["components"]
        by_label = {
            item.get("props", {}).get("label"): item
            for item in components
            if item.get("props", {}).get("label")
        }
        self.assertEqual(by_label["重试模式"]["props"]["value"], "finite")
        self.assertEqual(by_label["最多补偿轮次（不含首次）"]["props"]["value"], 2)
        self.assertEqual(by_label["每轮重试间隔（秒）"]["props"]["value"], 15.0)
        self.assertEqual(
            set(by_label["允许重试的错误类型"]["props"]["value"]),
            {"lovart_service", "network", "timeout", "gemini_page", "gemini_upload"},
        )

        component_labels = {
            item["id"]: item.get("props", {}).get("label")
            for item in components
        }
        save_event = next(
            item
            for item in demo.config["dependencies"]
            if item["api_name"] == "save_failed_retry_settings"
        )
        self.assertEqual(
            [component_labels[item] for item in save_event["inputs"]],
            [
                "重试模式",
                "最多补偿轮次（不含首次）",
                "每轮重试间隔（秒）",
                "允许重试的错误类型",
            ],
        )

    def test_save_gemini_browser_settings_persists_fallback_without_losing_other_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "gemini:\n  thinking_mode: true\nother:\n  keep: true\n",
                encoding="utf-8",
            )
            status = webui.save_gemini_browser_settings(True, config_path=path)
            saved = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertIn("已保存", status)
        self.assertTrue(saved["gemini"]["allow_regular_chat_fallback"])
        self.assertTrue(saved["gemini"]["thinking_mode"])
        self.assertTrue(saved["other"]["keep"])

    @patch("webui.load_config", return_value={})
    def test_gemini_regular_chat_fallback_setting_defaults_off_in_ui(self, _load_config):
        demo = build_ui()
        components = demo.config["components"]
        checkbox = next(
            item
            for item in components
            if item.get("props", {}).get("label")
            == "临时聊天不可用时，允许使用普通聊天继续"
        )
        self.assertEqual(checkbox["type"], "checkbox")
        self.assertFalse(checkbox["props"]["value"])

        component_labels = {
            item["id"]: item.get("props", {}).get("label")
            for item in components
        }
        save_event = next(
            item
            for item in demo.config["dependencies"]
            if item["api_name"] == "save_gemini_browser_settings"
        )
        self.assertEqual(
            [component_labels[item] for item in save_event["inputs"]],
            ["临时聊天不可用时，允许使用普通聊天继续"],
        )

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

    def test_locked_preview_mentions_selected_image_provider_excel_and_lovart(self):
        preview = reset_prompt_settings_form()[-1]
        self.assertIn("所有提示词生成模型", preview)
        self.assertIn("Excel", preview)
        self.assertIn("Lovart", preview)
        self.assertIn("用户选择的图片生成提供商", preview)
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

        for api_name, endpoint_label, provider_model_label in (
            ("probe_gemini_model", "Gemini API 地址", "Gemini 模型"),
            ("probe_nvidia_model", "NVIDIA API 地址", "NVIDIA 模型"),
        ):
            event = dependencies[api_name]
            input_labels = {component_labels[item] for item in event["inputs"]}
            output_labels = [component_labels[item] for item in event["outputs"]]
            self.assertIn(endpoint_label, input_labels)
            self.assertIn("提示词引擎", input_labels)
            self.assertEqual(len(event["outputs"]), 4)
            self.assertIn(provider_model_label, output_labels)
            self.assertIn("提示词模型", output_labels)
            self.assertIn(None, output_labels)  # runtime catalog State

        run_event = dependencies["run_process"]
        run_labels = {component_labels[item] for item in run_event["inputs"]}
        self.assertIn("Gemini API 地址", run_labels)
        self.assertIn("NVIDIA API 地址", run_labels)

        detect_event = dependencies["detect_lovart_unlimited_models"]
        self.assertEqual(
            [component_labels[item] for item in detect_event["inputs"]],
            ["LOVART_ACCESS_KEY", "LOVART_SECRET_KEY"],
        )
        save_lovart_event = dependencies["save_lovart_unlimited_models"]
        self.assertIn(
            "启用的无限模型",
            [component_labels[item] for item in save_lovart_event["inputs"]],
        )

    def test_api_save_and_run_offer_injectable_env_and_config_paths_without_new_ui_inputs(self):
        save_parameters = inspect.signature(save_api_settings).parameters
        run_parameters = inspect.signature(run_process).parameters
        self.assertIn("config_path", save_parameters)
        self.assertIn("env_path", save_parameters)
        self.assertIn("config_path", run_parameters)
        self.assertIn("env_path", run_parameters)
        self.assertIn("env_path", inspect.signature(save_env).parameters)

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
                "gemini-key", "nvidia-key", "lovart-access", "lovart-secret", "",
                "https://gemini.proxy.test/v1beta", "gemini-custom",
                "https://nvidia.proxy.test/v1", "nvidia/custom",
                "https://hapiopen.cc/v1", "gpt-image-2", "1K", "lovart", "lovart",
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
                "gemini-key", "nvidia-key", "lovart-access", "lovart-secret", "",
                "not-a-url", "gemini-custom",
                "https://nvidia.proxy.test/v1", "nvidia/custom",
                "https://hapiopen.cc/v1", "gpt-image-2", "1K", "lovart", "lovart",
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

    @patch("webui.subprocess.Popen")
    @patch("webui.save_config")
    @patch("webui.load_config")
    @patch("webui.save_env")
    def test_run_process_forwards_provider_overrides_as_literal_subprocess_argv(
        self, _save_env, load_config, _save_config, popen
    ):
        load_config.return_value = {
            "gemini_api": {"model": "gemini-model"},
            "nvidia_api": {"model": "nvidia-model"},
        }
        child = Mock()
        child.stdout = io.StringIO("")
        child.poll.return_value = 0
        popen.return_value = child

        output = list(run_process(
            None, "output folder with spaces", "gemini_api", "gemini-model",
            "unlimited", "auto", "https://gemini.test/v1beta",
            "https://nvidia.test/v1", "gemini-key", "nvidia-key", "", "",
            "openai-key", "https://hapiopen.cc/v1", "gpt-image-2", "2K",
            "openai_image", "lovart", False,
        ))

        self.assertTrue(output)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[argv.index("--support-provider") + 1], "openai_image")
        self.assertEqual(argv[argv.index("--detail-provider") + 1], "lovart")
        self.assertIn("output folder with spaces", popen.call_args.kwargs["env"]["LOVART_OUTPUT_DIR"])

    @patch("webui.subprocess.Popen")
    def test_lovart_only_launch_ignores_malformed_unused_gpt_settings(self, popen):
        child = Mock()
        child.stdout = io.StringIO("")
        child.poll.return_value = 0
        popen.return_value = child
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            env_path = Path(tmp) / ".env"
            config_path.write_text(
                "gemini_api:\n  base_url: https://gemini.test/v1beta\n  model: gemini-model\n"
                "nvidia_api:\n  base_url: https://nvidia.test/v1\n  model: nvidia-model\n"
                "openai_image:\n  base_url: deliberately-malformed\n  model: saved-model\n  resolution: BAD\n",
                encoding="utf-8",
            )

            output = list(run_process(
                None, "output", "gemini_api", "gemini-model", "unlimited", "auto",
                "https://gemini.test/v1beta", "https://nvidia.test/v1",
                "gemini-key", "nvidia-key", "lovart-access", "lovart-secret",
                "", "not-a-url", "", "BAD", "lovart", "lovart", False,
                config_path=config_path,
                env_path=env_path,
            ))
            saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertTrue(any("Starting" in item for item in output))
        self.assertEqual(saved["image_generation"], {
            "support_provider": "lovart",
            "detail_provider": "lovart",
        })
        self.assertEqual(saved["openai_image"]["base_url"], "deliberately-malformed")
        self.assertEqual(saved["openai_image"]["resolution"], "BAD")

    def test_selected_gpt_route_rejects_malformed_or_missing_selected_settings(self):
        cases = [
            ("not-a-url", "gpt-image-2", "1K", "test-key"),
            ("https://hapiopen.cc/v1", "gpt-image-2", "1K", ""),
        ]
        for base_url, model, resolution, key in cases:
            with self.subTest(base_url=base_url, key=bool(key)), tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / "config.yaml"
                env_path = Path(tmp) / ".env"
                config_path.write_text(
                    "gemini_api:\n  base_url: https://gemini.test/v1beta\n  model: gemini-model\n"
                    "nvidia_api:\n  base_url: https://nvidia.test/v1\n  model: nvidia-model\n",
                    encoding="utf-8",
                )
                process = run_process(
                    None, "output", "gemini_api", "gemini-model", "unlimited", "auto",
                    "https://gemini.test/v1beta", "https://nvidia.test/v1",
                    "gemini-key", "nvidia-key", "", "",
                    key, base_url, model, resolution,
                    "openai_image", "lovart", False,
                    config_path=config_path,
                    env_path=env_path,
                )
                with patch(
                    "webui.subprocess.Popen",
                    side_effect=AssertionError(
                        "invalid selected GPT settings launched subprocess"
                    ),
                ):
                    status = next(process)

            self.assertNotIn("Starting", status)
            self.assertTrue("GPT Image" in status or "API" in status)

    @patch("webui.subprocess.Popen")
    def test_start_ui_clear_flow_updates_indicator_and_resets_checkbox(self, popen):
        child = Mock()
        child.stdout = io.StringIO("")
        child.poll.return_value = 0
        popen.return_value = child
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            env_path = Path(tmp) / ".env"
            config_path.write_text(
                "gemini_api:\n  base_url: https://gemini.test/v1beta\n  model: gemini-model\n"
                "nvidia_api:\n  base_url: https://nvidia.test/v1\n  model: nvidia-model\n",
                encoding="utf-8",
            )
            env_path.write_text("OPENAI_IMAGE_API_KEY=old-secret\n", encoding="utf-8")

            updates = list(webui.run_process_from_ui(
                None, "output", "gemini_api", "gemini-model", "unlimited", "auto",
                "https://gemini.test/v1beta", "https://nvidia.test/v1",
                "gemini-key", "nvidia-key", "lovart-access", "lovart-secret",
                "", "malformed-unused", "", "BAD", "lovart", "lovart", True,
                config_path=config_path,
                env_path=env_path,
            ))

        synchronized = next(
            item for item in updates
            if isinstance(item, tuple) and isinstance(item[1], str)
        )
        self.assertIn("未保存", synchronized[1])
        self.assertFalse(synchronized[2])
        self.assertNotIn("old-secret", str(updates))

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
        self.assertIn("测试失败", choices[0][0])

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
        self.assertIn("测试失败", choices[0][0])

    @patch("webui.test_selected_model")
    def test_probe_success_keeps_custom_model_when_runtime_catalog_is_empty(self, test_model):
        test_model.return_value = ModelTestResult(True, "模型可用", 9)
        status, choices, selected, catalog = probe_provider_model(
            "gemini", "key", "https://google.test/v1beta", "custom/gemini-model", []
        )
        self.assertIn("模型可用", status)
        self.assertEqual(selected, "custom/gemini-model")
        self.assertEqual([value for _, value in choices], ["custom/gemini-model"])
        self.assertEqual(catalog[0]["model_id"], "custom/gemini-model")
        self.assertEqual(catalog[0]["provider"], "gemini")
        self.assertEqual(catalog[0]["image_input_status"], "verified")
        self.assertIn("图片已验证支持", choices[0][0])

    @patch("webui.test_selected_model")
    def test_probe_normalizes_custom_model_id_for_request_catalog_status_and_selection(self, test_model):
        test_model.return_value = ModelTestResult(True, "模型可用", 8)
        status, choices, selected, catalog = probe_provider_model(
            "gemini", "key", "https://google.test/v1beta", "  custom/gemini-model  ", []
        )
        self.assertIn("模型可用", status)
        test_model.assert_called_once_with(
            "gemini", "key", "https://google.test/v1beta", "custom/gemini-model"
        )
        self.assertEqual(selected, "custom/gemini-model")
        self.assertEqual([value for _, value in choices], ["custom/gemini-model"])
        self.assertEqual(catalog[0]["model_id"], "custom/gemini-model")
        self.assertEqual(catalog[0]["image_input_status"], "verified")

    def test_probe_invalid_model_id_uses_user_error_without_forging_catalog(self):
        status, choices, _selected, catalog = probe_provider_model(
            "nvidia", "key", "https://nvidia.test/v1", "bad\nmodel", []
        )
        self.assertIn("模型 ID 不能包含换行符", status)
        self.assertEqual(choices, [])
        self.assertEqual(catalog, [])

    @patch("webui.test_selected_model")
    def test_probe_error_keeps_custom_model_and_uses_test_failed_label(self, test_model):
        test_model.side_effect = ModelProviderError("network", "网络连接失败")
        status, choices, selected, catalog = probe_provider_model(
            "nvidia", "key", "https://nvidia.test/v1", "custom/nvidia-model", []
        )
        self.assertIn("网络连接失败", status)
        self.assertEqual(selected, "custom/nvidia-model")
        self.assertEqual([value for _, value in choices], ["custom/nvidia-model"])
        self.assertEqual(catalog[0]["image_input_status"], "failed")
        self.assertIn("测试失败", choices[0][0])
        self.assertNotIn("图片不支持", choices[0][0])

    def test_save_api_settings_rolls_back_both_files_when_second_replace_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                config_path = Path("config.yaml")
                env_path = Path(".env")
                config_path.write_text("original: config\n", encoding="utf-8")
                env_path.write_text("ORIGINAL_ENV=1\n", encoding="utf-8")
                before_config = config_path.read_bytes()
                before_env = env_path.read_bytes()
                real_replace = os.replace
                replace_count = 0

                def fail_second_replace(source, destination):
                    nonlocal replace_count
                    replace_count += 1
                    if replace_count == 2:
                        raise OSError("second target failed")
                    return real_replace(source, destination)

                with patch("webui.os.replace", side_effect=fail_second_replace):
                    status = save_api_settings(
                        "gemini-key", "nvidia-key", "lovart-access", "lovart-secret", "",
                        "https://gemini.test/v1beta", "gemini-model",
                        "https://nvidia.test/v1", "nvidia-model",
                        "https://hapiopen.cc/v1", "gpt-image-2", "1K", "lovart", "lovart",
                    )

                self.assertIn("second target failed", status)
                self.assertEqual(config_path.read_bytes(), before_config)
                self.assertEqual(env_path.read_bytes(), before_env)
            finally:
                os.chdir(original_cwd)

    def test_run_process_rolls_back_both_files_when_second_replace_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                config_path = Path("config.yaml")
                env_path = Path(".env")
                config_path.write_text(
                    "gemini_api:\n  model: gemini-old\n"
                    "nvidia_api:\n  model: nvidia-old\n",
                    encoding="utf-8",
                )
                env_path.write_text("ORIGINAL_ENV=1\n", encoding="utf-8")
                before_config = config_path.read_bytes()
                before_env = env_path.read_bytes()
                real_replace = os.replace
                replace_count = 0

                def fail_second_replace(source, destination):
                    nonlocal replace_count
                    replace_count += 1
                    if replace_count == 2:
                        raise OSError("second target failed")
                    return real_replace(source, destination)

                with patch("webui.os.replace", side_effect=fail_second_replace):
                    process = run_process(
                        None, "output", "gemini_api", "gemini-new", "unlimited", "auto",
                        "https://gemini.test/v1beta", "https://nvidia.test/v1",
                        "gemini-key", "nvidia-key", "lovart-access", "lovart-secret",
                    )
                    status = next(process)

                self.assertIn("second target failed", status)
                self.assertEqual(config_path.read_bytes(), before_config)
                self.assertEqual(env_path.read_bytes(), before_env)
            finally:
                os.chdir(original_cwd)

    def test_transaction_removes_new_files_when_second_replace_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                real_replace = os.replace
                replace_count = 0

                def fail_second_replace(source, destination):
                    nonlocal replace_count
                    replace_count += 1
                    if replace_count == 2:
                        raise OSError("second target failed")
                    return real_replace(source, destination)

                with patch("webui.os.replace", side_effect=fail_second_replace):
                    status = save_api_settings(
                        "gemini-key", "nvidia-key", "lovart-access", "lovart-secret", "",
                        "https://gemini.test/v1beta", "gemini-model",
                        "https://nvidia.test/v1", "nvidia-model",
                        "https://hapiopen.cc/v1", "gpt-image-2", "1K", "lovart", "lovart",
                    )

                self.assertIn("second target failed", status)
                self.assertFalse(Path("config.yaml").exists())
                self.assertFalse(Path(".env").exists())
            finally:
                os.chdir(original_cwd)

    def test_run_transaction_removes_new_files_when_second_replace_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                real_replace = os.replace
                replace_count = 0

                def fail_second_transaction_replace(source, destination):
                    nonlocal replace_count
                    replace_count += 1
                    # load_config creates its default with call 1; config transaction save is call 2.
                    # Fail the environment transaction save after allowing those two config writes.
                    if replace_count == 3:
                        raise OSError("second target failed")
                    return real_replace(source, destination)

                with patch("webui.os.replace", side_effect=fail_second_transaction_replace):
                    process = run_process(
                        None, "output", "gemini_api", "gemini-new", "unlimited", "auto",
                        "https://gemini.test/v1beta", "https://nvidia.test/v1",
                        "gemini-key", "nvidia-key", "lovart-access", "lovart-secret",
                    )
                    status = next(process)

                self.assertIn("second target failed", status)
                self.assertFalse(Path("config.yaml").exists())
                self.assertFalse(Path(".env").exists())
            finally:
                os.chdir(original_cwd)

    def test_transaction_reports_primary_and_restore_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            original_cwd = Path.cwd()
            os.chdir(tmp)
            try:
                original_config = b"original: config\n"
                Path("config.yaml").write_bytes(original_config)
                Path(".env").write_text("ORIGINAL_ENV=1\n", encoding="utf-8")
                rollback_path = Path(".config.yaml.rollback.tmp").resolve()
                real_replace = os.replace
                replace_count = 0

                def fail_second_save_and_first_restore(source, destination):
                    nonlocal replace_count
                    replace_count += 1
                    if replace_count == 2:
                        raise OSError("env write failed")
                    if replace_count == 3:
                        raise OSError("rollback denied")
                    return real_replace(source, destination)

                with patch("webui.os.replace", side_effect=fail_second_save_and_first_restore):
                    status = save_api_settings(
                        "gemini-key", "nvidia-key", "lovart-access", "lovart-secret", "",
                        "https://gemini.test/v1beta", "gemini-model",
                        "https://nvidia.test/v1", "nvidia-model",
                        "https://hapiopen.cc/v1", "gpt-image-2", "1K", "lovart", "lovart",
                    )
                self.assertTrue(rollback_path.exists())
                self.assertEqual(rollback_path.read_bytes(), original_config)
                self.assertIn(str(rollback_path), status)
            finally:
                os.chdir(original_cwd)
        self.assertIn("env write failed", status)
        self.assertIn("恢复也失败", status)
        self.assertIn("rollback denied", status)
