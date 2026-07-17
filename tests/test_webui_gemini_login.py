import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from gemini_browser_session import GeminiPageState, LoginStatus, login_runtime_paths, write_login_status
from webui import (
    build_ui,
    check_gemini_login_and_close,
    guard_gemini_browser_task,
    load_config,
    open_gemini_login_browser,
)


class WebUIGeminiLoginTests(unittest.TestCase):
    def test_webui_created_config_uses_exact_balanced_browser_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(Path(tmp) / "config.yaml")

        self.assertEqual(
            {key: config["browser"].get(key) for key in (
                "network_attempts", "page_ready_timeout", "product_attempts", "retry_delays",
            )},
            {
                "network_attempts": 5,
                "page_ready_timeout": 90,
                "product_attempts": 2,
                "retry_delays": [3, 6, 12, 20],
            },
        )

    @patch("webui.subprocess.Popen")
    @patch("webui.login_helper_is_active", return_value=False)
    def test_open_button_starts_one_helper_and_returns_waiting_status(self, _active, popen):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            message = open_gemini_login_browser(config)

        self.assertEqual(popen.call_count, 1)
        self.assertIn("登录浏览器已打开", message)

    @patch("webui.login_helper_is_active", return_value=True)
    def test_open_button_does_not_start_duplicate_helper(self, _active):
        with patch("webui.subprocess.Popen") as popen:
            message = open_gemini_login_browser("config.yaml")

        popen.assert_not_called()
        self.assertIn("已经打开", message)

    @patch("webui.login_helper_is_active", return_value=False)
    def test_concurrent_open_callbacks_start_at_most_one_helper(self, _active):
        with tempfile.TemporaryDirectory() as tmp, patch("webui.subprocess.Popen") as popen:
            config = Path(tmp) / "config.yaml"
            popen.side_effect = lambda *_args, **_kwargs: time.sleep(0.05)
            with ThreadPoolExecutor(max_workers=2) as executor:
                messages = list(executor.map(lambda _unused: open_gemini_login_browser(config), range(2)))

        self.assertEqual(popen.call_count, 1)
        self.assertEqual(sum("已打开" in message for message in messages), 1)
        self.assertEqual(sum("已经打开" in message for message in messages), 1)

    def test_check_does_not_close_when_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            paths = login_runtime_paths(config)
            write_login_status(paths.status_path, LoginStatus.create(
                GeminiPageState.WAITING_LOGIN, False, "https://accounts.google.com", "es", "等待登录", pid=42
            ))
            with patch("webui.login_helper_is_active", return_value=True):
                message = check_gemini_login_and_close(config)
            self.assertFalse(paths.close_request_path.exists())

        self.assertIn("尚未完成登录", message)

    def test_ready_check_requests_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            paths = login_runtime_paths(config)
            write_login_status(paths.status_path, LoginStatus.create(
                GeminiPageState.READY, True, "https://gemini.google.com/app", "es", "ready", pid=42
            ))
            with patch("webui.login_helper_is_active", return_value=True):
                message = check_gemini_login_and_close(config)
            self.assertTrue(paths.close_request_path.exists())

        self.assertIn("登录已确认", message)

    def test_browser_task_guard_blocks_active_helper_only_for_browser_source(self):
        with patch("webui.login_helper_is_active", return_value=True):
            self.assertIn("登录浏览器", guard_gemini_browser_task("gemini_browser"))
            self.assertIsNone(guard_gemini_browser_task("gemini_api"))
            self.assertIsNone(guard_gemini_browser_task("nvidia"))

    @patch("webui.load_config", return_value={})
    def test_real_gradio_event_graph_exposes_login_buttons_and_stable_api_names(self, _load_config):
        demo = build_ui()
        try:
            config = demo.get_config_file()
        finally:
            demo.close()

        components = config["components"]
        labels = {component.get("props", {}).get("value") for component in components}
        self.assertIn("打开 Gemini 登录浏览器", labels)
        self.assertIn("检查登录并关闭浏览器", labels)
        status_components = [
            component for component in components
            if component.get("props", {}).get("elem_id") == "gemini-login-status"
        ]
        self.assertEqual(len(status_components), 1)
        self.assertEqual(
            status_components[0]["props"].get("value"),
            "Gemini 登录浏览器尚未打开。",
        )
        dependencies = config["dependencies"]
        login_events = [
            dependency for dependency in dependencies
            if dependency.get("api_name") in {"open_gemini_login_browser", "check_gemini_login_and_close"}
        ]
        self.assertEqual({event["api_name"] for event in login_events}, {
            "open_gemini_login_browser", "check_gemini_login_and_close",
        })
        self.assertTrue(all(len(event["outputs"]) == 1 for event in login_events))
