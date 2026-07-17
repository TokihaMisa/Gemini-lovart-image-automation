import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main

from gemini_browser_session import (
    GeminiPageState,
    LoginStatus,
    clear_stale_login_runtime,
    inspect_gemini_page,
    login_runtime_paths,
    navigate_gemini_with_retry,
    read_login_status,
    request_login_helper_close,
    write_login_status,
)
from network_retry import RetryPolicy


class FakePage:
    def __init__(self, url, payload):
        self.url = url
        self.payload = payload
        self.goto_calls = 0

    def evaluate(self, _script):
        return self.payload

    def goto(self, _url, **_kwargs):
        self.goto_calls += 1


class GeminiBrowserSessionTests(unittest.TestCase):
    def test_accounts_url_is_waiting_login_even_if_editor_like_node_exists(self):
        page = FakePage("https://accounts.google.com/signin", {
            "language": "es", "has_editor": True, "has_login_prompt": True,
            "has_loading": False, "controls": [],
        })

        status = inspect_gemini_page(page)

        self.assertEqual(status.state, GeminiPageState.WAITING_LOGIN)
        self.assertFalse(status.ready)

    def test_structural_editor_marks_spanish_page_ready(self):
        page = FakePage("https://gemini.google.com/app", {
            "language": "es-ES", "has_editor": True, "has_login_prompt": False,
            "has_loading": False, "controls": ["Rápido", "Adjuntar archivos"],
        })

        status = inspect_gemini_page(page)

        self.assertEqual(status.state, GeminiPageState.READY)
        self.assertTrue(status.ready)
        self.assertEqual(status.language, "es-ES")

    def test_non_login_account_path_on_gemini_remains_ready(self):
        page = FakePage("https://gemini.google.com/app/account", {
            "language": "en", "has_editor": True, "has_login_prompt": False,
            "has_loading": False, "controls": [],
        })

        status = inspect_gemini_page(page)

        self.assertEqual(status.state, GeminiPageState.READY)
        self.assertTrue(status.ready)

    def test_atomic_status_round_trip_and_close_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            paths = login_runtime_paths(config)
            status = LoginStatus.create(
                GeminiPageState.READY,
                True,
                "https://gemini.google.com/app",
                "es",
                "ready",
                pid=42,
            )

            write_login_status(paths.status_path, status)

            self.assertEqual(read_login_status(paths.status_path), status)
            request_login_helper_close(paths.close_request_path)
            self.assertTrue(paths.close_request_path.exists())
            self.assertFalse(paths.status_path.with_suffix(".tmp").exists())

    @patch("gemini_browser_session.process_is_alive", return_value=False)
    def test_stale_status_is_cleared_without_killing_any_browser(self, _alive):
        with tempfile.TemporaryDirectory() as tmp:
            paths = login_runtime_paths(Path(tmp) / "config.yaml")
            write_login_status(
                paths.status_path,
                LoginStatus.create(GeminiPageState.READY, True, "", "", "", pid=99),
            )

            self.assertTrue(clear_stale_login_runtime(paths))
            self.assertFalse(paths.status_path.exists())

    def test_navigation_retries_transient_failure_then_requires_ready_page(self):
        page = FakePage("https://gemini.google.com/app", {
            "language": "en", "has_editor": True, "has_login_prompt": False,
            "has_loading": False, "controls": [],
        })
        failures = [RuntimeError("net::ERR_CONNECTION_RESET")]
        original_goto = page.goto

        def flaky_goto(url, **kwargs):
            if failures:
                raise failures.pop()
            return original_goto(url, **kwargs)

        page.goto = flaky_goto

        with patch("network_retry.time.sleep"):
            status = navigate_gemini_with_retry(
                page, "https://gemini.google.com", RetryPolicy()
            )

        self.assertTrue(status.ready)
        self.assertEqual(page.goto_calls, 1)

    def test_navigation_retries_when_page_readiness_times_out(self):
        page = FakePage("https://gemini.google.com/app", {
            "language": "en", "has_editor": False, "has_login_prompt": False,
            "has_loading": True, "controls": [],
        })
        policy = RetryPolicy(network_attempts=2, page_ready_timeout=0, retry_delays=(0,))

        with patch("network_retry.time.sleep"):
            with self.assertRaises(TimeoutError):
                navigate_gemini_with_retry(page, "https://gemini.google.com", policy)

        self.assertEqual(page.goto_calls, 2)

    def test_formal_browser_flow_uses_shared_launch_and_navigation_apis(self):
        class FormalPage:
            url = "https://gemini.google.com/app"

            def goto(self, _url, **_kwargs):
                pass

            def wait_for_timeout(self, _milliseconds):
                pass

        class FormalContext:
            def __init__(self):
                self.pages = [FormalPage()]
                self.launch_options = None
                self.closed = False

            def new_page(self):
                return self.pages[0]

            def close(self):
                self.closed = True

        class FormalPlaywrightManager:
            def __init__(self, context):
                self.context = context
                self.playwright = type("Playwright", (), {
                    "chromium": type("Chromium", (), {
                        "launch_persistent_context": self.launch,
                    })(),
                })()

            def launch(self, **kwargs):
                self.context.launch_options = kwargs
                return self.context

            def __enter__(self):
                return self.playwright

            def __exit__(self, _exc_type, _exc, _traceback):
                return False

        class Logger:
            def info(self, _message):
                pass

            def warning(self, _message):
                pass

        config = {
            "browser": {"user_data_dir": "browser_profile"},
            "gemini": {"base_url": "https://gemini.google.com"},
        }
        context = FormalContext()
        ready = LoginStatus.create(
            GeminiPageState.READY, True, "https://gemini.google.com/app", "en", "ready"
        )
        launch_options = {"user_data_dir": "shared-profile"}
        config_path = Path("custom-settings") / "browser-config.yaml"
        with patch("main.sync_playwright", return_value=FormalPlaywrightManager(context)), patch(
            "main.build_browser_launch_options", return_value=launch_options
        ) as build_options, patch.object(
            main, "navigate_gemini_with_retry", create=True, return_value=ready
        ) as navigate, patch("main._process_products", return_value=(1, 0, 0, 0)):
            result = main._run_browser_flow(
                config,
                products=[object()],
                lovart=object(),
                logger=Logger(),
                run_dir=Path("runs/test"),
                wait_for_ready=False,
                config_path=config_path,
            )

        self.assertEqual(result, (1, 0, 0, 0))
        build_options.assert_called_once_with(config, config_path=config_path)
        navigate.assert_called_once()
        self.assertEqual(context.launch_options, launch_options)
