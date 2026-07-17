import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main

from gemini_browser_session import (
    GeminiPageState,
    LoginStatus,
    acquire_login_helper_owner,
    build_login_helper_command,
    clear_stale_login_runtime,
    inspect_gemini_page,
    login_runtime_paths,
    navigate_gemini_with_retry,
    process_is_alive,
    read_login_status,
    release_login_helper_owner,
    request_login_helper_close,
    run_login_helper,
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

    def test_helper_command_matches_source_and_frozen_entry_points(self):
        source = build_login_helper_command("config.yaml", executable="python.exe", frozen=False)
        frozen = build_login_helper_command("config.yaml", executable="Lovart_Auto.exe", frozen=True)

        self.assertEqual(source[:2], ["python.exe", str(Path("app.py").resolve())])
        self.assertEqual(
            source[2:],
            ["--gemini-login-helper", "--config", str(Path("config.yaml").resolve())],
        )
        self.assertEqual(
            frozen,
            ["Lovart_Auto.exe", "--gemini-login-helper", "--config", str(Path("config.yaml").resolve())],
        )

    @patch("gemini_browser_session.time.sleep", side_effect=lambda _seconds: None)
    @patch("gemini_browser_session.navigate_gemini_with_retry")
    @patch("gemini_browser_session.build_browser_launch_options")
    @patch("gemini_browser_session.clear_stale_login_runtime")
    @patch("gemini_browser_session.login_helper_is_active", return_value=False)
    @patch("gemini_browser_session.sync_playwright")
    def test_login_helper_closes_ready_context_after_close_request(
        self,
        sync_playwright,
        _active,
        _clear_stale,
        build_options,
        navigate,
        _sleep,
    ):
        class Context:
            def __init__(self):
                self.closed = False

            def new_page(self):
                return object()

            def close(self):
                self.closed = True

        class Manager:
            def __init__(self, context):
                self.context = context

            def __enter__(self):
                return type("Playwright", (), {
                    "chromium": type("Chromium", (), {
                        "launch_persistent_context": lambda _self, **_kwargs: self.context,
                    })(),
                })()

            def __exit__(self, *_args):
                return False

        ready = LoginStatus.create(GeminiPageState.READY, True, "https://gemini.google.com/app", "en", "ready")
        context = Context()
        sync_playwright.return_value = Manager(context)
        build_options.return_value = {"user_data_dir": "profile"}
        navigate.return_value = ready

        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            request_login_helper_close(login_runtime_paths(config).close_request_path)
            with patch("gemini_browser_session.inspect_gemini_page", return_value=ready):
                result = run_login_helper(config)
            status = read_login_status(login_runtime_paths(config).status_path)

        self.assertEqual(result, 0)
        self.assertTrue(context.closed)
        self.assertEqual(status.state, GeminiPageState.CLOSED)
        self.assertFalse(status.ready)

    def test_profile_scoped_owner_lock_rejects_second_helper_without_overwriting_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_root = Path(tmp)
            shared_profile = config_root / "shared-profile"
            first_config = config_root / "one" / "config.yaml"
            second_config = config_root / "two" / "config.yaml"
            first_config.parent.mkdir(parents=True)
            second_config.parent.mkdir(parents=True)
            config_text = f"browser:\n  user_data_dir: {shared_profile}\n"
            first_config.write_text(config_text, encoding="utf-8")
            second_config.write_text(config_text, encoding="utf-8")
            first_paths = login_runtime_paths(first_config)
            second_paths = login_runtime_paths(second_config)

            owner = acquire_login_helper_owner(first_paths)
            self.assertIsNotNone(owner)
            self.assertEqual(first_paths.owner_lock_path, second_paths.owner_lock_path)
            starting = LoginStatus.create(GeminiPageState.STARTING, False, "", "", "starting")
            write_login_status(first_paths.status_path, starting)
            try:
                with patch("gemini_browser_session.sync_playwright") as sync_playwright:
                    self.assertEqual(run_login_helper(second_config), 1)
                sync_playwright.assert_not_called()
                self.assertEqual(read_login_status(first_paths.status_path), starting)
            finally:
                release_login_helper_owner(first_paths, owner)

    def test_windows_liveness_check_queries_process_without_signalling_it(self):
        class Kernel32:
            def OpenProcess(self, _access, _inherit, _pid):
                return 123

            def GetExitCodeProcess(self, _handle, exit_code):
                exit_code._obj.value = 259
                return 1

            def CloseHandle(self, _handle):
                return 1

        with patch("gemini_browser_session.os.name", "nt"), patch(
            "gemini_browser_session.ctypes.WinDLL", return_value=Kernel32(), create=True
        ), patch("gemini_browser_session.os.kill") as kill:
            self.assertTrue(process_is_alive(42))

        kill.assert_not_called()

    @patch("gemini_browser_session.process_is_alive", return_value=False)
    def test_owner_lock_recovers_only_confirmed_dead_owner(self, _is_alive):
        with tempfile.TemporaryDirectory() as tmp:
            paths = login_runtime_paths(Path(tmp) / "config.yaml")
            paths.owner_lock_path.parent.mkdir(parents=True)
            paths.owner_lock_path.write_text(
                '{"pid": 999999, "token": "dead-owner", "created_at": 0}',
                encoding="utf-8",
            )

            owner = acquire_login_helper_owner(paths)

            self.assertIsNotNone(owner)
            self.assertNotEqual(owner.token, "dead-owner")
            release_login_helper_owner(paths, owner)

    @patch("gemini_browser_session.navigate_gemini_with_retry")
    @patch("gemini_browser_session.build_browser_launch_options")
    @patch("gemini_browser_session.sync_playwright")
    def test_login_helper_records_closed_when_browser_is_manually_closed(
        self, sync_playwright, build_options, navigate
    ):
        class Page:
            url = "https://gemini.google.com/app"

            def is_closed(self):
                return True

        class Context:
            def __init__(self):
                self.closed = False
                self.pages = [Page()]

            def close(self):
                self.closed = True

        class Manager:
            def __init__(self, context):
                self.context = context

            def __enter__(self):
                return type("Playwright", (), {
                    "chromium": type("Chromium", (), {
                        "launch_persistent_context": lambda _self, **_kwargs: self.context,
                    })(),
                })()

            def __exit__(self, *_args):
                return False

        context = Context()
        sync_playwright.return_value = Manager(context)
        build_options.return_value = {"user_data_dir": "profile"}
        navigate.return_value = LoginStatus.create(
            GeminiPageState.READY, True, Page.url, "en", "ready"
        )
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            result = run_login_helper(config)
            status = read_login_status(login_runtime_paths(config).status_path)

        self.assertEqual(result, 0)
        self.assertEqual(status.state, GeminiPageState.CLOSED)
        self.assertFalse(status.ready)
        self.assertTrue(context.closed)
