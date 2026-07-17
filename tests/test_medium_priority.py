import csv
import json
import os
import ssl
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from PIL import Image

import main
from excel_reader import resolve_image_scan_config
from gemini_bot import (
    EXTENDED_THINKING_TERMS,
    MODE_TERMS,
    TEMPORARY_CHAT_TERMS,
    UPLOAD_TERMS,
    GeminiBot,
    GeminiPageStructureError,
    matches_ui_term,
    normalize_ui_text,
    save_gemini_diagnostics,
)
from gemini_browser_session import (
    GeminiLoginRequiredError,
    GeminiPermanentTlsError,
    GeminiPageNotReadyError,
    GeminiPageState,
    LoginStatus,
)
from main import (
    _backfill_result_project_urls,
    _choose_lovart_tool_options,
    _choose_prompt_source,
    _process_products,
    _resolve_lovart_mode,
    _resolve_browser_executable_for_run,
    parse_args,
    resolve_browser_executable,
)
from network_retry import RetryKind
from utils import update_status, write_run_summary


class FakeFormalLogger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass


class FakeFormalContext:
    def __init__(self):
        self.pages = [object()]
        self.closed = False

    def new_page(self):
        return self.pages[0]

    def close(self):
        self.closed = True


class FakeFormalPlaywrightManager:
    def __init__(self, context):
        self.playwright = type("Playwright", (), {
            "chromium": type("Chromium", (), {
                "launch_persistent_context": lambda _self, **_kwargs: context,
            })(),
        })()

    def __enter__(self):
        return self.playwright

    def __exit__(self, _exc_type, _exc, _traceback):
        return False


def run_formal_flow_for_test(*, wait_for_ready=False):
    config = {
        "browser": {"user_data_dir": "browser_profile"},
        "gemini": {"base_url": "https://gemini.google.com", "thinking_mode": True},
    }
    context = FakeFormalContext()
    with patch("main.sync_playwright", return_value=FakeFormalPlaywrightManager(context)), patch(
        "main.build_browser_launch_options", return_value={}
    ), patch("main._resolve_browser_executable_for_run", return_value=None):
        result = main._run_browser_flow(
            config,
            products=[object()],
            lovart=object(),
            logger=FakeFormalLogger(),
            run_dir=Path("runs/test"),
            resume=False,
            wait_for_ready=wait_for_ready,
            prompt_settings={},
        )
    return result, context


class MediumPriorityBehaviorTests(unittest.TestCase):
    def test_upload_completion_fails_closed_for_evaluate_errors_files_and_unrelated_images(self):
        class Page:
            def __init__(self, value):
                self.value = value

            def wait_for_timeout(self, _milliseconds):
                pass

            def evaluate(self, _script):
                if isinstance(self.value, Exception):
                    raise self.value
                return self.value

        class Logger:
            def warning(self, _message):
                pass

        cases = (
            RuntimeError("playwright sentinel"),
            {"busy": False, "fileCount": 2, "attachments": 0, "sendDisabled": False},
            {"busy": False, "fileCount": 0, "attachments": 99, "sendDisabled": False},
        )
        for value in cases:
            bot = GeminiBot(Page(value), {"gemini": {"upload_timeout": 1}}, Logger())
            with patch("gemini_bot.time.time", side_effect=[0] * 10 + [2]):
                self.assertFalse(bot._wait_for_uploads_complete(2))

    def test_upload_wait_deduplicates_nested_attachments_and_respects_spanish_busy_labels(self):
        class Page:
            def __init__(self, state):
                self.state = state

            def wait_for_timeout(self, _milliseconds):
                pass

            def evaluate(self, _script):
                return self.state

        class Logger:
            def warning(self, _message):
                pass

        nested_one = {"busy": False, "attachment_ids": ["one"]}
        spanish_busy = {"busy": False, "attachment_ids": ["one", "two"], "visible_text": "  SUBIENDO  "}
        two_unique = {"busy": False, "attachment_ids": ["one", "two"]}
        for state, expected in ((nested_one, False), (spanish_busy, False), (two_unique, True)):
            bot = GeminiBot(Page(state), {"gemini": {"upload_timeout": 1}}, Logger())
            with patch("gemini_bot.time.time", side_effect=[0] * 10 + [2]):
                self.assertEqual(bot._wait_for_uploads_complete(2), expected)

    def test_normalized_dom_fallbacks_match_real_spanish_text_and_attributes(self):
        class EmptyLocator:
            first = None

            def __init__(self):
                self.first = self

            def is_visible(self, **_kwargs):
                return False

        class Page:
            def locator(self, _selector):
                return EmptyLocator()

            def wait_for_timeout(self, _milliseconds):
                pass

            def evaluate(self, script):
                self.assert_normalized_script(script)
                if "aria-checked" in script:
                    return True
                if "chat temporal" in script.casefold():
                    return "CHAT   TEMPORAL"
                if "rapido" in script.casefold():
                    return "RÁPIDO"
                if "pensamiento ampliado" in script.casefold():
                    return "Pensamiento ampliado"
                if "ampliado" in script.casefold():
                    return "Ampliado"
                return False

            @staticmethod
            def assert_normalized_script(script):
                if "normalize('NFKD')" not in script or "data-tooltip" not in script:
                    raise AssertionError("DOM fallback must normalize all visible control attributes")

        page = Page()
        bot = GeminiBot(page, {"gemini": {}}, FakeFormalLogger())
        self.assertTrue(bot._start_temporary_chat())
        self.assertTrue(bot._open_mode_menu())
        self.assertTrue(bot._click_extended_thinking_option())
        self.assertTrue(bot._click_extended_thinking_level())
        self.assertTrue(bot._extended_thinking_option_is_checked())

    def test_real_playwright_dom_normalizes_accented_and_whitespace_controls(self):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.set_content("""
                    <button data-tooltip='  CHAT   TEMPORAL  '>temporary</button>
                    <button title='RÁPIDO'>mode</button>
                    <button aria-label='Pensamiento ampliado'>thinking</button>
                    <div role='menuitemcheckbox' title=' PENSAMIENTO   AMPLIADO ' aria-checked='true'>selected</div>
                """)
                bot = GeminiBot(page, {"gemini": {}}, FakeFormalLogger())
                self.assertTrue(bot._start_temporary_chat())
                self.assertTrue(bot._open_mode_menu())
                self.assertTrue(bot._click_extended_thinking_option())
                self.assertTrue(bot._extended_thinking_option_is_checked())
            finally:
                browser.close()

    def test_product_retry_uses_explicit_browser_allowlist(self):
        class Bot(GeminiBot):
            def __init__(self, error):
                super().__init__(object(), {"gemini": {}, "browser": {"product_attempts": 2, "retry_delays": [0]}}, FakeFormalLogger())
                self.error = error
                self.calls = 0

            def _generate_prompt_once(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise self.error
                return "ok"

        with patch("gemini_bot.classify_network_error", return_value=RetryKind.TRANSIENT):
            for marker in ("net::ERR_INVALID_URL", "net::ERR_TOO_MANY_REDIRECTS", "net::ERR_FILE_NOT_FOUND"):
                bot = Bot(RuntimeError(marker))
                with self.assertRaises(RuntimeError):
                    bot.generate_prompt("产品", "Spanish", "卖点", [])
                self.assertEqual(bot.calls, 1)
        for marker in ("net::ERR_CONNECTION_RESET", "net::ERR_NETWORK_CHANGED", "net::ERR_NAME_NOT_RESOLVED", "net::ERR_SSL_PROTOCOL_ERROR"):
            bot = Bot(RuntimeError(marker))
            self.assertEqual(bot.generate_prompt("产品", "Spanish", "卖点", []), "ok")
            self.assertEqual(bot.calls, 2)

    def test_product_retry_handles_only_transient_http_and_sanitizes_final_raw_errors(self):
        class Bot(GeminiBot):
            def __init__(self, error):
                super().__init__(object(), {"gemini": {}, "browser": {"product_attempts": 2, "retry_delays": [0]}}, FakeFormalLogger())
                self.error = error
                self.calls = 0

            def _generate_prompt_once(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise self.error
                return "ok"

        for status in (408, 429, 500, 503):
            bot = Bot(HTTPError("https://gemini.test", status, "raw-token@example.com", {}, None))
            self.assertEqual(bot.generate_prompt("产品", "Spanish", "卖点", []), "ok")
            self.assertEqual(bot.calls, 2)
        for status in (401, 403, 404):
            bot = Bot(HTTPError("https://gemini.test", status, "raw-token@example.com", {}, None))
            with self.assertRaises(GeminiPageNotReadyError) as raised:
                bot.generate_prompt("产品", "Spanish", "卖点", [])
            self.assertEqual(bot.calls, 1)
            self.assertNotIn("raw-token@example.com", "".join(traceback.format_exception(raised.exception)))

        bot = Bot(ssl.SSLCertVerificationError("raw-token@example.com"))
        with self.assertRaises(GeminiPermanentTlsError) as raised:
            bot.generate_prompt("产品", "Spanish", "卖点", [])
        self.assertEqual(bot.calls, 1)
        self.assertNotIn("raw-token@example.com", "".join(traceback.format_exception(raised.exception)))

    def test_diagnostics_reject_untrusted_language_controls_and_write_valid_safe_png(self):
        sentinels = ("secret@example.com", "token=private", "C:\\Users\\private", "/tmp/private", "https://x.test/app?key=private", "//server/share/private", "unrecognized-control-987")

        class Page:
            url = "https://gemini.google.com/app?token=private"

            def evaluate(self, _script):
                return {"language": "secret@example.com", "controls": list(sentinels) + ["Adjuntar archivos"]}

        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = save_gemini_diagnostics(Page(), tmp, "SKU", "private", 1, "private")
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["language"], "unknown")
            self.assertEqual(payload["controls"], ["upload"])
            with Image.open(metadata_path.with_suffix(".png")) as image:
                image.load()
                self.assertEqual(image.size, (1, 1))
            for artifact in metadata_path.parent.iterdir():
                data = artifact.read_bytes().decode("utf-8", errors="ignore")
                for sentinel in sentinels:
                    self.assertNotIn(sentinel, data)

    def test_final_retryable_exception_is_safe_and_traceback_has_no_raw_detail(self):
        sentinel = "net::ERR_CONNECTION_RESET token=private@example.com"

        class Bot(GeminiBot):
            def __init__(self):
                super().__init__(object(), {"gemini": {}, "browser": {"product_attempts": 2, "retry_delays": [0]}}, FakeFormalLogger())
                self.calls = 0

            def _generate_prompt_once(self, *_args, **_kwargs):
                self.calls += 1
                raise RuntimeError(sentinel)

        bot = Bot()
        with self.assertRaises(GeminiPageNotReadyError) as raised:
            bot.generate_prompt("产品", "Spanish", "卖点", [])
        self.assertEqual(bot.calls, 2)
        trace = "".join(traceback.format_exception(raised.exception))
        self.assertNotIn(sentinel, str(raised.exception))
        self.assertNotIn(sentinel, trace)

    def test_temporary_chat_failure_stops_each_attempt_before_preamble(self):
        class Bot(GeminiBot):
            def __init__(self):
                page = type("Page", (), {"goto": lambda _self, *_args, **_kwargs: None, "wait_for_timeout": lambda _self, _ms: None})()
                super().__init__(page, {"gemini": {}, "browser": {"product_attempts": 2}}, FakeFormalLogger())
                self.chats = 0

            def _start_temporary_chat(self):
                self.chats += 1
                return False

            def _send_message(self, _text):
                raise AssertionError("must not continue in the current chat")

        bot = Bot()
        with self.assertRaises(GeminiPageStructureError):
            bot.generate_prompt("产品", "Spanish", "卖点", [])
        self.assertEqual(bot.chats, 2)

    def test_spanish_structural_fallbacks_drive_temporary_mode_thinking_and_upload(self):
        class Locator:
            def __init__(self, page, selector):
                self.page = page
                self.selector = selector
                self.first = self
                self.last = self

            def count(self):
                if self.selector == 'input[type="file"]':
                    self.page.file_queries += 1
                    return 1 if self.page.file_queries > 1 else 0
                return 0

            def is_visible(self, timeout=None):
                return any(term in self.selector for term in ("Chat temporal", "Rápido", "Pensamiento ampliado", "Adjuntar"))

            def click(self, **_kwargs):
                self.page.clicked.append(self.selector)

            def set_input_files(self, _paths):
                self.page.uploaded = True

        class Page:
            def __init__(self):
                self.clicked = []
                self.file_queries = 0
                self.uploaded = False

            def locator(self, selector):
                return Locator(self, selector)

            def wait_for_timeout(self, _milliseconds):
                pass

        class Bot(GeminiBot):
            def _wait_for_uploads_complete(self, _expected_count):
                return True

        page = Page()
        bot = Bot(page, {"gemini": {}}, FakeFormalLogger())
        self.assertTrue(bot._start_temporary_chat())
        self.assertTrue(bot._open_mode_menu())
        self.assertTrue(bot._click_extended_thinking_option())
        self.assertTrue(bot._upload_images_once(["image.jpg"]))
        self.assertTrue(page.uploaded)
        self.assertTrue(any("Chat temporal" in item for item in page.clicked))
        self.assertTrue(any("Rápido" in item for item in page.clicked))
        self.assertTrue(any("Pensamiento ampliado" in item for item in page.clicked))
        self.assertTrue(any("Adjuntar" in item for item in page.clicked))

    def test_thinking_ready_retries_once_and_error_is_not_page_structure(self):
        class Bot(GeminiBot):
            def __init__(self):
                super().__init__(object(), {"gemini": {}}, FakeFormalLogger())
                self.selects = 0

            def _select_thinking_mode(self):
                self.selects += 1
                return self.selects == 2

        ready = LoginStatus.create(GeminiPageState.READY, True, "https://gemini.google.com/app", "es", "ready")
        bot = Bot()
        with patch("gemini_bot.inspect_gemini_page", return_value=ready):
            bot._select_thinking_mode_with_recovery("SKU")
        self.assertEqual(bot.selects, 2)

        error = LoginStatus.create(GeminiPageState.ERROR, False, "https://gemini.google.com/app", "es", "error")
        bot = Bot()
        with patch("gemini_bot.inspect_gemini_page", return_value=error):
            with self.assertRaises(GeminiPageNotReadyError):
                bot._select_thinking_mode_with_recovery("SKU")
        self.assertEqual(bot.selects, 1)

    def test_permanent_browser_errors_do_not_retry_but_reset_retries_once(self):
        for error in (
            RuntimeError("net::ERR_CERT_REVOKED"),
            RuntimeError("net::ERR_ACCESS_DENIED"),
            RuntimeError("blocked by policy"),
        ):
            class Bot(GeminiBot):
                def __init__(self):
                    super().__init__(object(), {"gemini": {}, "browser": {"product_attempts": 2}}, FakeFormalLogger())
                    self.calls = 0

                def _generate_prompt_once(self, *_args, **_kwargs):
                    self.calls += 1
                    raise error

            bot = Bot()
            with self.assertRaises(RuntimeError):
                bot.generate_prompt("产品", "Spanish", "卖点", [])
            self.assertEqual(bot.calls, 1)

        class ResetBot(GeminiBot):
            def __init__(self):
                super().__init__(object(), {"gemini": {}, "browser": {"product_attempts": 2, "retry_delays": [0]}}, FakeFormalLogger())
                self.calls = 0

            def _generate_prompt_once(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("net::ERR_CONNECTION_RESET")
                return "ok"

        self.assertEqual(ResetBot().generate_prompt("产品", "Spanish", "卖点", []), "ok")

    def test_diagnostics_redact_every_artifact_use_unique_names_and_logs_hide_raw_errors(self):
        sentinel = "secret@example.com token=private /tmp/private"

        class Page:
            url = "https://secret@example.com@gemini.google.com/app/token/private?token=private"

            def evaluate(self, _script):
                return {"language": "es", "controls": [sentinel]}

            def screenshot(self, path, **_kwargs):
                Path(path).write_text(sentinel, encoding="utf-8")

            def content(self):
                return f"<html>{sentinel}</html>"

        class Logger:
            def __init__(self):
                self.messages = []

            def info(self, _message):
                pass

            def warning(self, message):
                self.messages.append(message)

        with tempfile.TemporaryDirectory() as tmp:
            first = save_gemini_diagnostics(Page(), tmp, "SKU", sentinel, 1, sentinel)
            second = save_gemini_diagnostics(Page(), tmp, "SKU", sentinel, 1, sentinel)
            self.assertNotEqual(first, second)
            for artifact in first.parent.iterdir():
                self.assertNotIn("secret@example.com", artifact.read_bytes().decode("utf-8", errors="ignore"))
                self.assertNotIn("private", artifact.read_bytes().decode("utf-8", errors="ignore"))

        logger = Logger()
        page = type("Page", (), {"evaluate": lambda _self, _script: (_ for _ in ()).throw(RuntimeError(sentinel))})()
        GeminiBot(page, {"gemini": {}}, logger)._start_temporary_chat()
        self.assertNotIn(sentinel, " ".join(logger.messages))

    def test_spanish_text_normalization_removes_accents_and_case(self):
        self.assertEqual(normalize_ui_text("  PENSAMIENTO RÁPIDO  "), "pensamiento rapido")

    def test_spanish_mode_and_upload_terms_are_recognized(self):
        self.assertTrue(matches_ui_term("Rápido", MODE_TERMS))
        self.assertTrue(matches_ui_term("Pensamiento ampliado", EXTENDED_THINKING_TERMS))
        self.assertTrue(matches_ui_term("Adjuntar archivos", UPLOAD_TERMS))
        self.assertTrue(matches_ui_term("Chat temporal", TEMPORARY_CHAT_TERMS))

    def test_product_prompt_is_not_sent_when_upload_never_completes(self):
        class NullPage:
            def goto(self, _url, **_kwargs):
                pass

            def wait_for_timeout(self, _milliseconds):
                pass

        class Logger:
            def info(self, _message):
                pass

            def warning(self, _message):
                pass

        class OrderedGeminiBot(GeminiBot):
            def __init__(self):
                super().__init__(NullPage(), {"gemini": {"thinking_mode": True}, "browser": {"product_attempts": 2}}, Logger())
                self.events = []

            def _start_temporary_chat(self):
                self.events.append("temporary_chat")
                return True

            def _select_thinking_mode(self):
                self.events.append("thinking")
                return True

            def _response_count(self):
                return 0

            def _send_message(self, text):
                self.events.append("product_prompt" if "产品信息/卖点" in text else "preamble")

            def _wait_for_reply(self, **_kwargs):
                pass

            def _upload_images(self, _paths):
                self.events.append("upload")
                return False

        bot = OrderedGeminiBot()
        with self.assertRaisesRegex(RuntimeError, "image upload did not complete"):
            bot.generate_prompt("产品", "Spanish", "卖点", ["image.jpg"])
        self.assertNotIn("product_prompt", bot.events)

    def test_transient_first_product_attempt_restarts_temporary_chat_then_succeeds(self):
        class NullPage:
            pass

        class Logger:
            def info(self, _message):
                pass

            def warning(self, _message):
                pass

        class RetryGeminiBot(GeminiBot):
            def __init__(self):
                super().__init__(NullPage(), {"gemini": {}, "browser": {"product_attempts": 2, "retry_delays": [0]}}, Logger())
                self.attempts = 0
                self.events = []

            def _generate_prompt_once(self, *_args, **_kwargs):
                self.attempts += 1
                self.events.append("temporary_chat")
                if self.attempts == 1:
                    raise RuntimeError("net::ERR_CONNECTION_RESET")
                return "generated prompt"

        bot = RetryGeminiBot()
        with patch("network_retry.time.sleep"):
            self.assertEqual(bot.generate_prompt("产品", "Spanish", "卖点", ["image.jpg"]), "generated prompt")
        self.assertEqual(bot.attempts, 2)
        self.assertEqual(bot.events.count("temporary_chat"), 2)

    def test_product_retry_stops_after_two_attempts(self):
        class RetryGeminiBot(GeminiBot):
            def __init__(self):
                super().__init__(object(), {"gemini": {}, "browser": {"product_attempts": 9, "retry_delays": [0]}}, FakeFormalLogger())
                self.attempts = 0

            def _generate_prompt_once(self, *_args, **_kwargs):
                self.attempts += 1
                raise GeminiPageStructureError("missing control")

        bot = RetryGeminiBot()
        with self.assertRaises(GeminiPageStructureError), patch("network_retry.time.sleep"):
            bot.generate_prompt("产品", "Spanish", "卖点", [])
        self.assertEqual(bot.attempts, 2)

    def test_login_tls_auth_and_verified_upload_errors_are_not_retried(self):
        for error, expected_type in (
            (GeminiLoginRequiredError(), GeminiLoginRequiredError),
            (ssl.SSLCertVerificationError("certificate verify failed"), GeminiPermanentTlsError),
            (HTTPError("https://gemini.google.com/app", 403, "denied", {}, None), GeminiPageNotReadyError),
            (RuntimeError("Gemini image upload did not complete"), GeminiPageNotReadyError),
        ):
            class RetryGeminiBot(GeminiBot):
                def __init__(self):
                    super().__init__(object(), {"gemini": {}, "browser": {"product_attempts": 2}}, FakeFormalLogger())
                    self.attempts = 0

                def _generate_prompt_once(self, *_args, **_kwargs):
                    self.attempts += 1
                    raise error

            bot = RetryGeminiBot()
            with self.assertRaises(expected_type):
                bot.generate_prompt("产品", "Spanish", "卖点", [])
            self.assertEqual(bot.attempts, 1)

    def test_thinking_recovery_raises_login_error_without_retrying_controls(self):
        class RecoveryBot(GeminiBot):
            def __init__(self):
                super().__init__(object(), {"gemini": {}}, FakeFormalLogger())
                self.selects = 0

            def _select_thinking_mode(self):
                self.selects += 1
                return False

        bot = RecoveryBot()
        waiting = LoginStatus.create(GeminiPageState.WAITING_LOGIN, False, "https://accounts.google.com/signin?email=secret@example.com", "es", "login")
        with patch("gemini_bot.inspect_gemini_page", return_value=waiting):
            with self.assertRaises(GeminiLoginRequiredError):
                bot._select_thinking_mode_with_recovery("SKU-1")
        self.assertEqual(bot.selects, 1)

    def test_diagnostics_redacts_query_email_and_raw_error(self):
        class DiagnosticPage:
            url = "https://gemini.google.com/app?email=secret@example.com&token=private"

            def evaluate(self, _script):
                return {"language": "es", "controls": ["Adjuntar archivos", "secret@example.com", "x" * 300]}

            def screenshot(self, **_kwargs):
                pass

            def content(self):
                return "<html>private</html>"

        with tempfile.TemporaryDirectory() as tmp:
            result = save_gemini_diagnostics(
                DiagnosticPage(), Path(tmp), "SKU/1", "failed: secret@example.com", 2, "raw token=private",
            )
            payload = json.loads(result.read_text(encoding="utf-8"))
        self.assertEqual(payload["url"], "https://gemini.google.com/app")
        self.assertNotIn("secret@example.com", json.dumps(payload))
        self.assertNotIn("private", json.dumps(payload))
        self.assertLessEqual(len(payload["controls"]), 20)
        self.assertLessEqual(max(map(len, payload["controls"])), 160)

    def test_ui_mode_uses_defaults_without_console_input(self):
        config = {"gemini_api": {}, "nvidia_api": {}, "lovart": {}}
        args = parse_args(["--prompt-source", "ask", "--lovart", "ask"])

        with patch.dict(os.environ, {"UI_MODE": "1"}), patch(
            "builtins.input", side_effect=AssertionError("UI mode must not prompt")
        ):
            source = _choose_prompt_source(config, args)
            _choose_lovart_tool_options(config, args)
            fast_mode = _resolve_lovart_mode(args.lovart)

        self.assertEqual(source, "gemini_browser")
        self.assertEqual(config["lovart"]["image_model"], "auto")
        self.assertFalse(fast_mode)

    @patch("main._process_products")
    @patch("main.navigate_gemini_with_retry")
    def test_formal_browser_flow_blocks_products_when_login_is_required(self, navigate, process):
        navigate.return_value = LoginStatus.create(
            GeminiPageState.WAITING_LOGIN,
            False,
            "https://accounts.google.com/signin",
            "zh-CN",
            "waiting",
            pid=1,
        )

        with self.assertRaises(main.GeminiLoginRequiredError) as raised:
            run_formal_flow_for_test()

        self.assertIn("未登录", str(raised.exception))
        process.assert_not_called()

    @patch("main._process_products")
    @patch("main.navigate_gemini_with_retry")
    def test_formal_browser_flow_blocks_products_when_page_is_not_ready(self, navigate, process):
        navigate.return_value = LoginStatus.create(
            GeminiPageState.PAGE_LOADING,
            False,
            "https://gemini.google.com/app",
            "zh-CN",
            "loading",
            pid=1,
        )

        with self.assertRaises(main.GeminiPageNotReadyError) as raised:
            run_formal_flow_for_test()

        self.assertIn("未准备", str(raised.exception))
        process.assert_not_called()

    @patch("main._process_products")
    @patch("main.navigate_gemini_with_retry")
    def test_formal_browser_flow_requires_ready_state_as_well_as_ready_flag(self, navigate, process):
        navigate.return_value = LoginStatus.create(
            GeminiPageState.ERROR,
            True,
            "https://gemini.google.com/app",
            "zh-CN",
            "incorrectly marked ready",
            pid=1,
        )

        with self.assertRaises(main.GeminiPageNotReadyError):
            run_formal_flow_for_test()

        process.assert_not_called()

    @patch("main._process_products")
    @patch("main.navigate_gemini_with_retry", side_effect=TimeoutError("private timeout detail"))
    def test_formal_browser_flow_maps_timeout_to_safe_page_not_ready_error(self, _navigate, process):
        with self.assertRaises(main.GeminiPageNotReadyError) as raised:
            run_formal_flow_for_test()

        self.assertIn("未准备", str(raised.exception))
        self.assertNotIn("private timeout detail", str(raised.exception))
        self.assertNotIn(
            "private timeout detail", "".join(traceback.format_exception(raised.exception))
        )
        process.assert_not_called()

    @patch("main._process_products")
    @patch(
        "main.navigate_gemini_with_retry",
        side_effect=ssl.SSLCertVerificationError("private certificate detail"),
    )
    def test_formal_browser_flow_maps_permanent_tls_to_safe_error(self, _navigate, process):
        with self.assertRaises(main.GeminiPermanentTlsError) as raised:
            run_formal_flow_for_test()

        self.assertIn("证书", str(raised.exception))
        self.assertNotIn("private certificate detail", str(raised.exception))
        self.assertNotIn(
            "private certificate detail", "".join(traceback.format_exception(raised.exception))
        )
        process.assert_not_called()

    @patch("main._process_products", return_value=(1, 0, 0, 0))
    @patch("main.navigate_gemini_with_retry")
    def test_formal_browser_flow_processes_only_after_ready(self, navigate, process):
        navigate.return_value = LoginStatus.create(
            GeminiPageState.READY,
            True,
            "https://gemini.google.com/app",
            "es",
            "ready",
            pid=1,
        )

        result, _context = run_formal_flow_for_test()

        self.assertEqual(result, (1, 0, 0, 0))
        process.assert_called_once()

    @patch("main._process_products", return_value=(1, 0, 0, 0))
    @patch("main.navigate_gemini_with_retry")
    def test_formal_browser_flow_never_reads_console_input_in_ui_mode(self, navigate, _process):
        navigate.return_value = LoginStatus.create(
            GeminiPageState.READY,
            True,
            "https://gemini.google.com/app",
            "en",
            "ready",
            pid=1,
        )

        with patch.dict(os.environ, {"UI_MODE": "1"}), patch(
            "builtins.input", side_effect=AssertionError("UI mode must not prompt")
        ):
            result, _context = run_formal_flow_for_test(wait_for_ready=True)

        self.assertEqual(result, (1, 0, 0, 0))

    def test_backfill_result_project_urls_reads_gbk_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "output" / "results.csv"
            product_dir = root / "output" / "SKU-GBK"
            results.parent.mkdir(parents=True, exist_ok=True)
            product_dir.mkdir(parents=True, exist_ok=True)
            results.write_bytes(
                "product_id,product_name,status,project_url,error\nSKU-GBK,测试商品,success,,\n".encode("gbk")
            )
            update_status(product_dir, "lovart_project_created", project_id="project-123")

            cwd = Path.cwd()
            os.chdir(root)
            try:
                changed = _backfill_result_project_urls(results)
            finally:
                os.chdir(cwd)

            self.assertEqual(changed, 1)
            self.assertIn("projectId=project-123", results.read_text(encoding="utf-8"))

    def test_process_products_records_resumed_completed_product_in_results_csv(self):
        class Product:
            id = "SKU-COMPLETED"
            name_cn = "\u5df2\u5b8c\u6210\u5546\u54c1"
            language = "Portuguese"
            selling_points = "points"
            image_paths = ["output/SKU-COMPLETED/image_1.png"]

        class Gemini:
            def generate_prompt(self, **kwargs):
                raise AssertionError("completed product should be skipped")

        class Lovart:
            pass

        class Logger:
            def info(self, message):
                pass

            def warning(self, message):
                pass

            def error(self, message):
                raise AssertionError(message)

        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                product_dir = Path("output") / "SKU-COMPLETED"
                product_dir.mkdir(parents=True, exist_ok=True)
                update_status(
                    product_dir,
                    "lovart_done",
                    project_url="https://www.lovart.ai/canvas?projectId=project-completed",
                )

                result = _process_products([Product()], Gemini(), Lovart(), Logger(), Path("runs") / "run")

                with (Path("output") / "results.csv").open("r", encoding="utf-8", newline="") as fh:
                    rows = list(csv.DictReader(fh))
            finally:
                os.chdir(cwd)

        self.assertEqual(result, (0, 0, 1, 0))
        self.assertEqual(rows[0]["product_id"], "SKU-COMPLETED")
        self.assertEqual(rows[0]["status"], "success")
        self.assertEqual(rows[0]["project_url"], "https://www.lovart.ai/canvas?projectId=project-completed")

    def test_resolve_browser_executable_uses_configured_existing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            chrome = Path(tmp) / "chrome.exe"
            chrome.write_text("", encoding="utf-8")

            result = resolve_browser_executable({"chrome_exe": str(chrome)})

        self.assertEqual(result, str(chrome))

    def test_resolve_browser_executable_falls_back_to_bundled_chromium(self):
        result = resolve_browser_executable(
            {"chrome_exe": "C:\\missing\\chrome.exe"},
            candidate_paths=[],
        )

        self.assertIsNone(result)

    def test_resolve_browser_executable_for_run_accepts_manual_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            chrome = Path(tmp) / "chrome.exe"
            chrome.write_text("", encoding="utf-8")
            answers = iter([str(chrome)])

            result = _resolve_browser_executable_for_run(
                {"chrome_exe": "C:\\missing\\chrome.exe"},
                interactive=True,
                candidate_paths=[],
                input_func=lambda prompt="": next(answers),
            )

        self.assertEqual(result, str(chrome))

    def test_resolve_browser_executable_for_run_uses_bundled_chromium_when_manual_path_blank(self):
        result = _resolve_browser_executable_for_run(
            {"chrome_exe": "C:\\missing\\chrome.exe"},
            interactive=True,
            candidate_paths=[],
            input_func=lambda prompt="": "",
        )

        self.assertIsNone(result)

    def test_resolve_image_scan_config_supports_end_column(self):
        cfg = {"image_columns": {"start": "E", "end": "H", "empty_streak": 3}}
        result = resolve_image_scan_config(cfg)
        self.assertEqual(result["start_col"], 5)
        self.assertEqual(result["end_col"], 8)
        self.assertEqual(result["empty_streak"], 3)

    def test_resolve_image_scan_config_supports_max_columns(self):
        cfg = {"image_columns": {"start": "E", "max_columns": 4}}
        result = resolve_image_scan_config(cfg)
        self.assertEqual(result["start_col"], 5)
        self.assertEqual(result["end_col"], 8)
        self.assertEqual(result["empty_streak"], 2)

    def test_write_run_summary_outputs_json_and_csv(self):
        rows = [{
            "product_id": "SKU-1",
            "product_name": "Name, one",
            "status": "success",
            "project_url": "https://example.test",
            "gemini_chars": 123,
            "artifact_count": 2,
            "duration_seconds": 9,
            "error": "",
        }]
        with tempfile.TemporaryDirectory() as tmp:
            write_run_summary(tmp, rows)
            data = json.loads((Path(tmp) / "summary.json").read_text(encoding="utf-8"))
            with (Path(tmp) / "summary.csv").open("r", encoding="utf-8", newline="") as fh:
                csv_rows = list(csv.DictReader(fh))

        self.assertEqual(data[0]["product_id"], "SKU-1")
        self.assertEqual(csv_rows[0]["product_name"], "Name, one")

    def test_gemini_debug_snapshot_writes_html_and_screenshot(self):
        class FakePage:
            def screenshot(self, path, full_page):
                Path(path).write_bytes(b"png")

            def content(self):
                return "<html>debug</html>"

        class FakeLogger:
            def warning(self, message):
                raise AssertionError(message)

        with tempfile.TemporaryDirectory() as tmp:
            bot = GeminiBot(FakePage(), {"gemini": {}}, FakeLogger(), run_dir=tmp)
            bot._save_debug_snapshot("SKU/1", "upload failed")
            files = list((Path(tmp) / "browser-debug" / "SKU_1").iterdir())

        self.assertTrue(any(path.suffix == ".png" for path in files))
        self.assertTrue(any(path.suffix == ".html" for path in files))

    def test_gemini_browser_flow_uses_temporary_chat_and_waits_in_order(self):
        class FakePage:
            def goto(self, url, wait_until):
                events.append("goto")

            def wait_for_timeout(self, ms):
                events.append(f"page_wait:{ms}")

        class FakeLogger:
            def info(self, message):
                pass

            def warning(self, message):
                pass

        class OrderedGeminiBot(GeminiBot):
            def _start_temporary_chat(self):
                events.append("temporary_chat")
                return True

            def _select_thinking_mode(self):
                events.append("thinking_mode")
                return True

            def _response_count(self):
                events.append("response_count")
                return len([event for event in events if event.startswith("wait_reply")])

            def _send_message(self, text):
                label = "preamble" if "preamble" in text else "product_prompt"
                events.append(f"send:{label}")

            def _wait_for_reply(self, previous_response_count=None, require_design_keywords=True):
                events.append(f"wait_reply:{require_design_keywords}:{previous_response_count}")

            def _upload_images(self, image_paths):
                events.append("upload_images")
                return True

            def _get_last_response(self):
                return "主标题\n" + ("设计方案\n" * 120)

        events = []
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            Path("preamble.txt").write_text("preamble", encoding="utf-8")
            try:
                bot = OrderedGeminiBot(FakePage(), {"gemini": {}}, FakeLogger())
                bot.generate_prompt("产品", "葡萄牙语", "卖点", ["image.jpeg"], product_id="SKU-1")
            finally:
                os.chdir(cwd)

        self.assertEqual(
            events[:9],
            [
                "goto",
                "page_wait:4000",
                "temporary_chat",
                "thinking_mode",
                "response_count",
                "send:preamble",
                "wait_reply:False:0",
                "upload_images",
                "response_count",
            ],
        )
        self.assertEqual(events[9], "send:product_prompt")
        self.assertEqual(events[10], "wait_reply:True:1")

    def test_gemini_upload_waits_for_completion_after_setting_files(self):
        class FakeLocator:
            def __init__(self, selector):
                self.selector = selector
                self.first = self
                self.last = self

            def count(self):
                if "add_2" in self.selector:
                    return 1
                if 'input[type="file"]' in self.selector:
                    return 1
                return 0

            def click(self, timeout=None, force=False):
                events.append(f"click:{self.selector}")

            def set_input_files(self, image_paths):
                events.append(f"set_files:{len(image_paths)}")

        class FakePage:
            def locator(self, selector):
                return FakeLocator(selector)

            def wait_for_timeout(self, ms):
                events.append(f"wait:{ms}")

        class FakeLogger:
            def info(self, message):
                pass

            def warning(self, message):
                raise AssertionError(message)

        class UploadGeminiBot(GeminiBot):
            def _wait_for_uploads_complete(self, expected_count):
                events.append(f"wait_upload_complete:{expected_count}")
                return True

        events = []
        bot = UploadGeminiBot(FakePage(), {"gemini": {}}, FakeLogger())
        self.assertTrue(bot._upload_images(["a.jpeg", "b.jpeg"]))
        self.assertIn("set_files:2", events)
        self.assertIn("wait_upload_complete:2", events)

    def test_gemini_upload_retries_when_first_menu_attempt_finds_no_file_input(self):
        class FakeLocator:
            def __init__(self, page, selector):
                self.page = page
                self.selector = selector
                self.first = self
                self.last = self

            def count(self):
                if "add_2" in self.selector:
                    return 1
                if 'input[type="file"]' in self.selector:
                    return 1 if self.page.add_clicks >= 2 else 0
                return 0

            def click(self, timeout=None, force=False):
                if "add_2" in self.selector:
                    self.page.add_clicks += 1
                events.append(f"click:{self.selector}")

            def is_visible(self, timeout=None):
                return "add_2" in self.selector

            def set_input_files(self, image_paths):
                events.append(f"set_files:{len(image_paths)}")

        class FakePage:
            def __init__(self):
                self.add_clicks = 0

            def locator(self, selector):
                return FakeLocator(self, selector)

            def wait_for_timeout(self, ms):
                events.append(f"wait:{ms}")

        class FakeLogger:
            def info(self, message):
                pass

            def warning(self, message):
                events.append(f"warning:{message}")

        class UploadGeminiBot(GeminiBot):
            def _wait_for_uploads_complete(self, expected_count):
                events.append(f"wait_upload_complete:{expected_count}")
                return True

        events = []
        bot = UploadGeminiBot(FakePage(), {"gemini": {"upload_attempts": 2}}, FakeLogger())
        self.assertTrue(bot._upload_images(["a.jpeg", "b.jpeg"]))
        self.assertEqual(len([event for event in events if event.startswith("click:button:has")]), 2)
        self.assertIn("set_files:2", events)

    def test_gemini_thinking_mode_accepts_already_selected_flash_extended(self):
        class FakePage:
            def evaluate(self, script):
                return True

            def wait_for_timeout(self, ms):
                pass

        class FakeLogger:
            def __init__(self):
                self.messages = []

            def info(self, message):
                self.messages.append(message)

            def warning(self, message):
                self.messages.append(message)

        class AlreadySelectedBot(GeminiBot):
            def _open_mode_menu(self):
                return False

        logger = FakeLogger()
        bot = AlreadySelectedBot(FakePage(), {"gemini": {}}, logger)

        self.assertTrue(bot._select_thinking_mode())
        self.assertTrue(any("already selected" in message for message in logger.messages))

    def test_gemini_thinking_mode_accepts_checked_extended_thinking_option(self):
        events = []

        class FakePage:
            def wait_for_timeout(self, ms):
                events.append(f"wait:{ms}")

        class FakeLogger:
            def __init__(self):
                self.messages = []

            def info(self, message):
                self.messages.append(message)

            def warning(self, message):
                self.messages.append(message)

        class NewMenuBot(GeminiBot):
            def _open_mode_menu(self):
                events.append("open_menu")
                return True

            def _click_flash_model(self):
                events.append("click_flash")
                return True

            def _extended_thinking_option_is_checked(self):
                events.append("checked")
                return True

            def _click_extended_thinking_option(self):
                events.append("click_extended")
                return False

        logger = FakeLogger()
        bot = NewMenuBot(FakePage(), {"gemini": {}}, logger)

        self.assertTrue(bot._select_thinking_mode())
        self.assertIn("checked", events)
        self.assertNotIn("click_extended", events)
        self.assertTrue(any("extended thinking option already selected" in message for message in logger.messages))

    def test_gemini_thinking_mode_clicks_unchecked_extended_thinking_option(self):
        events = []

        class FakePage:
            def wait_for_timeout(self, ms):
                events.append(f"wait:{ms}")

        class FakeLogger:
            def __init__(self):
                self.messages = []

            def info(self, message):
                self.messages.append(message)

            def warning(self, message):
                self.messages.append(message)

        class NewMenuBot(GeminiBot):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.check_count = 0

            def _open_mode_menu(self):
                events.append("open_menu")
                return True

            def _click_flash_model(self):
                events.append("click_flash")
                return True

            def _extended_thinking_option_is_checked(self):
                self.check_count += 1
                events.append("checked")
                return False

            def _click_extended_thinking_option(self):
                events.append("click_extended")
                return True

        logger = FakeLogger()
        bot = NewMenuBot(FakePage(), {"gemini": {}}, logger)

        self.assertTrue(bot._select_thinking_mode())
        self.assertIn("click_extended", events)
        self.assertTrue(any("selected Flash with extended thinking" in message for message in logger.messages))

    def test_gemini_reply_wait_uses_generation_state_not_text_keywords(self):
        class FakePage:
            def wait_for_timeout(self, ms):
                pass

        class FakeLogger:
            def __init__(self):
                self.completed = False

            def info(self, message):
                if "reply complete" in message:
                    self.completed = True

            def warning(self, message):
                pass

        class StateGeminiBot(GeminiBot):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.states = [
                    {"generating": True, "response_count": 0},
                    {"generating": False, "response_count": 0},
                    {"generating": False, "response_count": 0},
                    {"generating": False, "response_count": 0},
                ]

            def _read_generation_state(self):
                return self.states.pop(0) if self.states else {"generating": False, "response_count": 0}

        logger = FakeLogger()
        bot = StateGeminiBot(FakePage(), {"gemini": {"reply_timeout": 0.01}}, logger)
        bot._wait_for_reply(previous_response_count=0)

        self.assertTrue(logger.completed)

    def test_gemini_reply_wait_does_not_complete_from_text_alone(self):
        class FakePage:
            def wait_for_timeout(self, ms):
                pass

        class FakeLogger:
            def __init__(self):
                self.completed = False
                self.timed_out = False

            def info(self, message):
                if "reply complete" in message:
                    self.completed = True

            def warning(self, message):
                if "timed out" in message:
                    self.timed_out = True

        class IdleGeminiBot(GeminiBot):
            def _read_generation_state(self):
                return {"generating": False, "response_count": 0}

        logger = FakeLogger()
        bot = IdleGeminiBot(FakePage(), {"gemini": {"reply_timeout": 0.01}}, logger)
        bot._wait_for_reply(previous_response_count=0)

        self.assertFalse(logger.completed)
        self.assertTrue(logger.timed_out)

    def test_process_products_resumes_failed_product_with_existing_lovart_support_images(self):
        class Product:
            id = "SKU-RESUME"
            name_cn = "Product"
            language = "Portuguese"
            selling_points = "points"
            image_paths = ["output/SKU-RESUME/image_1.png"]

        class Gemini:
            def generate_prompt(self, **kwargs):
                events.append(("gemini_images", kwargs["image_paths"]))
                return "generated prompt"

        class Lovart:
            def create_project(self, product_id, product_name_cn=""):
                raise AssertionError("should reuse existing project")

            def create_support_image(self, **kwargs):
                raise AssertionError("should reuse support image")

            def create_and_generate(self, **kwargs):
                events.append(("detail_project", kwargs["project_id"]))
                events.append(("detail_images", kwargs["image_paths"]))
                return {"generation_succeeded": True, "project_id": kwargs["project_id"]}

        class Logger:
            def info(self, message):
                pass

            def warning(self, message):
                pass

            def error(self, message):
                raise AssertionError(message)

        events = []
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                product_dir = Path("output") / "SKU-RESUME"
                white = product_dir / "lovart_steps" / "white_bg" / "white.png"
                scene = product_dir / "lovart_steps" / "scene" / "scene.png"
                white.parent.mkdir(parents=True, exist_ok=True)
                scene.parent.mkdir(parents=True, exist_ok=True)
                white.write_bytes(b"white")
                scene.write_bytes(b"scene")
                product_image = product_dir / "image_1.png"
                product_image.write_bytes(b"product")
                update_status(
                    product_dir,
                    "failed",
                    project_id="project-123",
                    project_url="https://www.lovart.ai/canvas?projectId=project-123",
                    lovart_final_images=[str(white), str(scene)],
                    reason="Gemini failed",
                )

                result = _process_products([Product()], Gemini(), Lovart(), Logger(), Path("runs") / "run")
            finally:
                os.chdir(cwd)

        self.assertEqual(result, (1, 0, 0, 0))
        self.assertIn(("detail_project", "project-123"), events)
        self.assertIn(("gemini_images", [str(white), str(scene)]), events)
        self.assertIn(("detail_images", [str(white), str(scene)]), events)

    def test_process_products_restarts_when_existing_lovart_project_is_invalid(self):
        class Product:
            id = "SKU-INVALID"
            name_cn = "Product"
            language = "Portuguese"
            selling_points = "points"
            image_paths = ["output/SKU-INVALID/image_1.png"]

        class Gemini:
            def generate_prompt(self, **kwargs):
                events.append(("gemini_images", kwargs["image_paths"]))
                return "generated prompt"

        class Lovart:
            def validate_project(self, project_id):
                events.append(("validate_project", project_id))
                return False

            def create_project(self, product_id, product_name_cn=""):
                events.append(("create_project", product_id, product_name_cn))
                return "new-project"

            def create_support_image(self, **kwargs):
                events.append(("support", kwargs["step_name"], kwargs["project_id"], kwargs["image_paths"]))
                local_path = Path("output") / "SKU-INVALID" / f"{kwargs['step_name']}.png"
                local_path.write_bytes(kwargs["step_name"].encode("utf-8"))
                return {"local_path": str(local_path), "project_id": kwargs["project_id"]}

            def create_and_generate(self, **kwargs):
                events.append(("detail_project", kwargs["project_id"]))
                events.append(("detail_images", kwargs["image_paths"]))
                return {"generation_succeeded": True, "project_id": kwargs["project_id"]}

        class Logger:
            def info(self, message):
                pass

            def warning(self, message):
                pass

            def error(self, message):
                raise AssertionError(message)

        events = []
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                product_dir = Path("output") / "SKU-INVALID"
                old_white = product_dir / "lovart_steps" / "white_bg" / "old-white.png"
                old_scene = product_dir / "lovart_steps" / "scene" / "old-scene.png"
                old_white.parent.mkdir(parents=True, exist_ok=True)
                old_scene.parent.mkdir(parents=True, exist_ok=True)
                old_white.write_bytes(b"old-white")
                old_scene.write_bytes(b"old-scene")
                product_image = product_dir / "image_1.png"
                product_image.write_bytes(b"product")
                update_status(
                    product_dir,
                    "failed",
                    project_id="old-project",
                    project_url="https://www.lovart.ai/canvas?projectId=old-project",
                    lovart_final_images=[str(old_white), str(old_scene)],
                    reason="Gemini failed",
                )

                result = _process_products([Product()], Gemini(), Lovart(), Logger(), Path("runs") / "run")
            finally:
                os.chdir(cwd)

        self.assertEqual(result, (1, 0, 0, 0))
        self.assertIn(("validate_project", "old-project"), events)
        self.assertIn(("create_project", "SKU-INVALID", "Product"), events)
        self.assertIn(("detail_project", "new-project"), events)
        self.assertNotIn(("detail_project", "old-project"), events)
        self.assertTrue(any(event[0] == "support" and event[2] == "new-project" for event in events))
        self.assertFalse(any(str(old_white) in str(event) or str(old_scene) in str(event) for event in events))

    def test_process_products_marks_support_timeout_as_still_running(self):
        class Product:
            id = "SKU-TIMEOUT"
            name_cn = "Product"
            language = "Portuguese"
            selling_points = "points"
            image_paths = ["output/SKU-TIMEOUT/image_1.png"]

        class Gemini:
            def generate_prompt(self, **kwargs):
                raise AssertionError("should stop before Gemini when support image is still running")

        class Lovart:
            def create_project(self, product_id, product_name_cn=""):
                return "project-timeout"

            def create_support_image(self, **kwargs):
                return {
                    "final_status": "timeout",
                    "project_id": kwargs["project_id"],
                    "generation_succeeded": False,
                }

        class Logger:
            def info(self, message):
                pass

            def warning(self, message):
                pass

            def error(self, message):
                raise AssertionError(message)

        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                product_dir = Path("output") / "SKU-TIMEOUT"
                product_dir.mkdir(parents=True)
                (product_dir / "image_1.png").write_bytes(b"product")

                result = _process_products([Product()], Gemini(), Lovart(), Logger(), Path("runs") / "run")
                with (Path("runs") / "run" / "summary.csv").open(encoding="utf-8", newline="") as fh:
                    rows = list(csv.DictReader(fh))
            finally:
                os.chdir(cwd)

        self.assertEqual(result, (0, 0, 0, 1))
        self.assertEqual(rows[0]["status"], "lovart_still_running")
        self.assertIn("white-background", rows[0]["error"])

    def test_process_products_marks_support_credit_confirmation_as_manual_action(self):
        class Product:
            id = "SKU-CREDIT"
            name_cn = "Product"
            language = "Portuguese"
            selling_points = "points"
            image_paths = ["output/SKU-CREDIT/image_1.png"]

        class Gemini:
            def generate_prompt(self, **kwargs):
                raise AssertionError("should stop before Gemini when support image needs confirmation")

        class Lovart:
            def create_project(self, product_id, product_name_cn=""):
                return "project-credit"

            def create_support_image(self, **kwargs):
                return {
                    "final_status": "pending_confirmation",
                    "project_id": kwargs["project_id"],
                    "generation_succeeded": False,
                    "warning": "Lovart showed a 52-credit confirmation in unlimited mode.",
                }

        class Logger:
            def info(self, message):
                pass

            def warning(self, message):
                pass

            def error(self, message):
                raise AssertionError(message)

        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                product_dir = Path("output") / "SKU-CREDIT"
                product_dir.mkdir(parents=True)
                (product_dir / "image_1.png").write_bytes(b"product")

                result = _process_products([Product()], Gemini(), Lovart(), Logger(), Path("runs") / "run")
                with (Path("runs") / "run" / "summary.csv").open(encoding="utf-8", newline="") as fh:
                    rows = list(csv.DictReader(fh))
            finally:
                os.chdir(cwd)

        self.assertEqual(result, (0, 1, 0, 0))
        self.assertEqual(rows[0]["status"], "needs_manual_action")
        self.assertIn("credit confirmation", rows[0]["error"])


if __name__ == "__main__":
    unittest.main()
