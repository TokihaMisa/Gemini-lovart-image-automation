import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from excel_reader import resolve_image_scan_config
from gemini_bot import GeminiBot
from main import _process_products, _resolve_browser_executable_for_run, resolve_browser_executable
from utils import update_status, write_run_summary


class MediumPriorityBehaviorTests(unittest.TestCase):
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

            def count(self):
                if "add_2" in self.selector:
                    return 1
                if 'input[type="file"]' in self.selector:
                    return 1
                return 0

            def click(self, timeout=None):
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
                return False

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
            def create_project(self, product_id):
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

            def create_project(self, product_id):
                events.append(("create_project", product_id))
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
        self.assertIn(("create_project", "SKU-INVALID"), events)
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
            def create_project(self, product_id):
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
            def create_project(self, product_id):
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
