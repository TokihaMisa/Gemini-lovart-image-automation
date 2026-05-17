import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from excel_reader import resolve_image_scan_config
from gemini_bot import GeminiBot
from utils import write_run_summary


class MediumPriorityBehaviorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
