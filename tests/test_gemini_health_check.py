import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gemini_health_check import (
    CheckState,
    HealthCheckResult,
    PlaywrightGeminiHealthCheckHarness,
    render_health_check,
    run_checks,
    run_gemini_health_check,
    run_health_check_cli,
    run_health_check_cli_from_argv,
)


class FakeHealthCheckHarness:
    def __init__(self, *, temporary_chat=True, upload_error=None):
        self.temporary_chat = temporary_chat
        self.upload_error = upload_error
        self.upload_calls = 0
        self.send_probe_calls = 0
        self.context_closed = False
        self.blank_image_exists = True

    def check_browser(self):
        return True

    def check_profile_lock(self):
        return True

    def check_page_ready(self):
        return True

    def check_logged_in(self):
        return True

    def check_editor(self):
        return True

    def enter_temporary_chat(self):
        return self.temporary_chat

    def open_mode_menu(self):
        return True

    def select_flash(self):
        return True

    def select_extended_thinking(self):
        return True

    def check_upload_control(self):
        return True

    def upload_blank_image(self):
        self.upload_calls += 1
        if self.upload_error:
            raise self.upload_error
        return True

    def check_send_button(self):
        self.send_probe_calls += 1
        return True

    def close(self):
        self.context_closed = True
        self.blank_image_exists = False


class CollectingReporter:
    def __init__(self):
        self.results = []

    def emit(self, result):
        self.results.append(result)


def result_for(results, name):
    return next(item for item in results if item.name == name)


class GeminiHealthCheckTests(unittest.TestCase):
    def test_health_check_result_json_is_bounded_and_rendered_in_order(self):
        bounded = HealthCheckResult(
            "页" * 41,
            CheckState.PASS,
            "已" * 241,
        ).to_json()
        self.assertEqual(len(bounded["name"]), 40)
        self.assertEqual(len(bounded["message"]), 240)
        self.assertEqual(bounded["state"], "pass")

        results = [
            HealthCheckResult("页面加载", CheckState.PASS, "Gemini 页面已就绪"),
            HealthCheckResult("临时对话", CheckState.FAIL, "未找到临时对话按钮"),
            HealthCheckResult("图片上传", CheckState.SKIP, "前置检查失败"),
        ]
        self.assertEqual(
            results[0].to_json(),
            {
                "name": "页面加载",
                "state": "pass",
                "message": "Gemini 页面已就绪",
            },
        )
        rendered = render_health_check(results)
        self.assertLess(rendered.index("页面加载"), rendered.index("临时对话"))
        self.assertLess(rendered.index("临时对话"), rendered.index("图片上传"))
        self.assertIn("✅", rendered)
        self.assertIn("❌", rendered)
        self.assertIn("⚠️", rendered)
        self.assertIn("正常 1 · 异常 1 · 跳过 1", rendered)

    def test_runner_emits_all_twelve_checks_in_approved_order(self):
        harness = FakeHealthCheckHarness()

        results = list(run_checks(harness))

        self.assertEqual(
            [item.name for item in results],
            [
                "浏览器环境",
                "配置占用",
                "页面加载",
                "登录状态",
                "提示词编辑器",
                "临时对话",
                "模式菜单",
                "Flash 模型",
                "扩展思考",
                "上传控件",
                "图片上传",
                "发送按钮",
            ],
        )
        self.assertTrue(all(item.state is CheckState.PASS for item in results))
        self.assertEqual(harness.upload_calls, 1)
        self.assertEqual(harness.send_probe_calls, 1)
        self.assertTrue(harness.context_closed)

    def test_runner_skips_upload_when_temporary_chat_fails(self):
        harness = FakeHealthCheckHarness(temporary_chat=False)

        results = list(run_checks(harness))

        self.assertIs(
            result_for(results, "临时对话").state,
            CheckState.FAIL,
        )
        self.assertIs(result_for(results, "图片上传").state, CheckState.SKIP)
        self.assertIs(result_for(results, "发送按钮").state, CheckState.SKIP)
        self.assertEqual(harness.upload_calls, 0)
        self.assertEqual(harness.send_probe_calls, 0)

    def test_runner_always_closes_and_hides_raw_upload_error(self):
        harness = FakeHealthCheckHarness(
            upload_error=RuntimeError("private raw error C:\\Users\\Somebody")
        )

        results = list(run_checks(harness))

        self.assertTrue(harness.context_closed)
        self.assertFalse(harness.blank_image_exists)
        rendered = render_health_check(results)
        self.assertNotIn("private raw error", rendered)
        self.assertNotIn("C:\\Users", rendered)
        self.assertEqual(result_for(results, "图片上传").message, "上传失败")

    def test_run_gemini_health_check_reports_results_and_returns_failure_count(self):
        reporter = CollectingReporter()
        harness = FakeHealthCheckHarness(temporary_chat=False)

        exit_code = run_gemini_health_check(
            "config.yaml",
            reporter,
            harness_factory=lambda _path: harness,
        )

        self.assertEqual(exit_code, 1)
        self.assertEqual(len(reporter.results), 12)
        self.assertIs(
            result_for(reporter.results, "临时对话").state,
            CheckState.FAIL,
        )

    def test_playwright_harness_uses_production_controls_without_sending(self):
        class FakeBot:
            def __init__(self):
                self.calls = []
                self.cfg = {}

            def health_check_editor(self):
                self.calls.append("editor")
                return True

            def _start_temporary_chat(self):
                self.calls.append("temporary_chat")
                return True

            def health_check_temporary_chat_active(self):
                return True

            def _open_mode_menu(self):
                self.calls.append("open_mode")
                return True

            def _click_flash_model(self):
                self.calls.append("flash")
                return True

            def _select_extended_thinking_option(self):
                self.calls.append("extended")
                return True

            def health_check_upload_control(self):
                self.calls.append("upload_control")
                return True

            def _upload_images(self, paths):
                self.calls.append(("upload_blank", len(paths)))
                return True

            def health_check_send_button(self):
                self.calls.append("send_probe")
                return True

        bot = FakeBot()
        harness = PlaywrightGeminiHealthCheckHarness.from_test_bot(bot)

        results = list(run_checks(harness))

        self.assertTrue(all(item.state is CheckState.PASS for item in results))
        self.assertEqual(
            bot.calls,
            [
                "editor",
                "temporary_chat",
                "open_mode",
                "flash",
                "extended",
                "upload_control",
                ("upload_blank", 1),
                "send_probe",
            ],
        )
        self.assertFalse(hasattr(bot, "send_message"))

    def test_cli_writes_flushed_jsonl_protocol(self):
        def fake_run(_config_path, reporter):
            reporter.emit(
                HealthCheckResult("页面加载", CheckState.PASS, "Gemini 页面已就绪")
            )
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            status_file = Path(tmp) / "status.jsonl"
            with patch(
                "gemini_health_check.run_gemini_health_check",
                side_effect=fake_run,
            ):
                exit_code = run_health_check_cli("config.yaml", status_file)

            records = [
                json.loads(line)
                for line in status_file.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(exit_code, 0)
        self.assertEqual(records[0], {"event": "running", "results": []})
        self.assertEqual(records[1]["event"], "result")
        self.assertEqual(records[1]["result"]["name"], "页面加载")
        self.assertEqual(records[-1], {"event": "complete", "exit_code": 0})

    def test_cli_argument_router_requires_status_file(self):
        with self.assertRaises(SystemExit):
            run_health_check_cli_from_argv(
                ["--gemini-health-check", "--config", "config.yaml"]
            )


if __name__ == "__main__":
    unittest.main()
