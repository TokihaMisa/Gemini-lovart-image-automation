import json
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import webui
from webui import (
    build_gemini_health_check_command,
    build_ui,
    ui_run_gemini_health_check,
)


class FinishedProcess:
    returncode = 0

    def poll(self):
        return self.returncode


class RunningProcess:
    def __init__(self):
        self.stopped = False

    def poll(self):
        return 0 if self.stopped else None


class GeminiHealthCheckWebUITests(unittest.TestCase):
    def test_health_check_command_uses_source_entrypoint(self):
        command = build_gemini_health_check_command(
            "config.yaml",
            "runs/check.jsonl",
            executable="python.exe",
            frozen=False,
        )

        self.assertEqual(
            command,
            [
                "python.exe",
                str(Path(webui.__file__).with_name("app.py").resolve()),
                "--gemini-health-check",
                "--config",
                str(Path("config.yaml").resolve()),
                "--status-file",
                str(Path("runs/check.jsonl").resolve()),
            ],
        )

    def test_health_check_command_uses_packaged_executable(self):
        command = build_gemini_health_check_command(
            "config.yaml",
            "runs/check.jsonl",
            executable="Lovart_Auto.exe",
            frozen=True,
        )

        self.assertEqual(command[0:2], ["Lovart_Auto.exe", "--gemini-health-check"])
        self.assertNotIn("app.py", command)

    def test_app_routes_health_check_arguments_before_webui_start(self):
        calls = []
        fake_module = types.ModuleType("gemini_health_check")

        def fake_cli(config_path, status_file):
            calls.append((config_path, status_file))
            return 7

        fake_module.run_health_check_cli = fake_cli
        argv = [
            "app.py",
            "--gemini-health-check",
            "--config",
            "config.yaml",
            "--status-file",
            "status.jsonl",
        ]
        with patch.dict(sys.modules, {"gemini_health_check": fake_module}), patch.object(
            sys, "argv", argv
        ), self.assertRaises(SystemExit) as raised:
            runpy.run_path(
                str(Path(webui.__file__).with_name("app.py")),
                run_name="__main__",
            )

        self.assertEqual(raised.exception.code, 7)
        self.assertEqual(calls, [("config.yaml", "status.jsonl")])

    def test_ui_health_check_streams_partial_and_final_results_and_cleans_run_dir(self):
        records = [
            {"event": "running", "results": []},
            {
                "event": "result",
                "result": {
                    "name": "页面加载",
                    "state": "pass",
                    "message": "Gemini 页面已就绪",
                },
            },
            {"event": "complete", "exit_code": 0},
        ]

        def start_process(_config_path, status_file):
            status_file.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
                encoding="utf-8",
            )
            return FinishedProcess()

        with tempfile.TemporaryDirectory() as tmp, patch(
            "webui._health_check_runs_root", return_value=Path(tmp)
        ), patch(
            "webui._start_gemini_health_check_process", side_effect=start_process
        ), patch(
            "webui.time.sleep"
        ):
            values = list(ui_run_gemini_health_check(Path(tmp) / "config.yaml"))
            remaining = list(Path(tmp).iterdir())

        self.assertIn("正在启动", values[0])
        self.assertIn("页面加载", values[-1])
        self.assertIn("正常 1", values[-1])
        self.assertEqual(remaining, [])

    def test_ui_health_check_reports_child_exit_without_complete_record(self):
        def start_process(_config_path, status_file):
            status_file.write_text(
                json.dumps(
                    {
                        "event": "result",
                        "result": {
                            "name": "页面加载",
                            "state": "fail",
                            "message": "页面未就绪",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            process = FinishedProcess()
            process.returncode = 2
            return process

        with tempfile.TemporaryDirectory() as tmp, patch(
            "webui._health_check_runs_root", return_value=Path(tmp)
        ), patch(
            "webui._start_gemini_health_check_process", side_effect=start_process
        ), patch(
            "webui.time.sleep"
        ):
            values = list(ui_run_gemini_health_check())

        self.assertIn("异常 1", values[-1])
        self.assertIn("进程提前退出", values[-1])

    def test_ui_health_check_stops_child_when_stream_is_cancelled(self):
        process = RunningProcess()

        def start_process(_config_path, status_file):
            status_file.write_text(
                json.dumps(
                    {
                        "event": "result",
                        "result": {
                            "name": "页面加载",
                            "state": "pass",
                            "message": "页面已就绪",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            return process

        with tempfile.TemporaryDirectory() as tmp, patch(
            "webui._health_check_runs_root", return_value=Path(tmp)
        ), patch(
            "webui._start_gemini_health_check_process", side_effect=start_process
        ), patch(
            "webui._stop_health_check_process",
            side_effect=lambda child: setattr(child, "stopped", True),
        ) as stop:
            stream = ui_run_gemini_health_check()
            next(stream)
            next(stream)
            stream.close()

        stop.assert_called_once_with(process)
        self.assertTrue(process.stopped)

    @patch("webui.load_config", return_value={})
    def test_webui_contains_health_check_controls_and_event_chain(self, _load_config):
        demo = build_ui()
        try:
            config = demo.get_config_file()
        finally:
            demo.close()

        components = config["components"]
        values = {
            component.get("props", {}).get("value")
            for component in components
        }
        self.assertIn("Gemini 一键完整体检", values)
        result_components = [
            component
            for component in components
            if component.get("props", {}).get("elem_id")
            == "gemini-health-check-result"
        ]
        self.assertEqual(len(result_components), 1)
        self.assertEqual(
            result_components[0]["props"].get("value"),
            "尚未运行 Gemini 完整体检。",
        )

        run_event = next(
            dependency
            for dependency in config["dependencies"]
            if dependency.get("api_name") == "run_gemini_health_check"
        )
        self.assertEqual(len(run_event["outputs"]), 1)
        button_id = next(
            component["id"]
            for component in components
            if component.get("props", {}).get("value")
            == "Gemini 一键完整体检"
        )
        button_update_events = [
            dependency
            for dependency in config["dependencies"]
            if button_id in dependency.get("outputs", [])
        ]
        self.assertGreaterEqual(len(button_update_events), 2)


if __name__ == "__main__":
    unittest.main()
