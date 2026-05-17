import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from excel_reader import _build_dispimg_map
from lovart_bot import LovartBot, resolve_lovart_tool_config
from main import _dry_run_products, _choose_lovart_tool_options, parse_args
from utils import read_status


class LowPriorityBehaviorTests(unittest.TestCase):
    def test_parse_args_supports_noninteractive_run(self):
        args = parse_args([
            "--gemini", "api",
            "--lovart", "unlimited",
            "--limit", "2",
            "--dry-run",
            "--no-resume",
            "--lovart-image-model", "nano_banana_2,nano_banana_pro",
            "--lovart-model-selection", "force",
            "--lovart-reasoning", "thinking",
        ])
        self.assertEqual(args.gemini, "api")
        self.assertEqual(args.lovart, "unlimited")
        self.assertEqual(args.limit, 2)
        self.assertTrue(args.dry_run)
        self.assertFalse(args.resume)
        self.assertEqual(args.lovart_image_model, "nano_banana_2,nano_banana_pro")
        self.assertEqual(args.lovart_model_selection, "force")
        self.assertEqual(args.lovart_reasoning, "thinking")

    def test_choose_lovart_tool_options_prompts_when_not_overridden(self):
        args = parse_args(["--gemini", "api", "--lovart", "unlimited"])
        config = {"lovart": {"image_model": "auto", "model_selection": "prefer", "reasoning_mode": "fast"}}
        answers = iter(["4,5", "2", "2"])
        with patch("builtins.input", lambda prompt="": next(answers)):
            _choose_lovart_tool_options(config, args)

        self.assertEqual(config["lovart"]["image_model"], "nano_banana_2,nano_banana_pro")
        self.assertEqual(config["lovart"]["model_selection"], "force")
        self.assertEqual(config["lovart"]["reasoning_mode"], "thinking")

    def test_choose_lovart_tool_options_does_not_prompt_for_cli_overrides(self):
        args = parse_args([
            "--lovart-image-model", "gpt_image_2",
            "--lovart-model-selection", "force",
            "--lovart-reasoning", "thinking",
        ])
        config = {"lovart": {"image_model": "auto", "model_selection": "prefer", "reasoning_mode": "fast"}}
        with patch("builtins.input", side_effect=AssertionError("should not prompt")):
            _choose_lovart_tool_options(config, args)

        self.assertEqual(config["lovart"]["image_model"], "gpt_image_2")
        self.assertEqual(config["lovart"]["model_selection"], "force")
        self.assertEqual(config["lovart"]["reasoning_mode"], "thinking")

    def test_dry_run_writes_status_and_summary_without_external_clients(self):
        class Product:
            id = "SKU-DRY"
            name_cn = "测试商品"
            language = "Portuguese"
            selling_points = "卖点"
            image_paths = ["output/SKU-DRY/image_1.jpeg"]

        class Logger:
            def info(self, message):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            run_dir = Path(tmp) / "runs" / "run"
            success, fail, skipped, still_running = _dry_run_products([Product()], Logger(), run_dir, output_dir)
            status = read_status(output_dir / "SKU-DRY")
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual((success, fail, skipped, still_running), (0, 0, 1, 0))
        self.assertTrue(status["dry_run"])
        self.assertEqual(summary[0]["status"], "dry_run")

    def test_lovart_result_success_requires_done_status_and_artifact(self):
        result = LovartBot._normalize_result(
            {"items": [{"artifacts": [{"type": "image", "content": "https://example.test/a.png"}]}]},
            "done",
            "project-1",
        )
        self.assertTrue(result["generation_succeeded"])

        text_only = LovartBot._normalize_result({"items": [{"text": "no image"}]}, "done", "project-1")
        self.assertFalse(text_only["generation_succeeded"])
        self.assertIn("warning", text_only)

    def test_resolve_lovart_tool_config_can_force_image_model_and_thinking(self):
        cfg = {
            "image_model": "nano_banana_2,nano_banana_pro",
            "model_selection": "force",
            "reasoning_mode": "thinking",
        }
        tool_config = resolve_lovart_tool_config(cfg)
        self.assertEqual(tool_config["include_tools"], [
            "generate_image_nano_banana_2",
            "generate_image_nano_banana_pro",
        ])
        self.assertIsNone(tool_config["prefer_models"])
        self.assertEqual(tool_config["mode"], "thinking")

    def test_resolve_lovart_tool_config_prefers_gpt_image_2(self):
        cfg = {"image_model": "gpt_image_2", "model_selection": "prefer"}
        tool_config = resolve_lovart_tool_config(cfg)
        self.assertEqual(tool_config["prefer_models"], {"IMAGE": ["generate_image_gpt_image_2"]})
        self.assertIsNone(tool_config["include_tools"])

    def test_resolve_lovart_tool_config_ignores_auto_in_multi_model(self):
        cfg = {"image_model": "auto,nano_banana_pro", "model_selection": "prefer"}
        tool_config = resolve_lovart_tool_config(cfg)
        self.assertEqual(tool_config["image_models"], ["nano_banana_pro"])
        self.assertEqual(tool_config["prefer_models"], {"IMAGE": ["generate_image_nano_banana_pro"]})

    def test_submit_and_poll_reuses_existing_project(self):
        class Logger:
            def warning(self, message):
                pass

        bot = LovartBot.__new__(LovartBot)
        bot.cfg = {}
        bot.tool_config = resolve_lovart_tool_config({
            "image_model": "gpt_image_2",
            "model_selection": "prefer",
            "reasoning_mode": "fast",
        })
        bot.logger = Logger()
        calls = []

        def fake_submit_once(**kwargs):
            calls.append(kwargs)
            return {"final_status": "done", "generation_succeeded": True}, "same-project", "t3"

        bot._submit_and_poll_once = fake_submit_once
        with tempfile.TemporaryDirectory() as tmp:
            product_dir = Path(tmp)
            result, project_id, thread_id = bot._submit_and_poll(
                product_dir=product_dir,
                product_id="SKU-1",
                step_name="detail",
                prompt="prompt",
                image_paths=[],
                confirmation_advisor=None,
                product_name_cn="产品",
                language="Portuguese",
                selling_points="points",
                project_id="same-project",
            )

        self.assertEqual((result["final_status"], project_id, thread_id), ("done", "same-project", "t3"))
        self.assertEqual([call["project_id"] for call in calls], ["same-project"])
        self.assertEqual(calls[0]["tool_config"]["prefer_models"], {"IMAGE": ["generate_image_gpt_image_2"]})

    def test_credit_confirmation_in_unlimited_mode_keeps_polling(self):
        class Logger:
            def __init__(self):
                self.messages = []

            def info(self, message):
                self.messages.append(message)

            def warning(self, message):
                self.messages.append(message)

        class Skill:
            def __init__(self):
                self.result_calls = 0

            def get_status(self, thread_id):
                return {"status": "running"} if self.result_calls < 2 else {"status": "done"}

            def get_result(self, thread_id):
                self.result_calls += 1
                if self.result_calls == 1:
                    return {"pending_confirmation": {"message": "consume 50 credits"}}
                return {"items": [{"artifacts": [{"type": "image", "content": "https://example.test/a.png"}]}]}

        bot = LovartBot.__new__(LovartBot)
        bot.cfg = {"wait_timeout": 5, "poll_interval": 1}
        bot.logger = Logger()
        bot.skill = Skill()
        bot._fast_mode = False

        with tempfile.TemporaryDirectory() as tmp, patch("time.sleep", lambda seconds: None):
            product_dir = Path(tmp)
            result = bot._poll_with_progress("thread-1", "project-1", product_dir=product_dir)
            status = read_status(product_dir)

        self.assertEqual(result["final_status"], "done")
        self.assertTrue(result["generation_succeeded"])
        self.assertTrue(status["lovart_credit_prompt_waiting"])
        self.assertFalse(status["needs_manual_action"])

    def test_build_dispimg_map_reads_wps_cellimages_relationships(self):
        cellimages = """<?xml version="1.0" encoding="UTF-8"?>
<etc:cellImages xmlns:etc="http://www.wps.cn/officeDocument/2017/etCustomData"
 xmlns:xdr="http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <etc:cellImage>
    <xdr:pic>
      <xdr:nvPicPr><xdr:cNvPr name="ID_ABC"/></xdr:nvPicPr>
      <xdr:blipFill><a:blip r:embed="rId1"/></xdr:blipFill>
    </xdr:pic>
  </etc:cellImage>
</etc:cellImages>
"""
        rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Target="media/image1.jpeg"/>
</Relationships>
"""

        class Logger:
            def info(self, message):
                pass

            def warning(self, message):
                raise AssertionError(message)

        with tempfile.TemporaryDirectory() as tmp:
            xlsx = Path(tmp) / "sample.xlsx"
            with zipfile.ZipFile(xlsx, "w") as zf:
                zf.writestr("xl/cellimages.xml", cellimages)
                zf.writestr("xl/_rels/cellimages.xml.rels", rels)
                zf.writestr("xl/media/image1.jpeg", b"jpeg")
            mapping = _build_dispimg_map(str(xlsx), Logger())

        self.assertEqual(mapping, {"ID_ABC": "xl/media/image1.jpeg"})


if __name__ == "__main__":
    unittest.main()
