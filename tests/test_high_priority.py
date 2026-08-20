import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import (
    append_result,
    build_design_prompt,
    build_final_lovart_images,
    build_lovart_image_note,
    build_lovart_prompt,
    env_or_config,
    is_product_completed,
    load_dotenv,
    organize_output_folders,
    product_output_dir,
    read_status,
    split_image_roles,
    update_status,
)


class HighPriorityBehaviorTests(unittest.TestCase):
    def test_repository_examples_contain_no_real_api_keys(self):
        examples = [
            Path("config.example.yaml"),
            Path(".env.example"),
            Path("README.md"),
            Path("PROJECT_OVERVIEW.md"),
        ]
        for path in examples:
            self.assertNotIn("sk-", path.read_text(encoding="utf-8"))

        self.assertIn(
            "OPENAI_IMAGE_API_KEY=your_openai_image_api_key",
            Path(".env.example").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "Copy this tracked example to your local untracked .env",
            Path(".env.example").read_text(encoding="utf-8"),
        )
        self.assertIn("GPT Image", Path("README.md").read_text(encoding="utf-8"))
        overview = Path("PROJECT_OVERVIEW.md").read_text(encoding="utf-8")
        self.assertIn("detail_page_count_snapshot", overview)
        self.assertNotIn("生成 12 屏详情页提示词", overview)

    def test_gpt_image_examples_document_the_async_media_task_migration(self):
        config = Path("config.example.yaml").read_text(encoding="utf-8")
        env = Path(".env.example").read_text(encoding="utf-8")
        readme = Path("README.md").read_text(encoding="utf-8")

        self.assertIn('base_url: ""', config)
        self.assertIn('model: "gpt-image-2"', config)
        self.assertIn('resolution: "1K"', config)
        self.assertIn('merge_reference_images: false', config)
        self.assertIn("with and without `/v1`", readme)
        self.assertIn("`/v1/media/generate`", readme)
        self.assertIn("`/v1/media/status`", readme)
        self.assertIn("14 direct Data URL", readme)
        self.assertIn("optional merge", readme)
        self.assertIn("submit once", readme)
        self.assertIn("task ID", readme)
        self.assertIn("5 seconds, then 10 seconds", readme)
        self.assertIn("600-second", readme)
        self.assertIn("only in your local `.env`", readme)
        self.assertIn("without task IDs", readme)
        self.assertIn("not recoverable", readme)
        self.assertIn("migrated once", readme)
        self.assertIn("OPENAI_IMAGE_API_KEY=your_openai_image_api_key", env)

        forbidden = (
            "hapi",
            "openai-compatible",
            "/images/" + "edits",
            "async_" + "edits",
            "sync fallback",
        )
        overview = Path("PROJECT_OVERVIEW.md").read_text(encoding="utf-8")
        webui_source = Path("webui.py").read_text(encoding="utf-8")
        self.assertIn("`/v1/media/generate`", overview)
        self.assertIn("`/v1/media/status`", overview)
        self.assertIn("任务 ID", overview)
        self.assertIn("GPT Image 媒体任务", webui_source)
        for path, source in (
            ("config.example.yaml", config),
            ("README.md", readme),
            ("PROJECT_OVERVIEW.md", overview),
        ):
            self.assertTrue(all(token not in source.lower() for token in forbidden), path)
        webui_forbidden = tuple(token for token in forbidden if token != "async_" + "edits")
        self.assertTrue(
            all(token not in webui_source.lower() for token in webui_forbidden),
            "webui.py",
        )
        self.assertEqual(webui_source.count('pop("async_edits", None)'), 2)

    def test_env_or_config_prefers_environment(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "from-env"}):
            value = env_or_config({"api_key": "from-config"}, "api_key", "GEMINI_API_KEY")
        self.assertEqual(value, "from-env")

    def test_load_dotenv_overrides_stale_environment_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("GEMINI_API_KEY=from-file\nLOVART_ACCESS_KEY='ak_test'\n", encoding="utf-8")

            with patch.dict(os.environ, {"GEMINI_API_KEY": "from-env"}, clear=False):
                load_dotenv(path)
                self.assertEqual(os.environ["GEMINI_API_KEY"], "from-file")
                self.assertEqual(os.environ["LOVART_ACCESS_KEY"], "ak_test")

    def test_product_output_dir_uses_product_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = product_output_dir("SKU-123", tmp)
            self.assertEqual(out, Path(tmp) / "SKU-123")
            self.assertTrue(out.exists())

    def test_status_json_tracks_completed_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = product_output_dir("SKU-123", tmp)
            update_status(out, "lovart_done", project_url="https://example.test")

            data = read_status(out)
            self.assertTrue(data["lovart_done"])
            self.assertEqual(data["project_url"], "https://example.test")
            self.assertTrue(is_product_completed(out))

    def test_organizer_moves_gpt_image_completed_product_to_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            product_dir = base / "3_处理中" / "SKU-GPT-DONE"
            product_dir.mkdir(parents=True)
            detail_path = product_dir / "gpt_image" / "detail" / "01.png"
            detail_path.parent.mkdir(parents=True)
            detail_path.write_bytes(b"generated")
            update_status(
                product_dir,
                "detail_generation_done",
                detail_generation_complete=True,
                lovart_done=False,
                detail_images=[str(detail_path)],
            )

            organize_output_folders(base)

            moved_dir = base / "1_完全做好" / "SKU-GPT-DONE"
            self.assertTrue(moved_dir.is_dir())
            self.assertFalse(product_dir.exists())
            status = read_status(moved_dir)
            self.assertEqual(
                status["detail_images"],
                [str(moved_dir / "gpt_image" / "detail" / "01.png")],
            )

    def test_append_result_writes_header_escapes_csv_and_upserts_by_product_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            append_result(path, "SKU-123", 'Name, "quoted"', "https://example.test")
            append_result(path, "SKU-456", "Failed item", status="failed", error='bad, "quoted"')
            append_result(path, "SKU-456", "Recovered item", "https://example.test/2", status="success")

            with path.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.reader(fh))

        self.assertEqual(rows[0], ["product_id", "product_name", "status", "project_url", "error", "used_model"])
        self.assertEqual(rows[1], ["SKU-123", 'Name, "quoted"', "success", "https://example.test", "", ""])
        self.assertEqual(rows[2], ["SKU-456", "Recovered item", "success", "https://example.test/2", "", ""])
        self.assertEqual(len(rows), 3)

    def test_append_result_reads_existing_gbk_results_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            content = (
                "product_id,product_name,status,project_url,error\n"
                "SKU-OLD,\u6d4b\u8bd5\u5546\u54c1,success,https://example.test/old,\n"
            )
            path.write_bytes(content.encode("gbk"))

            append_result(path, "SKU-NEW", "\u65b0\u5546\u54c1", "https://example.test/new")

            with path.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))

        self.assertEqual(rows[0]["product_name"], "\u6d4b\u8bd5\u5546\u54c1")
        self.assertEqual(rows[0]["used_model"], "")
        self.assertEqual(rows[1]["product_id"], "SKU-NEW")
        self.assertEqual(rows[1]["project_url"], "https://example.test/new")

    def test_append_result_keeps_existing_model_for_explicit_same_provider_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            append_result(
                path,
                "SKU-MODEL",
                "Product",
                status="success",
                used_model="gpt-image-2",
            )

            append_result(
                path,
                "SKU-MODEL",
                "Product",
                status="failed",
                error="temporary failure",
                preserve_existing_model=True,
            )

            with path.open("r", encoding="utf-8", newline="") as fh:
                row = next(csv.DictReader(fh))

        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["used_model"], "gpt-image-2")

    def test_append_result_blank_model_does_not_keep_prior_provider_model_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            append_result(
                path,
                "SKU-MODEL-SWITCH",
                "Product",
                status="success",
                used_model="nano_banana_2",
            )

            append_result(
                path,
                "SKU-MODEL-SWITCH",
                "Product",
                status="failed",
                error="new provider failed",
            )

            with path.open("r", encoding="utf-8", newline="") as fh:
                row = next(csv.DictReader(fh))

        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["used_model"], "")

    def test_split_image_roles_preserves_empty_accessory_and_dimension_slots(self):
        roles = split_image_roles(["product.png", "", "", "ref1.png", "", "ref2.png"])

        self.assertEqual(roles["product_image"], "product.png")
        self.assertEqual(roles["accessory_image"], "")
        self.assertEqual(roles["dimension_image"], "")
        self.assertEqual(roles["reference_images"], ["ref1.png", "ref2.png"])

    def test_final_lovart_images_keep_reference_sheet_last(self):
        images = build_final_lovart_images(
            white_image="white.png",
            scene_image="scene.png",
            accessory_image="accessory.png",
            dimension_image="dimension.png",
            reference_sheet="reference_sheet.jpg",
        )

        self.assertEqual(images, [
            "white.png",
            "scene.png",
            "accessory.png",
            "dimension.png",
            "reference_sheet.jpg",
        ])

    def test_lovart_image_note_marks_last_image_as_reference_only(self):
        note = build_lovart_image_note(
            has_reference_sheet=True,
            has_accessory_image=True,
            has_dimension_image=True,
        )

        self.assertIn("最后一张图（图5）才是合并参考图", note)
        self.assertIn("除最后一张参考图以外", note)

    def test_lovart_image_note_allows_same_product_reference_for_shape(self):
        note = build_lovart_image_note(
            has_reference_sheet=True,
            has_accessory_image=False,
            has_dimension_image=False,
            reference_images_are_product=True,
        )

        self.assertIn("同一个产品", note)
        self.assertIn("外形", note)
        self.assertIn("其他角度", note)

    def test_lovart_image_note_limits_non_product_reference_to_style(self):
        note = build_lovart_image_note(
            has_reference_sheet=True,
            has_accessory_image=False,
            has_dimension_image=False,
            reference_images_are_product=False,
        )

        self.assertIn("只参考风格", note)
        self.assertIn("不要把参考图里的产品当成我的产品", note)

    def test_prompts_use_source_image_size_instead_of_default_square_ratio(self):
        design_prompt = build_design_prompt("Product", "Portuguese", "points", image_size="4:5")
        lovart_prompt = build_lovart_prompt(
            product_name_cn="Product",
            language="Portuguese",
            selling_points="points",
            generated_prompt="generated detail prompt",
            image_size="4:5",
        )

        self.assertIn("4:5", design_prompt)
        self.assertNotIn("1:1", design_prompt)
        self.assertIn("4:5", lovart_prompt)
        self.assertNotIn("1:1", lovart_prompt)

    def test_design_prompt_requires_stable_screen_markers(self):
        prompt = build_design_prompt("杯子", "英语", "保温", prompt_settings={"detail_page_count": 3})

        self.assertIn("[[SCREEN 01]]", prompt)
        self.assertIn("[[/SCREEN 01]]", prompt)
        self.assertIn("[[SCREEN 03]]", prompt)


if __name__ == "__main__":
    unittest.main()
