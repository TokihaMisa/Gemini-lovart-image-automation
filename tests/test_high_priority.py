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
    product_output_dir,
    read_status,
    split_image_roles,
    update_status,
)


class HighPriorityBehaviorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
