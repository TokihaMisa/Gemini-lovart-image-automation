import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import (
    append_result,
    build_final_lovart_images,
    build_lovart_image_note,
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

    def test_load_dotenv_sets_missing_values_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("GEMINI_API_KEY=from-file\nLOVART_ACCESS_KEY='ak_test'\n", encoding="utf-8")

            with patch.dict(os.environ, {"GEMINI_API_KEY": "from-env"}, clear=False):
                load_dotenv(path)
                self.assertEqual(os.environ["GEMINI_API_KEY"], "from-env")
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

        self.assertEqual(rows[0], ["product_id", "product_name", "status", "project_url", "error"])
        self.assertEqual(rows[1], ["SKU-123", 'Name, "quoted"', "success", "https://example.test", ""])
        self.assertEqual(rows[2], ["SKU-456", "Recovered item", "success", "https://example.test/2", ""])
        self.assertEqual(len(rows), 3)

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


if __name__ == "__main__":
    unittest.main()
