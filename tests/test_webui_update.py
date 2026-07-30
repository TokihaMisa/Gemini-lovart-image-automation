from __future__ import annotations

import unittest
from unittest import mock

import updater
from webui import ui_check_update


class WebUiUpdateTests(unittest.TestCase):
    def test_check_error_is_not_rendered_as_latest(self):
        result = updater.UpdateCheckResult(
            updater.UpdateStatus.ERROR, message="无法检查更新，请稍后重试。"
        )
        with mock.patch.object(updater, "check_update_details", return_value=result):
            messages = list(ui_check_update())
        self.assertIn("检查更新失败", messages[-1])
        self.assertIn("重试", messages[-1])
        self.assertNotIn("最新版本", messages[-1])

    def test_latest_is_rendered_separately(self):
        result = updater.UpdateCheckResult(updater.UpdateStatus.UP_TO_DATE)
        with mock.patch.object(updater, "check_update_details", return_value=result):
            messages = list(ui_check_update())
        self.assertIn("最新版本", messages[-1])
        self.assertNotIn("失败", messages[-1])

    def test_install_receives_verified_manifest_integrity_fields(self):
        result = updater.UpdateCheckResult(
            updater.UpdateStatus.UPDATE_AVAILABLE,
            version="1.3.1",
            url="https://downloads.example.test/update.zip",
            changelog="安全更新",
            sha256="a" * 64,
            size=123,
        )

        def install(_url, output_queue, **kwargs):
            self.assertEqual(kwargs["expected_sha256"], "a" * 64)
            self.assertEqual(kwargs["expected_size"], 123)
            output_queue.put("完整性校验成功。")
            output_queue.put(None)

        with mock.patch.object(updater, "check_update_details", return_value=result), mock.patch.object(
            updater, "download_and_install_update", side_effect=install
        ):
            messages = list(ui_check_update())
        self.assertTrue(any("完整性校验成功" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
