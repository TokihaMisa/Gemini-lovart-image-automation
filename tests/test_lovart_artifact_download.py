import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

from lovart_api import AgentSkill
from lovart_bot import LovartBot


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return b"generated-image"


class _Logger:
    def info(self, *_args):
        pass

    def warning(self, *_args):
        pass

    def error(self, *_args):
        pass


class LovartArtifactDownloadTests(unittest.TestCase):
    def test_download_retries_transient_cdn_failure(self):
        result = {
            "items": [
                {
                    "artifacts": [
                        {
                            "type": "image",
                            "content": "https://a.lovart.ai/private/image.png?token=secret",
                        }
                    ]
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "lovart_api.urllib.request.urlopen",
            side_effect=[URLError(ConnectionResetError("reset")), _Response()],
        ) as urlopen, patch("network_retry.time.sleep"):
            downloaded = AgentSkill.download_artifacts(result, tmp)

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(len(downloaded), 1)
        self.assertTrue(downloaded[0]["local_path"])
        self.assertNotIn("secret", downloaded[0].get("error", ""))

    def test_support_image_surfaces_cdn_download_failure(self):
        bot = LovartBot.__new__(LovartBot)
        bot.cfg = {}
        bot.logger = _Logger()
        bot.tool_config = {"image_model": "auto"}
        bot._execute_with_fallback = lambda *_args, **_kwargs: (
            {
                "generation_succeeded": True,
                "final_status": "done",
                "items": [
                    {
                        "artifacts": [
                            {
                                "type": "image",
                                "content": "https://a.lovart.ai/private/image.png?token=secret",
                            }
                        ]
                    }
                ],
            },
            "project-id",
            "thread-id",
        )

        class Skill:
            @staticmethod
            def download_artifacts(_result, _output_dir, prefix="lovart"):
                return [
                    {
                        "type": "image",
                        "url": "https://a.lovart.ai/private/image.png?token=secret",
                        "local_path": None,
                        "error": "download failed",
                        "error_kind": "transient",
                        "host": "a.lovart.ai",
                        "new": False,
                    }
                ]

        bot.skill = Skill()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "lovart_bot.product_output_dir", return_value=Path(tmp)
        ), patch("lovart_bot.update_status"), patch.object(
            bot, "_rename_project"
        ):
            returned = bot.create_support_image(
                product_id="SKU-1",
                step_name="white_bg",
                prompt="prompt",
                image_paths=["image.png"],
                project_id="project-id",
            )

        self.assertFalse(returned["generation_succeeded"])
        self.assertIn("a.lovart.ai", returned["warning"])
        self.assertIn("Clash", returned["warning"])
        self.assertNotIn("token=secret", returned["warning"])

    def test_support_step_reuses_completed_thread_after_download_failure(self):
        bot = LovartBot.__new__(LovartBot)
        bot.cfg = {}
        bot.logger = _Logger()

        class Skill:
            @staticmethod
            def get_result(thread_id):
                self.assertEqual(thread_id, "thread-id")
                return {
                    "items": [
                        {
                            "artifacts": [
                                {"type": "image", "content": "https://a.lovart.ai/image.png"}
                            ]
                        }
                    ]
                }

        bot.skill = Skill()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "lovart_bot.read_status",
            return_value={
                "thread_id": "thread-id",
                "lovart_white_bg_download_failed": True,
            },
        ), patch.object(bot, "_upload_images") as upload:
            result, project_id, thread_id = bot._submit_and_poll_once(
                product_dir=Path(tmp),
                product_id="SKU-1",
                step_name="white_bg",
                attempt_name="primary",
                project_id="project-id",
                prompt="prompt",
                image_paths=["image.png"],
                confirmation_advisor=None,
                product_name_cn="Product",
                language="English",
                selling_points="",
                tool_config={"image_model": "auto", "model_selection": "auto", "mode": None},
            )

        upload.assert_not_called()
        self.assertTrue(result["generation_succeeded"])
        self.assertEqual(project_id, "project-id")
        self.assertEqual(thread_id, "thread-id")

    def test_done_status_waits_for_delayed_artifact_result(self):
        bot = LovartBot.__new__(LovartBot)
        bot.cfg = {
            "wait_timeout": 10,
            "poll_interval": 0,
            "artifact_result_attempts": 3,
            "artifact_result_retry_delay": 0,
        }
        bot.logger = _Logger()
        bot._fast_mode = False

        class Skill:
            def __init__(self):
                self.result_calls = 0

            @staticmethod
            def get_status(_thread_id):
                return {"status": "done"}

            def get_result(self, _thread_id):
                self.result_calls += 1
                if self.result_calls < 3:
                    return {"items": [{"text": "still syncing"}]}
                return {
                    "items": [
                        {
                            "artifacts": [
                                {"type": "image", "content": "https://a.lovart.ai/image.png"}
                            ]
                        }
                    ]
                }

        bot.skill = Skill()
        with patch("lovart_bot.time.sleep"):
            result = bot._poll_with_progress("thread-id", "project-id")

        self.assertTrue(result["generation_succeeded"])
        self.assertEqual(bot.skill.result_calls, 3)

    def test_text_only_done_result_starts_fresh_model_fallback_thread(self):
        bot = LovartBot.__new__(LovartBot)
        bot.logger = _Logger()
        bot.tool_config = {
            "image_model": "auto",
            "tool_names": [],
        }
        calls = []

        def execute(**kwargs):
            calls.append(dict(kwargs))
            if len(calls) == 1:
                return {
                    "final_status": "done",
                    "generation_succeeded": False,
                    "warning": "Lovart finished without returning image artifacts.",
                }, "project-id", "text-only-thread"
            return {
                "final_status": "done",
                "generation_succeeded": True,
                "items": [
                    {
                        "artifacts": [
                            {"type": "image", "content": "https://a.lovart.ai/image.png"}
                        ]
                    }
                ],
            }, "project-id", "fallback-thread"

        with patch.object(bot, "set_image_model") as set_model:
            def select_model(model_name):
                bot.tool_config = {
                    "image_model": model_name,
                    "tool_names": [f"generate_image_{model_name}"],
                }

            set_model.side_effect = select_model
            result, project_id, thread_id = bot._execute_with_fallback(
                execute,
                project_id="project-id",
            )

        self.assertTrue(result["generation_succeeded"])
        self.assertEqual(project_id, "project-id")
        self.assertEqual(thread_id, "fallback-thread")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("force_new_thread", calls[0])
        self.assertTrue(calls[1]["force_new_thread"])
        set_model.assert_called_once_with("nano_banana_pro")


if __name__ == "__main__":
    unittest.main()
