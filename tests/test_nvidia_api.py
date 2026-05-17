import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main import _build_nvidia_api, _choose_prompt_source, parse_args
from nvidia_api import NvidiaAPI, resolve_nvidia_model


class NvidiaAPIBehaviorTests(unittest.TestCase):
    def test_parse_args_supports_kimi_nvidia_model(self):
        args = parse_args(["--prompt-source", "nvidia", "--nvidia-model", "kimi"])
        self.assertEqual(args.prompt_source, "nvidia")
        self.assertEqual(args.nvidia_model, "kimi")

    def test_parse_args_rejects_non_image_nvidia_models(self):
        with self.assertRaises(SystemExit):
            parse_args(["--prompt-source", "nvidia", "--nvidia-model", "glm_5_1"])

    def test_choose_prompt_source_uses_kimi_without_model_submenu(self):
        args = parse_args([])
        config = {"nvidia_api": {}}
        answers = iter(["3"])
        with patch("builtins.input", lambda prompt="": next(answers)):
            source = _choose_prompt_source(config, args)

        self.assertEqual(source, "nvidia")
        self.assertEqual(config["nvidia_api"]["model_choice"], "kimi")

    def test_resolve_nvidia_model_uses_configured_model_id(self):
        cfg = {"model_choice": "kimi", "models": {"kimi": "moonshotai/kimi-k2.5"}}
        self.assertEqual(resolve_nvidia_model(cfg), "moonshotai/kimi-k2.5")

    def test_nvidia_payload_includes_images_as_data_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "image.jpeg"
            image.write_bytes(b"fake-jpeg")
            client = NvidiaAPI(api_key="key", model="moonshotai/kimi-k2.5", logger=None)
            payload = client._build_payload("hello", [str(image)])

        content = payload["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "hello"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_build_nvidia_api_sends_images_for_kimi(self):
        cfg = {
            "nvidia_api": {
                "base_url": "https://integrate.api.nvidia.com/v1",
                "model_choice": "kimi",
                "send_images": True,
                "models": {"kimi": "moonshotai/kimi-k2.5"},
            }
        }
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "key"}):
            client = _build_nvidia_api(cfg, logger=None)

        self.assertTrue(client.send_images)

    def test_nvidia_extracts_chat_completion_text(self):
        data = {"choices": [{"message": {"content": "result text"}}]}
        self.assertEqual(NvidiaAPI._extract_text(data), "result text")


if __name__ == "__main__":
    unittest.main()
