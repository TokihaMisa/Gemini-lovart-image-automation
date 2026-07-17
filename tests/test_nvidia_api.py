import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from gemini_api import GeminiAPI
from main import _build_gemini_api, _build_nvidia_api, _choose_prompt_source, parse_args
from model_provider import ModelProviderError
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
        config = {"nvidia_api": {"api_key": "key"}}
        answers = iter(["3"])
        with patch("builtins.input", lambda prompt="": next(answers)), patch.dict(os.environ, {}, clear=True):
            source = _choose_prompt_source(config, args)

        self.assertEqual(source, "nvidia")
        self.assertEqual(config["nvidia_api"]["model_choice"], "kimi")

    def test_choose_prompt_source_reprompts_when_gemini_api_key_missing(self):
        args = parse_args([])
        config = {"gemini_api": {}, "nvidia_api": {}}
        answers = iter(["2", "1"])

        with patch("builtins.input", lambda prompt="": next(answers)), patch.dict(os.environ, {}, clear=True):
            source = _choose_prompt_source(config, args)

        self.assertEqual(source, "gemini_browser")

    def test_choose_prompt_source_reprompts_when_nvidia_api_key_missing(self):
        args = parse_args([])
        config = {"gemini_api": {}, "nvidia_api": {}}
        answers = iter(["3", "1"])

        with patch("builtins.input", lambda prompt="": next(answers)), patch.dict(os.environ, {}, clear=True):
            source = _choose_prompt_source(config, args)

        self.assertEqual(source, "gemini_browser")
        self.assertEqual(config["nvidia_api"]["model_choice"], "kimi")

    def test_choose_prompt_source_keeps_explicit_gemini_api_even_without_key(self):
        args = parse_args(["--prompt-source", "gemini_api"])
        config = {"gemini_api": {}}

        with patch("builtins.input", side_effect=AssertionError("should not prompt")), patch.dict(os.environ, {}, clear=True):
            source = _choose_prompt_source(config, args)

        self.assertEqual(source, "gemini_api")

    def test_resolve_nvidia_model_uses_configured_model_id(self):
        cfg = {"model_choice": "kimi", "models": {"kimi": "moonshotai/kimi-k2.5"}}
        self.assertEqual(resolve_nvidia_model(cfg), "moonshotai/kimi-k2.5")

    def test_resolve_nvidia_model_prefers_direct_model_id(self):
        cfg = {
            "model": "nvidia/new-vision-model",
            "model_choice": "kimi",
            "models": {"kimi": "moonshotai/kimi-k2.5"},
        }
        self.assertEqual(resolve_nvidia_model(cfg), "nvidia/new-vision-model")

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

    def test_nvidia_client_rejects_invalid_base_url(self):
        with self.assertRaises(ModelProviderError) as ctx:
            NvidiaAPI(api_key="key", model="nvidia/model", base_url="not-a-url")
        self.assertEqual(ctx.exception.code, "invalid_base_url")

    def test_nvidia_client_rejects_model_id_line_breaks(self):
        with self.assertRaises(ModelProviderError) as ctx:
            NvidiaAPI(
                api_key="key",
                model="nvidia/model\r\nInjected: value",
                base_url="https://nvidia.test/v1",
            )
        self.assertEqual(ctx.exception.code, "invalid_model")

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

    def test_build_nvidia_api_uses_configured_base_url(self):
        cfg = {
            "nvidia_api": {
                "base_url": "https://nvidia.proxy.test/v1",
                "model": "nvidia/custom",
                "send_images": True,
            }
        }
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "key"}):
            client = _build_nvidia_api(cfg, logger=None)
        self.assertEqual(client.base_url, "https://nvidia.proxy.test/v1")

    def test_build_gemini_api_uses_configured_base_url_and_model(self):
        cfg = {
            "gemini_api": {
                "base_url": "https://gemini.proxy.test/v1beta",
                "model": "gemini-custom",
            }
        }
        with patch.dict("os.environ", {"GEMINI_API_KEY": "key"}):
            client = _build_gemini_api(cfg, logger=MagicMock())
        self.assertEqual(client.base_url, "https://gemini.proxy.test/v1beta")
        self.assertEqual(client.model, "gemini-custom")

    @patch("urllib.request.urlopen")
    def test_gemini_formal_request_uses_client_base_url(self, urlopen):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({
                    "candidates": [{"content": {"parts": [{"text": "result"}]}}]
                }).encode("utf-8")

        urlopen.return_value = Response()
        client = GeminiAPI(
            api_key="key",
            model="gemini-custom",
            base_url="https://gemini.proxy.test/v1beta",
        )
        self.assertEqual(client._call("hello", []), "result")
        request = urlopen.call_args.args[0]
        self.assertTrue(request.full_url.startswith(
            "https://gemini.proxy.test/v1beta/models/gemini-custom:generateContent"
        ))

    def test_nvidia_extracts_chat_completion_text(self):
        data = {"choices": [{"message": {"content": "result text"}}]}
        self.assertEqual(NvidiaAPI._extract_text(data), "result text")


if __name__ == "__main__":
    unittest.main()
