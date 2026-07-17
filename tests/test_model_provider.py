import io
import json
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from model_provider import (
    DiscoveredModel,
    ModelProviderError,
    discover_models,
    model_choice_labels,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class ModelDiscoveryTests(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_gemini_discovery_paginates_and_keeps_generate_content_models(self, urlopen):
        urlopen.side_effect = [
            FakeResponse({
                "models": [{
                    "name": "models/gemini-3.5-flash",
                    "displayName": "Gemini 3.5 Flash",
                    "supportedGenerationMethods": ["generateContent"],
                    "thinking": True,
                }],
                "nextPageToken": "page-2",
            }),
            FakeResponse({
                "models": [{
                    "name": "models/gemini-3.5-pro",
                    "displayName": "Gemini 3.5 Pro",
                    "supportedGenerationMethods": ["generateContent"],
                    "thinking": True,
                }]
            }),
        ]
        models = discover_models("gemini", "key", "https://google.test/v1beta")
        self.assertEqual([m.model_id for m in models], ["gemini-3.5-flash", "gemini-3.5-pro"])
        self.assertIn("pageToken=page-2", urlopen.call_args_list[1].args[0].full_url)

    @patch("urllib.request.urlopen")
    def test_gemini_discovery_filters_non_prompt_models(self, urlopen):
        urlopen.return_value = FakeResponse({"models": [
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/imagen-4", "supportedGenerationMethods": ["predict"]},
            {"name": "models/gemini-live", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/veo-3", "supportedGenerationMethods": ["predictLongRunning"]},
            {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash", "supportedGenerationMethods": ["generateContent"]},
        ]})
        models = discover_models("gemini", "key", "https://google.test/v1beta")
        self.assertEqual([m.model_id for m in models], ["gemini-2.5-flash"])

    @patch("urllib.request.urlopen")
    def test_nvidia_discovery_sends_bearer_auth_and_filters_non_chat_models(self, urlopen):
        urlopen.return_value = FakeResponse({"data": [
            {"id": "nvidia/nv-embed-v1"},
            {"id": "black-forest-labs/flux.1"},
            {"id": "moonshotai/kimi-k2.5"},
        ]})
        models = discover_models("nvidia", "super-secret-key", "https://nvidia.test/v1")
        self.assertEqual([m.model_id for m in models], ["moonshotai/kimi-k2.5"])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer super-secret-key")

    @patch("urllib.request.urlopen")
    def test_discovery_rejects_missing_key_before_network_call(self, urlopen):
        with self.assertRaises(ModelProviderError) as ctx:
            discover_models("gemini", "", "https://google.test/v1beta")
        self.assertEqual(ctx.exception.code, "missing_key")
        urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_http_401_maps_to_secret_free_invalid_key_message(self, urlopen):
        urlopen.side_effect = HTTPError(
            "https://google.test", 401, "Unauthorized super-secret-key", {}, io.BytesIO(b"secret")
        )
        with self.assertRaises(ModelProviderError) as ctx:
            discover_models("gemini", "super-secret-key", "https://google.test/v1beta")
        self.assertEqual(ctx.exception.code, "authentication")
        self.assertNotIn("super-secret-key", ctx.exception.user_message)

    @patch("urllib.request.urlopen")
    def test_http_429_maps_to_rate_limit_without_secret(self, urlopen):
        urlopen.side_effect = HTTPError(
            "https://nvidia.test", 429, "Too Many Requests", {}, io.BytesIO(b"super-secret-key")
        )
        with self.assertRaises(ModelProviderError) as ctx:
            discover_models("nvidia", "super-secret-key", "https://nvidia.test/v1")
        self.assertEqual(ctx.exception.code, "rate_limit")
        self.assertNotIn("super-secret-key", str(ctx.exception))

    def test_malformed_gemini_base_url_never_exposes_key(self):
        with self.assertRaises(ModelProviderError) as ctx:
            discover_models("gemini", "super-secret-key", "not a valid url")
        self.assertEqual(ctx.exception.code, "network")
        self.assertNotIn("super-secret-key", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_empty_compatible_list_returns_empty_list(self, urlopen):
        urlopen.return_value = FakeResponse({"data": [{"id": "nvidia/nv-embed-v1"}]})
        self.assertEqual(discover_models("nvidia", "key", "https://nvidia.test/v1"), [])

    def test_model_choice_labels_show_thinking_and_image_status(self):
        models = [DiscoveredModel(
            provider="gemini",
            model_id="gemini-3.5-flash",
            display_name="Gemini 3.5 Flash",
            supports_generation=True,
            supports_thinking=True,
            image_input_status="unknown",
            recommendation="recommended",
        )]
        labels = model_choice_labels(models)
        self.assertEqual(labels[0][1], "gemini-3.5-flash")
        self.assertIn("Thinking", labels[0][0])
        self.assertIn("图片未验证", labels[0][0])
