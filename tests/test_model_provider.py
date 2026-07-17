import io
import http.client
import json
import socket
import ssl
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from model_provider import (
    DiscoveredModel,
    ModelProviderError,
    ModelTestResult,
    discover_models,
    model_choice_labels,
    test_selected_model,
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

    @patch("urllib.request.urlopen")
    def test_url_error_wrapped_timeout_dns_and_tls_are_secret_safe(self, urlopen):
        failures = [
            ("timeout", socket.timeout("super-secret-key"), "timeout"),
            ("dns", socket.gaierror("super-secret-key"), "network"),
            ("tls", ssl.SSLError("super-secret-key"), "network"),
        ]
        for name, reason, expected_code in failures:
            with self.subTest(name=name):
                urlopen.side_effect = URLError(reason)
                with self.assertRaises(ModelProviderError) as ctx:
                    discover_models("gemini", "super-secret-key", "https://google.test/v1beta")
                self.assertEqual(ctx.exception.code, expected_code)
                self.assertNotIn("super-secret-key", ctx.exception.user_message)

    @patch("urllib.request.urlopen")
    def test_raw_os_error_is_mapped_without_exposing_key(self, urlopen):
        urlopen.side_effect = ConnectionResetError("super-secret-key")
        with self.assertRaises(ModelProviderError) as ctx:
            discover_models("nvidia", "super-secret-key", "https://nvidia.test/v1")
        self.assertEqual(ctx.exception.code, "network")
        self.assertNotIn("super-secret-key", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_http_protocol_error_is_mapped_without_exposing_key(self, urlopen):
        urlopen.side_effect = http.client.HTTPException("super-secret-key")
        with self.assertRaises(ModelProviderError) as ctx:
            discover_models("nvidia", "super-secret-key", "https://nvidia.test/v1")
        self.assertEqual(ctx.exception.code, "network")
        self.assertNotIn("super-secret-key", str(ctx.exception))

    def test_malformed_gemini_base_url_never_exposes_key(self):
        with self.assertRaises(ModelProviderError) as ctx:
            discover_models("gemini", "super-secret-key", "not a valid url")
        self.assertEqual(ctx.exception.code, "network")
        self.assertNotIn("super-secret-key", str(ctx.exception))

    def test_invalid_ipv6_gemini_base_url_never_exposes_key(self):
        with self.assertRaises(ModelProviderError) as ctx:
            discover_models("gemini", "super-secret-key", "https://[bad")
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

    def test_model_choice_labels_distinguish_all_image_input_statuses(self):
        models = [
            DiscoveredModel("gemini", "reported", "Reported", True, None, "reported", "available"),
            DiscoveredModel("gemini", "verified", "Verified", True, None, "verified", "available"),
            DiscoveredModel("gemini", "failed", "Failed", True, None, "failed", "available"),
            DiscoveredModel("gemini", "unknown", "Unknown", True, None, "unknown", "available"),
        ]
        self.assertEqual(
            [label for label, _ in model_choice_labels(models)],
            [
                "Reported · 图片已报告支持",
                "Verified · 图片已验证支持",
                "Failed · 图片不支持",
                "Unknown · 图片未验证",
            ],
        )


class ModelCompatibilityTests(unittest.TestCase):
    def test_selected_model_is_not_collected_as_a_pytest_test(self):
        self.assertFalse(test_selected_model.__test__)

    @patch("urllib.request.urlopen")
    def test_gemini_model_test_sends_inline_png_and_small_output_limit(self, urlopen):
        urlopen.return_value = FakeResponse({"candidates": [{"content": {"parts": [{"text": "OK"}]}}]})
        result = test_selected_model("gemini", "key", "https://google.test/v1beta", "gemini-2.5-flash")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        parts = payload["contents"][0]["parts"]
        self.assertTrue(result.ok)
        self.assertTrue(request.full_url.startswith("https://google.test/v1beta/models/gemini-2.5-flash:generateContent"))
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 32)
        self.assertEqual(parts[1]["inline_data"]["mime_type"], "image/png")

    @patch("urllib.request.urlopen")
    def test_nvidia_model_test_sends_data_url_and_small_output_limit(self, urlopen):
        urlopen.return_value = FakeResponse({"choices": [{"message": {"content": "OK"}}]})
        result = test_selected_model("nvidia", "key", "https://nvidia.test/v1", "moonshotai/kimi-k2.5")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        content = payload["messages"][0]["content"]
        self.assertTrue(result.ok)
        self.assertEqual(request.full_url, "https://nvidia.test/v1/chat/completions")
        self.assertEqual(payload["max_tokens"], 32)
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    @patch("urllib.request.urlopen")
    def test_model_test_requires_nonempty_text(self, urlopen):
        urlopen.return_value = FakeResponse({"choices": [{"message": {"content": ""}}]})
        with self.assertRaises(ModelProviderError) as ctx:
            test_selected_model("nvidia", "key", "https://nvidia.test/v1", "model")
        self.assertEqual(ctx.exception.code, "empty_response")

    @patch("urllib.request.urlopen")
    def test_model_test_reports_latency_and_success_message(self, urlopen):
        urlopen.return_value = FakeResponse({"choices": [{"message": {"content": "OK"}}]})
        result = test_selected_model("nvidia", "key", "https://nvidia.test/v1", "model")
        self.assertIsInstance(result, ModelTestResult)
        self.assertGreaterEqual(result.latency_ms, 0)
        self.assertIn("支持图片输入", result.message)

    @patch("urllib.request.urlopen")
    def test_model_test_error_does_not_echo_api_key(self, urlopen):
        urlopen.side_effect = HTTPError(
            "https://nvidia.test", 403, "super-secret-key", {}, io.BytesIO(b"super-secret-key")
        )
        with self.assertRaises(ModelProviderError) as ctx:
            test_selected_model("nvidia", "super-secret-key", "https://nvidia.test/v1", "model")
        self.assertNotIn("super-secret-key", ctx.exception.user_message)
