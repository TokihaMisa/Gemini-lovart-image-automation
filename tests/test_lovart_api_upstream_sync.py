import json
import unittest
from urllib.error import URLError
from unittest.mock import patch

from lovart_api import AgentSkill, AgentSkillError


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps({"code": 0, "data": {"ok": True}}).encode("utf-8")


class LovartUpstreamSyncTests(unittest.TestCase):
    def test_post_uses_idempotency_key_without_automatic_retry(self):
        client = AgentSkill("https://lovart.test", "access-key", "secret-key")
        requests = []

        def fail_once(request, **_kwargs):
            requests.append(request)
            raise URLError(ConnectionResetError("temporary reset"))

        with patch("lovart_api.urllib.request.urlopen", side_effect=fail_once) as urlopen:
            with self.assertRaises(AgentSkillError):
                client._request("POST", "/v1/openapi/chat", body={"prompt": "test"})

        self.assertEqual(urlopen.call_count, 1)
        self.assertTrue(requests[0].get_header("Idempotency-key"))

    def test_get_keeps_transient_retry_without_idempotency_key(self):
        client = AgentSkill("https://lovart.test", "access-key", "secret-key")
        requests = []

        def request_then_succeed(request, **_kwargs):
            requests.append(request)
            if len(requests) == 1:
                raise URLError(ConnectionResetError("temporary reset"))
            return _Response()

        with patch(
            "lovart_api.urllib.request.urlopen", side_effect=request_then_succeed
        ) as urlopen, patch("network_retry.time.sleep"):
            result = client._request("GET", "/v1/openapi/chat/status")

        self.assertTrue(result["ok"])
        self.assertEqual(urlopen.call_count, 2)
        self.assertIsNone(requests[0].get_header("Idempotency-key"))

    def test_explicit_post_retry_reuses_same_idempotency_key(self):
        client = AgentSkill("https://lovart.test", "access-key", "secret-key")
        requests = []

        def request_then_succeed(request, **_kwargs):
            requests.append(request)
            if len(requests) == 1:
                raise URLError(ConnectionResetError("temporary reset"))
            return _Response()

        with patch(
            "lovart_api.urllib.request.urlopen", side_effect=request_then_succeed
        ), patch("network_retry.time.sleep"):
            result = client._request(
                "POST", "/v1/openapi/chat", body={"prompt": "test"}, retries=2
            )

        keys = [request.get_header("Idempotency-key") for request in requests]
        self.assertTrue(result["ok"])
        self.assertEqual(len(requests), 2)
        self.assertTrue(keys[0])
        self.assertEqual(keys[0], keys[1])


if __name__ == "__main__":
    unittest.main()
