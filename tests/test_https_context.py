import ssl
import urllib.request
from pathlib import Path
from unittest.mock import patch

import certifi
from PIL import Image

import network_retry
from openai_image_api import OpenAIImageAPI, OpenAIImageAPIConfig


def test_https_context_trusts_system_roots_and_certifi_bundle():
    loaded = []
    original = ssl.SSLContext.load_verify_locations

    def spy(self, *args, **kwargs):
        loaded.append(kwargs.get("cafile") or (args[0] if args else None))
        return original(self, *args, **kwargs)

    with patch.object(ssl.SSLContext, "load_verify_locations", spy):
        context = network_retry.build_https_context()

    assert certifi.where() in loaded
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_shared_https_context_is_reused_but_builder_returns_private_copies():
    assert network_retry.https_context() is network_retry.https_context()
    assert network_retry.build_https_context() is not network_retry.build_https_context()


@patch("openai_image_api.urllib.request.build_opener")
def test_gpt_image_requests_use_the_shared_https_context(build_opener, tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 2), "white").save(source)
    build_opener.return_value.open.side_effect = OSError("stop after opener is built")
    client = OpenAIImageAPI(
        OpenAIImageAPIConfig.from_config(
            {"openai_image": {"base_url": "https://api.lk888.ai/v1"}}, api_key="test-key"
        ),
        sleep=lambda _seconds: None,
    )

    try:
        client.generate_edit("prompt", [source], tmp_path / "out.png")
    except Exception:
        pass

    handlers = [
        handler
        for call in build_opener.call_args_list
        for handler in call.args
        if isinstance(handler, urllib.request.HTTPSHandler)
    ]
    assert handlers
    assert all(handler._context is network_retry.https_context() for handler in handlers)


def test_lovart_gets_a_private_context_it_may_relax():
    import lovart_api

    assert lovart_api._ssl_ctx is not network_retry.https_context()
