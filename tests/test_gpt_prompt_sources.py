import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

import main
import webui
from chatgpt_bot import ChatGPTBot, ChatGPTPageNotReadyError, ChatGPTReplyError
from error_guidance import explain_failure
from gemini_browser_session import GeminiPageState, build_login_helper_command, inspect_chatgpt_page
from gpt_text_api import GPTTextAPI, resolve_gpt_text_settings
from model_provider import ModelProviderError


class JsonResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_gpt_text_settings_default_to_sub2api_profile_and_gpt6():
    config = {"openai_image": {"profiles": {"sub2api": {"base_url": "https://relay.test/v1"}}}}
    assert resolve_gpt_text_settings(config) == ("https://relay.test/v1", "gpt-6")
    config["gpt_text_api"] = {"base_url": "https://text.test/v1", "model": "gpt-5.6"}
    assert resolve_gpt_text_settings(config) == ("https://text.test/v1", "gpt-5.6")


def test_gpt_text_api_sends_images_without_sampling_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("LOVART_OUTPUT_DIR", str(tmp_path / "output"))
    image = tmp_path / "white.png"
    Image.new("RGB", (8, 8), "white").save(image)
    captured = {}

    def fake_urlopen(request, timeout=None, context=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = dict(request.header_items())
        captured["url"] = request.full_url
        return JsonResponse({"choices": [{"message": {"content": "[[SCREEN 01]]\nok\n[[/SCREEN 01]]"}}]})

    client = GPTTextAPI(api_key="key", model="gpt-6", base_url="https://relay.test/v1")
    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.generate_prompt("甩棍", "es", "镀锌", [str(image)], product_id="SKU-1")

    body = captured["body"]
    assert captured["url"] == "https://relay.test/v1/chat/completions"
    assert body["model"] == "gpt-6"
    assert "temperature" not in body and "max_tokens" not in body
    parts = body["messages"][1]["content"]
    assert parts[0]["type"] == "text" and "甩棍" in parts[0]["text"]
    assert parts[1]["image_url"]["url"].startswith("data:image/")
    assert captured["headers"]["User-agent"].startswith("Lovart-Image-Automation")
    assert (tmp_path / "output" / "SKU-1" / "gemini_prompt.txt").read_text(encoding="utf-8") == result


def test_gpt_text_api_http_errors_are_mapped_without_secrets():
    from urllib.error import HTTPError
    import io

    def fail(request, timeout=None, context=None):
        raise HTTPError(request.full_url, 401, "secret-body", {}, io.BytesIO(b"secret-body"))

    client = GPTTextAPI(api_key="super-secret", model="gpt-6", base_url="https://relay.test/v1")
    with patch("urllib.request.urlopen", fail), pytest.raises(ModelProviderError) as ctx:
        client._call("hello", [])
    assert ctx.value.code == "authentication"
    assert "super-secret" not in str(ctx.value)


class FakeChatGPTPage:
    def __init__(self, url, payload=None, error=None, closed=False):
        self.url = url
        self.payload = payload or {}
        self.error = error
        self.closed = closed

    def evaluate(self, _script):
        if self.error:
            raise self.error
        return self.payload

    def is_closed(self):
        return self.closed


@pytest.mark.parametrize(
    ("url", "payload", "expected"),
    [
        ("https://chatgpt.com/?temporary-chat=true", {"has_composer": True, "has_account": True}, GeminiPageState.READY),
        # The composer renders before the session is known: not ready without the account menu.
        ("https://chatgpt.com/?temporary-chat=true", {"has_composer": True}, GeminiPageState.PAGE_LOADING),
        ("https://chatgpt.com/", {"has_composer": True, "has_login_button": True}, GeminiPageState.WAITING_LOGIN),
        ("https://auth.openai.com/mfa-challenge/x", {}, GeminiPageState.WAITING_LOGIN),
        ("https://chatgpt.com/", {"has_challenge": True, "has_composer": True, "has_account": True}, GeminiPageState.PAGE_LOADING),
    ],
)
def test_chatgpt_page_classification(url, payload, expected):
    assert inspect_chatgpt_page(FakeChatGPTPage(url, payload)).state is expected


def test_chatgpt_navigation_during_inspection_is_loading_not_error():
    navigating = FakeChatGPTPage("https://chatgpt.com/", error=RuntimeError("Execution context was destroyed"))
    assert inspect_chatgpt_page(navigating).state is GeminiPageState.PAGE_LOADING
    closed = FakeChatGPTPage("https://chatgpt.com/", error=RuntimeError("closed"), closed=True)
    assert inspect_chatgpt_page(closed).state is GeminiPageState.ERROR


def test_login_helper_command_passes_chatgpt_site():
    gemini = build_login_helper_command("config.yaml", executable="python.exe", frozen=False)
    chatgpt = build_login_helper_command("config.yaml", executable="L.exe", frozen=True, site="chatgpt")
    assert "--site" not in gemini
    assert chatgpt[-2:] == ["--site", "chatgpt"]


class Logger:
    def info(self, *_args):
        pass

    warning = info


def test_chatgpt_short_reply_is_reported_as_refusal(tmp_path, monkeypatch):
    monkeypatch.setenv("LOVART_OUTPUT_DIR", str(tmp_path))
    bot = ChatGPTBot(page=None, config={}, logger=Logger())
    monkeypatch.setattr(bot, "_open_temporary_chat", lambda: None)
    monkeypatch.setattr(bot, "_ask", lambda *_args: "抱歉，我无法帮助处理这个请求。")
    with pytest.raises(ChatGPTReplyError) as ctx:
        bot.generate_prompt("甩棍", "es", "", [], product_id="SKU-1")
    assert explain_failure(str(ctx.value)).cause == "ChatGPT 拒绝了这个商品"
    assert not (tmp_path / "SKU-1" / "gemini_prompt.txt").exists()


def test_chatgpt_refuses_to_continue_outside_temporary_chat(monkeypatch):
    class Page:
        url = "https://chatgpt.com/"

    bot = ChatGPTBot(page=Page(), config={}, logger=Logger())
    ready = type("Status", (), {"state": GeminiPageState.READY, "ready": True})()
    monkeypatch.setattr("chatgpt_bot.navigate_gemini_with_retry", lambda *_args, **_kwargs: ready)
    with pytest.raises(ChatGPTPageNotReadyError, match="临时聊天"):
        bot._open_temporary_chat()


def test_chatgpt_login_guidance_is_not_the_gemini_one():
    assert explain_failure("ChatGPT 未登录，请先登录").cause == "ChatGPT 未登录"
    assert explain_failure("Gemini requires login").cause == "Gemini 未登录"


def test_main_accepts_new_prompt_sources():
    assert main.parse_args(["--prompt-source", "gpt_api"]).prompt_source == "gpt_api"
    assert main.parse_args(["--prompt-source", "chatgpt_browser"]).prompt_source == "chatgpt_browser"


def test_workspace_model_dropdown_for_new_sources():
    choices, selected, interactive = webui.resolve_model_dropdown("gpt_api", [], [], {"gpt_text_api": {"model": "gpt-5.6"}})
    assert selected == "gpt-5.6" and interactive and ("gpt-6", "gpt-6") in choices
    _choices, selected, interactive = webui.resolve_model_dropdown("chatgpt_browser", [], [], {})
    assert selected == "由浏览器页面选择" and not interactive
    saved = webui.persist_selected_model({}, "gpt_api", "gpt-6.1-sol")
    assert saved["gpt_text_api"]["model"] == "gpt-6.1-sol"
    assert ("ChatGPT 网页临时聊天", "chatgpt_browser") in webui.PROMPT_SOURCE_CHOICES


@pytest.mark.real_preflight
def test_gpt_api_preflight_requires_saved_sub2api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    excel = tmp_path / "products.xlsx"
    import openpyxl

    workbook = openpyxl.Workbook()
    workbook.active.append(["id", "name"])
    workbook.active.append(["SKU-1", "甩棍"])
    workbook.save(excel)
    config = {
        "excel": {"path": str(excel)},
        "openai_image": {"profiles": {"sub2api": {"base_url": "https://relay.test/v1"}}},
    }
    kwargs = dict(
        prompt_source="gpt_api", support_provider="lovart", detail_provider="openai_image",
        gemini_key="", nvidia_key="", lovart_access="a", lovart_secret="b",
        output_dir=str(tmp_path / "out"),
    )
    problems = webui._preflight_problems(config, **kwargs)
    assert any("Sub2API 密钥" in problem for problem in problems)
    assert webui._preflight_problems(config, gpt_key_saved=True, **kwargs) == []


def test_chatgpt_browser_source_is_guarded_like_gemini(monkeypatch):
    monkeypatch.setattr(webui, "login_helper_is_active", lambda _paths: True)
    assert webui.guard_gemini_browser_task("chatgpt_browser") == webui._GEMINI_PROFILE_BUSY_MESSAGE
    assert webui.guard_gemini_browser_task("gpt_api") is None


def test_login_check_only_closes_the_matching_site(monkeypatch):
    status = type("Status", (), {"state": GeminiPageState.READY, "ready": True, "url": "https://chatgpt.com/"})()
    closed = []
    monkeypatch.setattr(webui, "login_helper_is_active", lambda _paths: True)
    monkeypatch.setattr(webui, "read_login_status", lambda _path: status)
    monkeypatch.setattr(webui, "request_login_helper_close", lambda path: closed.append(path))
    assert "ChatGPT 那一行" in webui.check_gemini_login_and_close(site="gemini")
    assert not closed
    assert "ChatGPT 登录已确认" in webui.check_gemini_login_and_close(site="chatgpt")
    assert closed


def test_exit_button_covers_page_before_server_stops():
    source = Path(webui.__file__).read_text(encoding="utf-8")
    assert "js=SHUTDOWN_PAGE_JS" in source
    assert ".toast-wrap" in webui.SHUTDOWN_PAGE_JS and "window.close()" in webui.SHUTDOWN_PAGE_JS
