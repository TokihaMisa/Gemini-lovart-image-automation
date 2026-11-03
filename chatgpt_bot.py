"""ChatGPT web temporary-chat prompt writer, driven through the shared browser profile."""

import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from gemini_browser_session import (
    CHATGPT_TEMPORARY_CHAT_URL,
    GeminiPageState,
    inspect_chatgpt_page,
    navigate_gemini_with_retry,
    retry_policy_from_config,
)
from prompt_settings import normalize_prompt_settings
from utils import (
    build_design_prompt,
    build_lovart_confirmation_prompt,
    get_resource_path,
    parse_lovart_confirmation_decision,
    product_output_dir,
    update_status,
)


class ChatGPTLoginRequiredError(RuntimeError):
    def __init__(self, message: str = "ChatGPT 未登录，请先在「API 与模型」页打开 ChatGPT 登录浏览器完成登录。"):
        super().__init__(message)


class ChatGPTPageNotReadyError(RuntimeError):
    def __init__(self, message: str = "ChatGPT 页面未准备完成（可能卡在人机验证），请检查网络或在登录浏览器中完成验证后重试。"):
        super().__init__(message)


class ChatGPTReplyError(RuntimeError):
    """The page did not deliver a usable answer (refusal, error banner or timeout)."""


COMPOSER_SELECTOR = "#prompt-textarea, div.ProseMirror[contenteditable=true]"
# Newer builds dropped data-testid; the send control is the composer form's submit button.
SEND_BUTTON_SELECTOR = (
    '[data-testid="send-button"], button#composer-submit-button, '
    'form button[type="submit"], button[aria-label="发送"], button[aria-label="Send prompt"]'
)
# The composer also carries a disabled single-file camera input; skip it.
FILE_INPUT_SELECTORS = (
    'input[type="file"][multiple]:not([disabled]):not([capture])',
    'input[type="file"]:not([disabled]):not([capture])',
)
# Shorter answers are refusals or error notes, never a usable design brief.
MIN_PROMPT_CHARS = 200

_STATE_SCRIPT = r"""
() => {
  const visible = (node) => {
    if (!node) return false;
    const rect = node.getBoundingClientRect();
    const style = getComputedStyle(node);
    return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
  };
  // Older builds tag the message itself; newer ones put an h4 role marker in a search unit
  // whose answer lives in [data-markdown-text-style="assistant-message"] blocks.
  const turns = document.querySelectorAll('[data-message-author-role="assistant"], [data-conversation-role="assistant"]');
  const last = turns.length ? turns[turns.length - 1] : null;
  let lastText = '';
  if (last && last.matches('[data-message-author-role]')) {
    lastText = (last.querySelector('.markdown') || last).innerText;
  } else if (last) {
    const unit = last.closest('[data-chatgpt-search-unit-key], [data-content-search-unit-key]') || last.parentElement;
    let blocks = [...unit.querySelectorAll('[data-markdown-text-style="assistant-message"]')];
    if (!blocks.length) blocks = [...unit.querySelectorAll('.markdown')];
    lastText = blocks.map((block) => block.innerText).join('\n\n');
  }
  const composer = document.querySelector('#prompt-textarea, div.ProseMirror[contenteditable=true]');
  const form = composer?.closest('form') || document.querySelector('form');
  const send = document.querySelector('[data-testid="send-button"], button#composer-submit-button')
    || form?.querySelector('button[type="submit"]');
  const attachments = form
    ? [...form.querySelectorAll('img, [data-testid*="attachment"], [data-testid*="file-tile"]')].filter(visible).length
    : 0;
  const uploading = form
    ? [...form.querySelectorAll('[role="progressbar"], circle[stroke-dasharray], .animate-spin')].some(visible)
    : false;
  return {
    assistant_count: turns.length,
    last_text: lastText,
    generating: [...document.querySelectorAll('[data-testid="stop-button"], button[aria-label*="Stop"], button[aria-label*="停止"]')].some(visible),
    send_ready: Boolean(send && visible(send) && !send.disabled && send.getAttribute('aria-disabled') !== 'true'),
    attachments,
    uploading,
  };
}
"""


class ChatGPTBot:
    """Same prompt-client interface as GeminiBot, using a fresh ChatGPT temporary chat per product."""

    def __init__(self, page, config, logger, run_dir=None, prompt_settings=None):
        self.page = page
        self.logger = logger
        self.run_dir = Path(run_dir) if run_dir else None
        self.cfg = config.get("chatgpt", {}) if isinstance(config.get("chatgpt"), dict) else {}
        self._browser_config = config.get("browser", {}) if isinstance(config.get("browser"), dict) else {}
        self.reply_timeout = float(self.cfg.get("reply_timeout", 600) or 600)
        self.upload_timeout = float(self.cfg.get("upload_timeout", 120) or 120)
        self.prompt_settings = normalize_prompt_settings(
            prompt_settings if prompt_settings is not None else config.get("prompt_settings")
        )

    def generate_prompt(
        self,
        product_name_cn: str,
        language: str,
        selling_points: str,
        image_paths: list[str],
        product_id: str | None = None,
        image_size: str = "",
    ) -> str:
        product_id = product_id or product_name_cn
        self.logger.info(f"ChatGPT: starting for '{product_name_cn}'")
        self._open_temporary_chat()
        preamble = get_resource_path("preamble.txt").read_text(encoding="utf-8")
        prompt = f"{preamble}\n\n---\n\n{build_design_prompt(product_name_cn, language, selling_points, image_size=image_size, prompt_settings=self.prompt_settings)}"
        result = self._ask(prompt, image_paths, product_id)
        if len(result) < MIN_PROMPT_CHARS:
            self._save_debug_snapshot(product_id, "short-response")
            raise ChatGPTReplyError(f"ChatGPT 回复过短（{len(result)} 字），可能拒绝了请求：{result[:80]}")
        out_dir = product_output_dir(product_id)
        (out_dir / "gemini_prompt.txt").write_text(result, encoding="utf-8")
        update_status(out_dir, "chatgpt_done", gemini_chars=len(result))
        self.logger.info(f"ChatGPT: got response ({len(result)} chars)")
        return result

    def advise_lovart_confirmation(
        self,
        product_id: str,
        product_name_cn: str,
        language: str,
        selling_points: str,
        confirmation_text: str,
        confirmation_payload,
        project_id: str,
        thread_id: str,
        round_index: int,
        max_auto_confirm_credits: int,
        lovart_mode: str,
    ) -> dict:
        prompt = build_lovart_confirmation_prompt(
            product_name_cn=product_name_cn,
            language=language,
            selling_points=selling_points,
            confirmation_text=confirmation_text,
            confirmation_payload=confirmation_payload,
            project_id=project_id,
            thread_id=thread_id,
            round_index=round_index,
            max_auto_confirm_credits=max_auto_confirm_credits,
            lovart_mode=lovart_mode,
        )
        self.logger.info(f"ChatGPT: Lovart confirmation decision requested round {round_index}")
        response = self._ask(prompt, [], product_id)
        decision = parse_lovart_confirmation_decision(response)
        out_dir = product_output_dir(product_id)
        (out_dir / f"lovart_confirmation_gemini_{round_index}.txt").write_text(response, encoding="utf-8")
        (out_dir / f"lovart_confirmation_decision_{round_index}.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        update_status(
            out_dir,
            "lovart_confirmation_advised",
            lovart_confirmation_round=round_index,
            lovart_confirmation_decision=decision["decision"],
            lovart_confirmation_reason=decision["reason"],
        )
        return decision

    def _open_temporary_chat(self) -> None:
        policy = retry_policy_from_config({"browser": self._browser_config})
        status = navigate_gemini_with_retry(
            self.page, CHATGPT_TEMPORARY_CHAT_URL, policy, logger=self.logger,
            inspect=inspect_chatgpt_page, label="ChatGPT",
        )
        if status.state is GeminiPageState.WAITING_LOGIN:
            raise ChatGPTLoginRequiredError()
        if status.state is not GeminiPageState.READY or not status.ready:
            raise ChatGPTPageNotReadyError()
        query = parse_qs(urlsplit(self.page.url).query)
        if query.get("temporary-chat") != ["true"]:
            # ChatGPT drops the flag when temporary chat is unavailable; never save history silently.
            raise ChatGPTPageNotReadyError("ChatGPT 没有进入临时聊天，已停止，避免在聊天记录里留下商品资料。")
        self.logger.info("ChatGPT: temporary chat ready")

    def _state(self) -> dict:
        try:
            state = self.page.evaluate(_STATE_SCRIPT)
        except Exception:
            return {}
        return state if isinstance(state, dict) else {}

    def _ask(self, text: str, image_paths: list[str], product_id: str) -> str:
        previous = int(self._state().get("assistant_count", 0) or 0)
        composer = self.page.locator(COMPOSER_SELECTOR).first
        composer.click(timeout=10000)
        self.page.keyboard.insert_text(text)
        self.logger.info(f"ChatGPT: text inserted ({len(text)} chars)")
        if image_paths:
            self._upload_images(image_paths, product_id)
        self._wait_until_sendable(product_id)
        self.page.locator(SEND_BUTTON_SELECTOR).first.click(timeout=10000)
        self.logger.info("ChatGPT: sent, waiting for reply")
        return self._wait_for_reply(previous, product_id)

    def _upload_images(self, image_paths: list[str], product_id: str) -> None:
        files = [str(Path(path)) for path in image_paths]
        target = None
        # The upload inputs mount a moment after the composer becomes usable.
        input_deadline = time.monotonic() + 20
        while target is None and time.monotonic() < input_deadline:
            for selector in FILE_INPUT_SELECTORS:
                inputs = self.page.locator(selector)
                if inputs.count():
                    target = inputs.first
                    break
            else:
                self.page.wait_for_timeout(500)
        if target is not None:
            if target.get_attribute("multiple") is not None:
                target.set_input_files(files)
            else:
                for file in files:
                    target.set_input_files(file)
                    self.page.wait_for_timeout(1000)
        else:
            self._save_debug_snapshot(product_id, "upload-input-missing")
            raise ChatGPTReplyError("ChatGPT 页面上找不到图片上传入口，页面可能改版了。")
        deadline = time.monotonic() + self.upload_timeout
        while time.monotonic() < deadline:
            state = self._state()
            if int(state.get("attachments", 0) or 0) >= len(image_paths) and not state.get("uploading"):
                self.logger.info(f"ChatGPT: uploaded {len(image_paths)} image(s)")
                return
            self.page.wait_for_timeout(1000)
        self._save_debug_snapshot(product_id, "upload-timeout")
        raise ChatGPTReplyError(f"ChatGPT 图片上传在 {int(self.upload_timeout)} 秒内没有完成。")

    def _wait_until_sendable(self, product_id: str) -> None:
        deadline = time.monotonic() + self.upload_timeout
        while time.monotonic() < deadline:
            state = self._state()
            if state.get("send_ready") and not state.get("uploading"):
                return
            self.page.wait_for_timeout(500)
        self._save_debug_snapshot(product_id, "send-disabled")
        raise ChatGPTReplyError("ChatGPT 发送按钮一直不可用，消息没有发出。")

    def _wait_for_reply(self, previous_count: int, product_id: str) -> str:
        ui_mode = os.environ.get("UI_MODE") == "1"
        start = time.monotonic()
        last_report = 0
        last_text = ""
        stable = 0
        while time.monotonic() - start < self.reply_timeout:
            self.page.wait_for_timeout(1000)
            state = self._state()
            elapsed = int(time.monotonic() - start)
            if elapsed // 15 > last_report:
                last_report = elapsed // 15
                message = f"ChatGPT: 正在生成回复（已 {last_report * 15} 秒）"
                if ui_mode:
                    print(f"  {message}", flush=True)
                else:
                    self.logger.info(message)
            if int(state.get("assistant_count", 0) or 0) <= previous_count or state.get("generating"):
                stable = 0
                continue
            text = str(state.get("last_text") or "").strip()
            # The stop button can vanish a moment before the last tokens render.
            stable = stable + 1 if text and text == last_text else 0
            last_text = text
            if stable >= 2:
                return text
        self._save_debug_snapshot(product_id, "reply-timeout")
        raise ChatGPTReplyError(f"ChatGPT 在 {int(self.reply_timeout)} 秒内没有完成回复。")

    def _save_debug_snapshot(self, product_id: str, kind: str) -> None:
        if not self.run_dir:
            return
        try:
            target = self.run_dir / "chatgpt_debug"
            target.mkdir(parents=True, exist_ok=True)
            stem = f"{product_id}-{kind}-{int(time.time())}"
            self.page.screenshot(path=str(target / f"{stem}.png"), full_page=False)
            self.logger.warning(f"ChatGPT: diagnostic screenshot saved to {target / (stem + '.png')}")
        except Exception:
            self.logger.warning("ChatGPT: diagnostic screenshot was unavailable")
