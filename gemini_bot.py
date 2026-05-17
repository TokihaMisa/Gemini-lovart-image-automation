import json
import time
from pathlib import Path

from playwright.sync_api import Page

from utils import (
    build_design_prompt,
    build_lovart_confirmation_prompt,
    parse_lovart_confirmation_decision,
    product_output_dir,
    sanitize_filename,
    update_status,
)


class GeminiBot:
    def __init__(self, page: Page, config: dict, logger, run_dir: str | Path | None = None):
        self.page = page
        self.cfg = config["gemini"]
        self.logger = logger
        self.run_dir = Path(run_dir) if run_dir else None

    def generate_prompt(
        self,
        product_name_cn: str,
        language: str,
        selling_points: str,
        image_paths: list[str],
        product_id: str | None = None,
    ) -> str:
        product_id = product_id or product_name_cn
        self.logger.info(f"Gemini: starting for '{product_name_cn}'")

        try:
            self.page.goto("https://gemini.google.com/app", wait_until="domcontentloaded")
            self.page.wait_for_timeout(4000)
            self._start_temporary_chat()
            if self.cfg.get("thinking_mode", True) and not self._select_thinking_mode():
                self._save_debug_snapshot(product_id, "thinking-mode-not-selected")
                raise RuntimeError("Gemini Thinking mode could not be selected")

            preamble = Path("preamble.txt").read_text(encoding="utf-8")
            previous_response_count = self._response_count()
            self._send_message(preamble)
            self.logger.info("Gemini: preamble sent, waiting for reply")
            self._wait_for_reply(
                previous_response_count=previous_response_count,
                require_design_keywords=False,
            )

            if image_paths and not self._upload_images(image_paths):
                self._save_debug_snapshot(product_id, "image-upload-failed")
                raise RuntimeError("Gemini image upload did not complete")

            prompt = build_design_prompt(product_name_cn, language, selling_points)
            previous_response_count = self._response_count()
            self._send_message(prompt)
            self.logger.info("Gemini: product prompt sent, waiting for reply")
            self._wait_for_reply(previous_response_count=previous_response_count)

            result = self._get_last_response()
            if len(result) < 200:
                self._save_debug_snapshot(product_id, "short-response")
            self.logger.info(f"Gemini: got response ({len(result)} chars)")

            out_dir = product_output_dir(product_id)
            (out_dir / "gemini_prompt.txt").write_text(result, encoding="utf-8")
            update_status(out_dir, "gemini_done", gemini_chars=len(result))
            return result
        except Exception:
            self._save_debug_snapshot(product_id, "exception")
            raise

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
        previous_response_count = self._response_count()
        self._send_message(prompt)
        self.logger.info(f"Gemini: Lovart confirmation decision requested round {round_index}")
        self._wait_for_reply(
            previous_response_count=previous_response_count,
            require_design_keywords=False,
        )
        response = self._get_last_response()
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

    def _start_temporary_chat(self) -> bool:
        """Open Gemini temporary chat/session when the UI exposes that control."""
        selectors = [
            'button[aria-label*="临时"]',
            '[role="button"][aria-label*="临时"]',
            'button:has-text("临时")',
            '[role="button"]:has-text("临时")',
            'button[aria-label*="Temporary"]',
            '[role="button"][aria-label*="Temporary"]',
            'button:has-text("Temporary")',
            '[role="button"]:has-text("Temporary")',
        ]
        for selector in selectors:
            try:
                control = self.page.locator(selector).first
                if control.is_visible(timeout=1500):
                    control.click(timeout=3000)
                    self.page.wait_for_timeout(1500)
                    self.logger.info(f"Gemini: temporary chat clicked via {selector}")
                    return True
            except Exception:
                continue

        script = """
        () => {
            const patterns = [/临时/, /temporary/i];
            const candidates = [...document.querySelectorAll('button, [role="button"], a')];
            const visible = (el) => {
                const rect = el.getBoundingClientRect();
                const style = getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
            };
            const matches = candidates
                .filter(visible)
                .map((el) => ({
                    el,
                    text: [
                        el.innerText,
                        el.getAttribute('aria-label'),
                        el.getAttribute('title'),
                        el.getAttribute('data-tooltip'),
                    ].filter(Boolean).join(' '),
                    rect: el.getBoundingClientRect(),
                }))
                .filter((item) => patterns.some((pattern) => pattern.test(item.text)));
            matches.sort((a, b) => (a.rect.top - b.rect.top) || (b.rect.left - a.rect.left));
            if (!matches.length) return null;
            matches[0].el.click();
            return matches[0].text;
        }
        """
        try:
            clicked = self.page.evaluate(script)
            if clicked:
                self.page.wait_for_timeout(1500)
                self.logger.info("Gemini: temporary chat clicked via DOM scan")
                return True
        except Exception as exc:
            self.logger.warning(f"Gemini: temporary chat DOM scan failed: {exc}")

        self.logger.warning("Gemini: temporary chat control not found; continuing in current chat")
        return False

    def _select_thinking_mode(self) -> bool:
        """Select Gemini Thinking mode before sending prompts."""
        if self._click_thinking_control():
            self.page.wait_for_timeout(1000)
            self.logger.info("Gemini: Thinking mode selected")
            return True

        if self._open_mode_menu():
            self.page.wait_for_timeout(500)
            if self._click_thinking_control():
                self.page.wait_for_timeout(1000)
                self.logger.info("Gemini: Thinking mode selected from mode menu")
                return True

        self.logger.warning("Gemini: Thinking mode control not found")
        return False

    def _click_thinking_control(self) -> bool:
        selectors = [
            'button[aria-label*="思考"]',
            '[role="button"][aria-label*="思考"]',
            '[role="menuitem"]:has-text("思考")',
            'button:has-text("思考")',
            '[role="button"]:has-text("思考")',
            'mat-option:has-text("思考")',
            'button[aria-label*="Thinking"]',
            '[role="button"][aria-label*="Thinking"]',
            '[role="menuitem"]:has-text("Thinking")',
            'button:has-text("Thinking")',
            '[role="button"]:has-text("Thinking")',
            'mat-option:has-text("Thinking")',
        ]
        for selector in selectors:
            try:
                control = self.page.locator(selector).first
                if control.is_visible(timeout=1200):
                    control.click(timeout=3000)
                    self.logger.info(f"Gemini: clicked Thinking mode via {selector}")
                    return True
            except Exception:
                continue

        script = """
        () => {
            const include = [/思考模式/, /^思考$/, /thinking mode/i, /^thinking$/i];
            const exclude = [/显示思路/, /show thinking/i, /思路/, /thought/i];
            const candidates = [...document.querySelectorAll('button, [role="button"], [role="menuitem"], mat-option, [aria-label], [title]')];
            const visible = (el) => {
                const rect = el.getBoundingClientRect();
                const style = getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
            };
            const matches = candidates
                .filter(visible)
                .map((el) => ({
                    el,
                    text: [
                        el.innerText,
                        el.getAttribute('aria-label'),
                        el.getAttribute('title'),
                        el.getAttribute('data-tooltip'),
                    ].filter(Boolean).join(' ').trim(),
                    rect: el.getBoundingClientRect(),
                }))
                .filter((item) => item.text)
                .filter((item) => include.some((pattern) => pattern.test(item.text)))
                .filter((item) => !exclude.some((pattern) => pattern.test(item.text)));
            matches.sort((a, b) => (b.rect.top - a.rect.top) || (b.rect.right - a.rect.right));
            if (!matches.length) return null;
            matches[0].el.click();
            return matches[0].text;
        }
        """
        try:
            clicked = self.page.evaluate(script)
            if clicked:
                self.logger.info("Gemini: clicked Thinking mode via DOM scan")
                return True
        except Exception as exc:
            self.logger.warning(f"Gemini: Thinking mode DOM scan failed: {exc}")
        return False

    def _open_mode_menu(self) -> bool:
        selectors = [
            'button[aria-label*="快速"]',
            '[role="button"][aria-label*="快速"]',
            'button:has-text("快速")',
            '[role="button"]:has-text("快速")',
            'button[aria-label*="Fast"]',
            '[role="button"][aria-label*="Fast"]',
            'button:has-text("Fast")',
            '[role="button"]:has-text("Fast")',
            'button[aria-label*="mode"]',
            '[role="button"][aria-label*="mode"]',
            'button[aria-label*="模式"]',
            '[role="button"][aria-label*="模式"]',
        ]
        for selector in selectors:
            try:
                control = self.page.locator(selector).first
                if control.is_visible(timeout=1200):
                    control.click(timeout=3000)
                    self.logger.info(f"Gemini: opened mode menu via {selector}")
                    return True
            except Exception:
                continue

        script = """
        () => {
            const patterns = [/快速模式/, /^快速$/, /fast mode/i, /^fast$/i, /mode/i, /模式/];
            const exclude = [/fast forward/i, /快速生成图片/];
            const candidates = [...document.querySelectorAll('button, [role="button"], [aria-label], [title]')];
            const visible = (el) => {
                const rect = el.getBoundingClientRect();
                const style = getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
            };
            const matches = candidates
                .filter(visible)
                .map((el) => ({
                    el,
                    text: [
                        el.innerText,
                        el.getAttribute('aria-label'),
                        el.getAttribute('title'),
                        el.getAttribute('data-tooltip'),
                    ].filter(Boolean).join(' ').trim(),
                    rect: el.getBoundingClientRect(),
                }))
                .filter((item) => item.text)
                .filter((item) => patterns.some((pattern) => pattern.test(item.text)))
                .filter((item) => !exclude.some((pattern) => pattern.test(item.text)));
            matches.sort((a, b) => (b.rect.top - a.rect.top) || (b.rect.right - a.rect.right));
            if (!matches.length) return null;
            matches[0].el.click();
            return matches[0].text;
        }
        """
        try:
            clicked = self.page.evaluate(script)
            if clicked:
                self.logger.info("Gemini: opened mode menu via DOM scan")
                return True
        except Exception as exc:
            self.logger.warning(f"Gemini: mode menu DOM scan failed: {exc}")
        return False

    def _upload_images(self, image_paths: list[str]) -> bool:
        """Upload images by clicking the add button, then setting the file input."""
        if not image_paths:
            return False

        clicked = False
        for selector in [
            'button:has(img[alt="add_2"])',
            'button[aria-label*="上传"]',
            'button[aria-label*="Upload"]',
            'button[aria-label*="upload"]',
            'button[aria-label*="Open file"]',
            'button[aria-label*="attach"]',
        ]:
            try:
                locator = self.page.locator(selector)
                if locator.count() > 0:
                    locator.first.click(timeout=3000)
                    self.page.wait_for_timeout(1000)
                    self.logger.info(f"Gemini: clicked add button via {selector}")
                    clicked = True
                    break
            except Exception:
                continue

        if not clicked:
            self.logger.warning("Gemini: add/upload button not found")
            return False

        file_inputs = self.page.locator('input[type="file"]')
        if file_inputs.count() > 0:
            try:
                file_inputs.first.set_input_files(image_paths)
                if self._wait_for_uploads_complete(len(image_paths)):
                    self.logger.info(f"Gemini: uploaded {len(image_paths)} image(s)")
                    return True
            except Exception as exc:
                self.logger.warning(f"Gemini: file input upload failed: {exc}")

        for selector in [
            'li:has-text("Upload")',
            'li:has-text("上传")',
            'div[role="menuitem"]:has-text("file")',
            'button:has-text("上传文件")',
        ]:
            try:
                item = self.page.locator(selector).first
                if item.is_visible(timeout=2000):
                    with self.page.expect_file_chooser(timeout=5000) as chooser:
                        item.click()
                    chooser.value.set_files(image_paths)
                    if self._wait_for_uploads_complete(len(image_paths)):
                        self.logger.info(f"Gemini: uploaded {len(image_paths)} image(s) via menu")
                        return True
            except Exception:
                continue

        self.logger.warning("Gemini: all upload methods failed")
        return False

    def _wait_for_uploads_complete(self, expected_count: int) -> bool:
        timeout_ms = self.cfg.get("upload_timeout", 120) * 1000
        deadline = time.time() + timeout_ms / 1000
        stable_ready = 0

        while time.time() < deadline:
            self.page.wait_for_timeout(1000)
            try:
                state = self.page.evaluate(
                    """
                    () => {
                        const text = document.body.innerText || '';
                        const busy = /(上传中|正在上传|处理中|uploading|processing|attaching)/i.test(text);
                        const fileInputs = [...document.querySelectorAll('input[type="file"]')];
                        const fileCount = fileInputs.reduce((total, input) => total + (input.files ? input.files.length : 0), 0);
                        const attachments = [...document.querySelectorAll('img, video, [aria-label], [data-test-id], mat-chip, .chip')]
                            .filter((el) => {
                                const label = [
                                    el.getAttribute('aria-label'),
                                    el.getAttribute('data-test-id'),
                                    el.getAttribute('title'),
                                    el.innerText,
                                ].filter(Boolean).join(' ');
                                return /(image|photo|picture|uploaded|attachment|图片|照片|附件|已上传)/i.test(label);
                            }).length;
                        const sendButtons = [...document.querySelectorAll('button[aria-label*="Send"], button[aria-label*="发送"], button[aria-label*="submit"]')];
                        const sendDisabled = sendButtons.some((button) => button.disabled || button.getAttribute('aria-disabled') === 'true');
                        return { busy, fileCount, attachments, sendDisabled };
                    }
                    """
                ) or {}
                has_file_signal = state.get("fileCount", 0) >= expected_count
                has_attachment_signal = state.get("attachments", 0) >= expected_count
                ready = (has_file_signal or has_attachment_signal) and not state.get("busy") and not state.get("sendDisabled")
                if ready:
                    stable_ready += 1
                    if stable_ready >= 2:
                        return True
                else:
                    stable_ready = 0
            except Exception:
                stable_ready += 1
                if stable_ready >= 5:
                    self.logger.warning("Gemini: upload state could not be verified; waited before continuing")
                    return True

        self.logger.warning("Gemini: image upload wait timed out")
        return False

    def _send_message(self, text: str) -> None:
        for selector in ['div[contenteditable]', 'rich-textarea [contenteditable]', 'textarea', '[role="textbox"]']:
            try:
                box = self.page.locator(selector).first
                if box.is_visible(timeout=2000):
                    box.click()
                    self.page.wait_for_timeout(300)
                    break
            except Exception:
                continue

        self.page.keyboard.insert_text(text)
        self.page.wait_for_timeout(800)
        self.logger.info(f"Gemini: text inserted ({len(text)} chars)")

        for selector in [
            'button[aria-label*="Send"]',
            'button[aria-label*="发送"]',
            'button[aria-label*="submit"]',
        ]:
            try:
                self.page.locator(selector).first.click(timeout=3000)
                self.logger.info("Gemini: clicked Send")
                break
            except Exception:
                continue
        else:
            self.page.keyboard.press("Enter")

        self.page.wait_for_timeout(1000)

    def _wait_for_reply(self, previous_response_count: int | None = None, require_design_keywords: bool = True) -> None:
        timeout_ms = self.cfg.get("reply_timeout", 300) * 1000
        deadline = time.time() + timeout_ms / 1000
        start = time.time()
        seen_generation = False
        seen_new_response = previous_response_count is None
        stable_done = 0
        progress_printed = False

        while time.time() < deadline:
            self.page.wait_for_timeout(1000)
            try:
                state = self._read_generation_state()
                response_count = int(state.get("response_count", 0) or 0)
                generating = bool(state.get("generating"))
                elapsed = int(time.time() - start)
                dots = "." * ((elapsed % 4) + 1)

                if previous_response_count is not None and response_count > previous_response_count:
                    seen_new_response = True

                if generating:
                    seen_generation = True
                    stable_done = 0
                    print(f"\r  Gemini: reply generation still running{dots}   ", end="", flush=True)
                    progress_printed = True
                    continue

                can_finish = seen_generation or seen_new_response
                if can_finish:
                    stable_done += 1
                else:
                    stable_done = 0

                if stable_done >= 3:
                    if progress_printed:
                        print("\r  Gemini: reply complete.                 ")
                    self.logger.info("Gemini: reply complete")
                    return
                print(f"\r  Gemini: waiting for reply{dots}   ", end="", flush=True)
                progress_printed = True
                self.logger.debug(
                    f"Gemini: waiting reply state generating={generating}, "
                    f"seen_generation={seen_generation}, seen_new_response={seen_new_response}, "
                    f"stable_done={stable_done}"
                )
            except Exception:
                pass

        if progress_printed:
            print()
        self.logger.warning("Gemini: reply wait timed out")

    def _read_generation_state(self) -> dict:
        script = """
        () => {
            const visible = (el) => {
                if (!el) return false;
                const rect = el.getBoundingClientRect();
                const style = getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
            };
            const labelOf = (el) => [
                el.innerText,
                el.getAttribute('aria-label'),
                el.getAttribute('title'),
                el.getAttribute('data-tooltip'),
            ].filter(Boolean).join(' ');

            const controls = [...document.querySelectorAll('button, [role="button"], [aria-label], [title]')].filter(visible);
            const hasStop = controls.some((el) => /(停止|中止|取消|stop|cancel)/i.test(labelOf(el)));
            const progress = [...document.querySelectorAll('[role="progressbar"], mat-progress-spinner, mat-spinner, .spinner, .loading')].some(visible);
            const busy = [...document.querySelectorAll('[aria-busy="true"], [data-loading="true"]')].some(visible);
            const responseCount = document.querySelectorAll('model-response, .model-response, article').length;
            return {
                generating: hasStop || progress || busy,
                has_stop: hasStop,
                progress,
                busy,
                response_count: responseCount,
            };
        }
        """
        try:
            state = self.page.evaluate(script) or {}
            if not isinstance(state, dict):
                return {"generating": False, "response_count": self._response_count()}
            return state
        except Exception:
            return {"generating": False, "response_count": self._response_count()}

    def _response_count(self) -> int:
        try:
            return self.page.locator("model-response, .model-response, article").count()
        except Exception:
            return 0

    def _latest_response_text(self) -> str:
        script = """
        () => {
            const responses = document.querySelectorAll('model-response, .model-response');
            if (!responses.length) return '';
            return responses[responses.length - 1].innerText || '';
        }
        """
        try:
            return str(self.page.evaluate(script) or "")
        except Exception:
            return ""

    def _get_last_response(self) -> str:
        script = """
        () => {
            const responses = document.querySelectorAll('model-response, .model-response');
            if (responses.length > 0) return responses[responses.length - 1].innerText;
            const articles = document.querySelectorAll('article');
            let best = '';
            for (const article of articles) {
                const text = article.innerText || '';
                if (text.length > best.length) best = text;
            }
            return best || null;
        }
        """
        try:
            text = self.page.evaluate(script)
            if text and len(text) > 200:
                cleaned = self._strip_gemini_chrome(text)
                self.logger.info(f"Gemini: JS extraction ({len(cleaned)} chars)")
                return cleaned.strip()
        except Exception as exc:
            self.logger.warning(f"Gemini: JS extraction failed: {exc}")

        try:
            body = self.page.locator("body").inner_text(timeout=10000)
            for marker in ["主标题", "副标题"]:
                idx = body.rfind(marker)
                if idx > 100:
                    return self._strip_gemini_chrome(body[idx:]).strip()
            return self._strip_gemini_chrome(body).strip()
        except Exception:
            return "(could not extract response)"

    def _save_debug_snapshot(self, product_id: str, label: str) -> None:
        if not self.run_dir:
            return
        debug_dir = self.run_dir / "browser-debug" / sanitize_filename(product_id)
        debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        base = debug_dir / f"{stamp}-{sanitize_filename(label)}"
        try:
            self.page.screenshot(path=str(base.with_suffix(".png")), full_page=True)
        except Exception as exc:
            self.logger.warning(f"Gemini: failed to save screenshot: {exc}")
        try:
            base.with_suffix(".html").write_text(self.page.content(), encoding="utf-8")
        except Exception as exc:
            self.logger.warning(f"Gemini: failed to save HTML snapshot: {exc}")

    @staticmethod
    def _strip_gemini_chrome(text: str) -> str:
        import re

        chrome_patterns = [
            r"^显示思路\s*$",
            r"^Gemini\s+说\s*$",
            r"^分析\s*$",
            r"^Show thinking\s*$",
            r"^Gemini says\s*$",
            r"^Analysis\s*$",
        ]
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if cleaned:
                    cleaned.append("")
                continue
            if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in chrome_patterns):
                continue
            cleaned.append(line)

        while cleaned and not cleaned[0].strip():
            cleaned.pop(0)
        while cleaned and not cleaned[-1].strip():
            cleaned.pop()
        return "\n".join(cleaned)
