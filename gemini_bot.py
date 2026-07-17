import json
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
import unicodedata
import re
import ssl
from urllib.error import HTTPError

from playwright.sync_api import Page

from prompt_settings import get_prompt_settings
from gemini_browser_session import (
    GeminiAuthenticationError,
    GeminiLoginRequiredError,
    GeminiPermanentTlsError,
    GeminiPageNotReadyError,
    GeminiResourceNotFoundError,
    GeminiPageState,
    inspect_gemini_page,
    navigate_gemini_with_retry,
)
from network_retry import RetryKind, classify_network_error, retry_policy_from_config
from utils import (
    build_design_prompt,
    build_lovart_confirmation_prompt,
    parse_lovart_confirmation_decision,
    product_output_dir,
    sanitize_filename,
    update_status,
)


def normalize_ui_text(text: str) -> str:
    """Normalize UI labels across Chinese, English, and accented Spanish."""
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_marks.casefold().split())


def matches_ui_term(text: str, terms) -> bool:
    value = normalize_ui_text(text)
    return any(normalize_ui_text(term) in value for term in terms)


MODE_TERMS = frozenset(("快速", "fast", "flash", "rápido", "modo rápido"))
EXTENDED_THINKING_TERMS = frozenset(("扩展思考", "extended thinking", "pensamiento ampliado"))
UPLOAD_TERMS = frozenset(("上传", "attach", "upload", "adjuntar archivos", "adjuntar"))
TEMPORARY_CHAT_TERMS = frozenset(("临时", "temporary", "chat temporal"))


class GeminiPageStructureError(RuntimeError):
    """Gemini loaded but its expected controls were not present."""


class GeminiUploadIncompleteError(RuntimeError):
    """Image attachment state could not be verified as complete."""


def _safe_origin_path(url: str) -> str:
    parts = urlsplit(str(url or ""))
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return ""
    host = parts.hostname
    if parts.port:
        host = f"{host}:{parts.port}"
    safe_segments = [segment for segment in parts.path.split("/") if segment in {"app", "chat"}]
    path = "/" + "/".join(safe_segments) if safe_segments else "/"
    return urlunsplit((parts.scheme, host, path, "", ""))


def _safe_controls(page: Page) -> list[str]:
    """Collect a small, privacy-safe control summary for diagnostics."""
    try:
        payload = page.evaluate(
            """() => ({ language: document.documentElement.lang || navigator.language || '',
                controls: [...document.querySelectorAll('button, [role=button], [role=menuitem], [aria-label], [title]')]
                  .filter((node) => { const rect = node.getBoundingClientRect(); const style = getComputedStyle(node); return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none'; })
                  .map((node) => [node.innerText, node.getAttribute('aria-label'), node.getAttribute('title'), node.getAttribute('data-tooltip')].filter(Boolean).join(' ').trim())
                  .filter(Boolean).slice(0, 40) })"""
        ) or {}
    except Exception:
        payload = {}
    controls = payload.get("controls", []) if isinstance(payload, dict) else []
    categories = (
        (TEMPORARY_CHAT_TERMS, "temporary_chat"),
        (EXTENDED_THINKING_TERMS, "extended_thinking"),
        (UPLOAD_TERMS, "upload"),
        (MODE_TERMS, "mode"),
    )
    unique: list[str] = []
    for item in controls if isinstance(controls, list) else []:
        text = " ".join(str(item).split())[:160]
        category = next((label for terms, label in categories if matches_ui_term(text, terms)), None)
        # Never persist raw control text: unknown values are intentionally omitted.
        if not category or category in unique:
            continue
        unique.append(category)
        if len(unique) >= 20:
            break
    return unique


def _safe_language(value: object) -> str:
    language = str(value or "")
    return language if re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8}){0,2}", language) else "unknown"


def save_gemini_diagnostics(
    page: Page,
    run_dir: str | Path,
    product_id: str,
    reason: str,
    attempts: int,
    error_kind: str,
) -> Path:
    """Persist only bounded, redacted metadata; debug artifacts stay best-effort."""
    debug_dir = Path(run_dir) / "browser-debug" / sanitize_filename(product_id)
    debug_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.time_ns()
    # Do not turn a raw exception (which can include addresses or URLs) into a filename.
    safe_label = "diagnostic"
    base = debug_dir / f"{stamp}-{safe_label}"
    # A screenshot can expose a signed-in account or page content. Keep a
    # safe placeholder rather than persisting unredactable browser pixels.
    from PIL import Image
    Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(base.with_suffix(".png"), format="PNG")
    # DOM content can contain account data and cannot be safely reconstructed
    # after the fact. Retain a deliberately content-free HTML diagnostic.
    base.with_suffix(".html").write_text(
        "<html><body>Gemini diagnostic content redacted.</body></html>", encoding="utf-8"
    )
    try:
        inspected = page.evaluate("() => ({ language: document.documentElement.lang || navigator.language || '' })") or {}
    except Exception:
        inspected = {}
    metadata = {
        "url": _safe_origin_path(getattr(page, "url", "")),
        "language": _safe_language(inspected.get("language", "") if isinstance(inspected, dict) else ""),
        "controls": _safe_controls(page),
        "attempts": max(1, int(attempts)),
        "error_kind": (
            normalize_ui_text(error_kind)
            if normalize_ui_text(error_kind) in {"transient", "permanent_tls", "auth", "not_found", "other", "login_required", "page_structure"}
            else "other"
        ),
    }
    result = base.with_suffix(".json")
    result.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


class GeminiBot:
    def __init__(self, page: Page, config: dict, logger, run_dir: str | Path | None = None):
        self.page = page
        self.cfg = config.get("gemini", {})
        self._browser_config = config.get("browser", {})
        self.logger = logger
        self.run_dir = Path(run_dir) if run_dir else None
        self.prompt_settings = get_prompt_settings(config)

    @staticmethod
    def _error_kind(error: BaseException) -> str:
        if isinstance(error, GeminiAuthenticationError):
            return "auth"
        if isinstance(error, GeminiResourceNotFoundError):
            return "not_found"
        if isinstance(error, GeminiLoginRequiredError):
            return "login_required"
        if isinstance(error, GeminiPermanentTlsError):
            return "permanent_tls"
        if isinstance(error, GeminiPageStructureError):
            return "page_structure"
        if isinstance(error, HTTPError):
            if error.code in {401, 403}:
                return "auth"
            if error.code == 404:
                return "not_found"
        message = str(error).casefold()
        if "err_access_denied" in message:
            return "auth"
        if "err_file_not_found" in message:
            return "not_found"
        return classify_network_error(error).value

    @staticmethod
    def _is_retryable_product_error(error: BaseException) -> bool:
        if isinstance(error, GeminiPageStructureError):
            return True
        if isinstance(error, HTTPError):
            return error.code in {408, 429} or 500 <= error.code <= 599
        message = str(error).casefold()
        permanent_markers = (
            "err_cert_", "certificate verify", "err_access_denied",
            "err_blocked_by_client", "err_blocked_by_response",
        )
        if any(marker in message for marker in permanent_markers):
            return False
        if isinstance(error, (TimeoutError, ConnectionError)):
            return True
        return any(marker in message for marker in (
            "err_connection_reset", "err_connection_closed", "err_connection_timed_out",
            "err_network_changed", "err_name_not_resolved", "temporary dns",
            "err_ssl_protocol_error", "connection reset", "connection timed out",
        ))

    @staticmethod
    def _safe_terminal_error(error: BaseException) -> BaseException:
        safe_errors = (
            GeminiAuthenticationError,
            GeminiLoginRequiredError,
            GeminiPermanentTlsError,
            GeminiPageNotReadyError,
            GeminiPageStructureError,
            GeminiResourceNotFoundError,
            GeminiUploadIncompleteError,
        )
        if isinstance(error, safe_errors):
            return error
        message = str(error).casefold()
        if isinstance(error, ssl.SSLCertVerificationError) or "err_cert_" in message or "certificate" in message:
            return GeminiPermanentTlsError()
        if isinstance(error, HTTPError):
            if error.code in {401, 403}:
                return GeminiAuthenticationError()
            if error.code == 404:
                return GeminiResourceNotFoundError()
        if "err_access_denied" in message:
            return GeminiAuthenticationError()
        if "err_file_not_found" in message:
            return GeminiResourceNotFoundError()
        return GeminiPageNotReadyError()

    def _select_thinking_mode_with_recovery(self, product_id: str) -> None:
        if self._select_thinking_mode():
            return
        status = inspect_gemini_page(self.page)
        if status.state is GeminiPageState.WAITING_LOGIN:
            raise GeminiLoginRequiredError()
        if status.state is GeminiPageState.ERROR:
            raise GeminiPageNotReadyError()
        if status.state is GeminiPageState.PAGE_LOADING:
            policy = retry_policy_from_config({"browser": self._browser_config})
            status = navigate_gemini_with_retry(
                self.page, "https://gemini.google.com/app", policy, logger=self.logger
            )
            if status.state is GeminiPageState.WAITING_LOGIN:
                raise GeminiLoginRequiredError()
            if status.state is GeminiPageState.ERROR:
                raise GeminiPageNotReadyError()
            if self._select_thinking_mode():
                return
        elif status.state is GeminiPageState.READY and self._select_thinking_mode():
            return
        self._save_debug_snapshot(product_id, "thinking-mode-not-selected", 1, "page_structure")
        raise GeminiPageStructureError("Gemini Thinking mode control is missing on a ready page")

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
        self.logger.info(f"Gemini: starting for '{product_name_cn}'")
        policy = retry_policy_from_config({"browser": self._browser_config})
        product_attempts = min(2, policy.product_attempts)
        last_error: Exception | None = None
        for attempt in range(1, product_attempts + 1):
            try:
                result = self._generate_prompt_once(
                    product_name_cn, language, selling_points, image_paths, product_id, image_size
                )
                out_dir = product_output_dir(product_id)
                (out_dir / "gemini_prompt.txt").write_text(result, encoding="utf-8")
                update_status(out_dir, "gemini_done", gemini_chars=len(result))
                return result
            except Exception as exc:
                last_error = exc
                retryable = self._is_retryable_product_error(exc)
                if not retryable:
                    self._save_debug_snapshot(product_id, "exception", attempt, self._error_kind(exc))
                    safe_error = self._safe_terminal_error(exc)
                    if safe_error is exc:
                        raise
                    raise safe_error from None
                if attempt >= product_attempts:
                    self._save_debug_snapshot(product_id, "exception", attempt, self._error_kind(exc))
                    safe_error = self._safe_terminal_error(exc)
                    if safe_error is exc:
                        raise
                    raise safe_error from None
                self.logger.warning(f"Gemini: retrying product prompt attempt {attempt + 1}/{product_attempts}")
                delay = policy.delay_after(attempt)
                if delay:
                    from network_retry import time as retry_time
                    retry_time.sleep(delay)
        raise last_error or RuntimeError("Gemini product prompt failed")

    def _generate_prompt_once(
        self,
        product_name_cn: str,
        language: str,
        selling_points: str,
        image_paths: list[str],
        product_id: str,
        image_size: str,
    ) -> str:
        try:
            try:
                self.page.goto("https://gemini.google.com/app", wait_until="domcontentloaded")
            except Exception as e:
                if "interrupted by another navigation" in str(e):
                    self.logger.info("Gemini: Navigation interrupted by redirect. Retrying...")
                    self.page.wait_for_timeout(2000)
                    self.page.goto("https://gemini.google.com/app", wait_until="domcontentloaded")
                else:
                    raise
            self.page.wait_for_timeout(4000)
            if not self._start_temporary_chat():
                raise GeminiPageStructureError("Gemini temporary chat control is missing")
            if self.cfg.get("thinking_mode", True):
                self._select_thinking_mode_with_recovery(product_id)

            from utils import get_resource_path
            preamble = get_resource_path("preamble.txt").read_text(encoding="utf-8")
            previous_response_count = self._response_count()
            self._send_message(preamble)
            self.logger.info("Gemini: preamble sent, waiting for reply")
            self._wait_for_reply(
                previous_response_count=previous_response_count,
                require_design_keywords=False,
            )

            if image_paths and not self._upload_images(image_paths):
                raise GeminiUploadIncompleteError("Gemini image upload did not complete")

            prompt = build_design_prompt(
                product_name_cn,
                language,
                selling_points,
                image_size=image_size,
                prompt_settings=self.prompt_settings,
            )
            previous_response_count = self._response_count()
            self._send_message(prompt)
            self.logger.info("Gemini: product prompt sent, waiting for reply")
            self._wait_for_reply(previous_response_count=previous_response_count)

            result = self._get_last_response()
            if len(result) < 200:
                self._save_debug_snapshot(product_id, "short-response")
            self.logger.info(f"Gemini: got response ({len(result)} chars)")

            return result
        except Exception:
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

    def _click_normalized_dom_term(self, terms, label: str) -> bool:
        """Structural selector fallback using normalized visible control attributes."""
        normalized_terms = [normalize_ui_text(term) for term in terms]
        script = rf"""
        () => {{
            const terms = {json.dumps(normalized_terms, ensure_ascii=False)};
            const normalize = (value) => (value || '').normalize('NFKD')
                .replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/\s+/g, ' ').trim();
            const visible = (node) => {{
                const rect = node.getBoundingClientRect(); const style = getComputedStyle(node);
                return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
            }};
            const candidates = [...document.querySelectorAll('button, [role="button"], [role="menuitem"], gem-menu-item, mat-option, [aria-label], [title], [data-tooltip]')]
                .filter(visible)
                .map((node) => ({{ node, text: normalize([node.innerText, node.getAttribute('aria-label'), node.getAttribute('title'), node.getAttribute('data-tooltip')].filter(Boolean).join(' ')) }}))
                .filter((item) => terms.some((term) => item.text.includes(term)));
            if (!candidates.length) return null;
            candidates[0].node.click();
            return candidates[0].text;
        }}
        """
        try:
            if self.page.evaluate(script):
                self.logger.info(f"Gemini: clicked {label} via normalized DOM scan")
                return True
        except Exception:
            self.logger.warning(f"Gemini: {label} normalized DOM scan was unavailable")
        return False

    def _normalized_dom_term_is_selected(self, terms) -> bool:
        normalized_terms = [normalize_ui_text(term) for term in terms]
        script = rf"""
        () => {{
            const terms = {json.dumps(normalized_terms, ensure_ascii=False)};
            const normalize = (value) => (value || '').normalize('NFKD')
                .replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/\s+/g, ' ').trim();
            const visible = (node) => {{ const rect = node.getBoundingClientRect(); const style = getComputedStyle(node); return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none'; }};
            return [...document.querySelectorAll('button, [role="button"], [role="menuitem"], [role="menuitemcheckbox"], [role="menuitemradio"], [aria-label], [title], [data-tooltip]')]
                .filter(visible).some((node) => {{
                    const text = normalize([node.innerText, node.getAttribute('aria-label'), node.getAttribute('title'), node.getAttribute('data-tooltip')].filter(Boolean).join(' '));
                    const selected = [node.getAttribute('aria-checked'), node.getAttribute('aria-selected'), node.getAttribute('data-selected'), node.getAttribute('selected'), node.className].filter(Boolean).join(' ');
                    return terms.some((term) => text.includes(term)) && /true|selected|checked|active/i.test(selected);
                }});
        }}
        """
        try:
            return bool(self.page.evaluate(script))
        except Exception:
            return False

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
            'button[aria-label*="Chat temporal"]',
            '[role="button"][aria-label*="Chat temporal"]',
            'button:has-text("Chat temporal")',
            '[role="button"]:has-text("Chat temporal")',
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

        if self._click_normalized_dom_term(TEMPORARY_CHAT_TERMS, "temporary chat"):
            self.page.wait_for_timeout(1500)
            return True

        script = """
        () => {
            const patterns = [/临时/, /臨時/, /temporary/i, /chat temporal/i];
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
        except Exception:
            self.logger.warning("Gemini: temporary chat DOM scan was unavailable")

        self.logger.warning("Gemini: temporary chat control not found; stopping this attempt")
        return False

    def _select_thinking_mode(self) -> bool:
        """Select Gemini 3 Flash and set thinking level to extended."""
        if not self._open_mode_menu():
            if self._current_model_is_flash() and self._current_thinking_level_is_extended():
                self.logger.info("Gemini: Flash extended thinking mode already selected")
                return True
            self.logger.warning("Gemini: mode menu control not found")
            return False

        self.page.wait_for_timeout(500)
        if not self._click_flash_model():
            self.logger.warning("Gemini: 3 Flash model option not found")
            return False

        self.page.wait_for_timeout(1000)
        if self._select_extended_thinking_option():
            self.logger.info("Gemini: selected Flash with extended thinking")
            return True

        if not self._open_thinking_level_menu():
            if not self._open_mode_menu():
                self.logger.warning("Gemini: mode menu could not be reopened for thinking level")
                return False
            if self._select_extended_thinking_option(menu_is_open=True):
                self.logger.info("Gemini: selected Flash with extended thinking")
                return True
            self.page.wait_for_timeout(500)
            if not self._open_thinking_level_menu():
                self.logger.warning("Gemini: thinking level menu not found")
                return False

        self.page.wait_for_timeout(500)
        if not self._click_extended_thinking_level():
            self.logger.warning("Gemini: extended thinking level option not found")
            return False

        self.page.wait_for_timeout(1000)
        self.logger.info("Gemini: selected 3 Flash with extended thinking level")
        return True

    def _select_extended_thinking_option(self, menu_is_open: bool = False) -> bool:
        """Select the newer top-level Gemini 'extended thinking' menu option."""
        if self._extended_thinking_option_is_checked():
            self.logger.info("Gemini: extended thinking option already selected")
            return True

        if not menu_is_open and not self._open_mode_menu():
            return False

        if self._extended_thinking_option_is_checked():
            self.logger.info("Gemini: extended thinking option already selected")
            return True

        if self._click_extended_thinking_option():
            self.page.wait_for_timeout(800)
            return True
        return False

    def _extended_thinking_option_is_checked(self) -> bool:
        if self._normalized_dom_term_is_selected(EXTENDED_THINKING_TERMS):
            return True
        script = """
        () => {
            const textMatches = (text) => /\\u6269\\u5c55\\u601d\\u8003|extended\\s+thinking/i.test(text || '');
            const visible = (el) => {
                const rect = el.getBoundingClientRect();
                const style = getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
            };
            const textOf = (el) => [
                el.innerText,
                el.textContent,
                el.getAttribute('aria-label'),
                el.getAttribute('title'),
                el.getAttribute('data-tooltip'),
            ].filter(Boolean).join(' ');
            const candidates = [...document.querySelectorAll('button, [role="button"], [role="menuitem"], [role="menuitemcheckbox"], [role="menuitemradio"], gem-menu-item, mat-option')]
                .filter(visible)
                .filter((el) => textMatches(textOf(el)));
            const hasCheck = (el) => {
                const nodes = [el, el.parentElement, el.closest('[role="menuitem"], [role="menuitemcheckbox"], [role="menuitemradio"], gem-menu-item, mat-option')].filter(Boolean);
                return nodes.some((node) => {
                    const attrs = [
                        node.getAttribute('aria-checked'),
                        node.getAttribute('aria-selected'),
                        node.getAttribute('data-selected'),
                        node.getAttribute('selected'),
                    ].filter(Boolean).join(' ');
                    const classes = node.className ? String(node.className) : '';
                    const text = textOf(node);
                    return /true|selected|checked/i.test(attrs)
                        || /selected|checked|active/i.test(classes)
                        || /(^|\\s)[✓✔](\\s|$)/.test(text);
                });
            };
            return candidates.some(hasCheck);
        }
        """
        try:
            return bool(self.page.evaluate(script))
        except Exception:
            return False

    def _click_extended_thinking_option(self) -> bool:
        selectors = [
            'gem-menu-item:has-text("扩展思考")',
            '[role="menuitem"]:has-text("扩展思考")',
            '[role="menuitemcheckbox"]:has-text("扩展思考")',
            '[role="menuitemradio"]:has-text("扩展思考")',
            'button:has-text("扩展思考")',
            '[role="button"]:has-text("扩展思考")',
            'mat-option:has-text("扩展思考")',
            'gem-menu-item:has-text("Extended thinking")',
            '[role="menuitem"]:has-text("Extended thinking")',
            '[role="menuitemcheckbox"]:has-text("Extended thinking")',
            '[role="menuitemradio"]:has-text("Extended thinking")',
            'button:has-text("Extended thinking")',
            '[role="button"]:has-text("Extended thinking")',
            'mat-option:has-text("Extended thinking")',
            'gem-menu-item:has-text("Pensamiento ampliado")',
            '[role="menuitem"]:has-text("Pensamiento ampliado")',
            'button:has-text("Pensamiento ampliado")',
        ]
        if self._click_normalized_dom_term(EXTENDED_THINKING_TERMS, "extended thinking option"):
            return True
        if self._click_gemini_control(
            selectors,
            r'[/\u6269\u5c55\u601d\u8003/, /extended\s+thinking/i, /pensamiento\s+ampliado/i]',
            "extended thinking option",
        ):
            return True
        return self._extended_thinking_option_is_checked()

    def _click_flash_model(self) -> bool:
        selectors = [
            'gem-menu-item:has-text("3.5 Flash")',
            '[role="menuitem"]:has-text("3.5 Flash")',
            'button:has-text("3.5 Flash")',
            '[role="button"]:has-text("3.5 Flash")',
            'gem-menu-item:has-text("3 Flash")',
            '[role="menuitem"]:has-text("3 Flash")',
            'button:has-text("3 Flash")',
            '[role="button"]:has-text("3 Flash")',
            'gem-menu-item:has-text("全方位帮助")',
            '[role="menuitem"]:has-text("全方位帮助")',
        ]
        if self._click_gemini_control(
            selectors,
            r'[/^3(?:\.\d+)?\s*Flash\b/i, /Flash[\s\S]*全方位帮助/i]',
            "Flash model",
            exclude_list=r'[/Flash-Lite/i, /Lite/i, /极速回答/]',
        ):
            return True
        return self._current_model_is_flash()

    def _open_thinking_level_menu(self) -> bool:
        selectors = [
            'gem-menu-item:has-text("思考等级")',
            '[role="menuitem"]:has-text("思考等级")',
            'button:has-text("思考等级")',
            '[role="button"]:has-text("思考等级")',
            'gem-menu-item:has-text("Thinking level")',
            '[role="menuitem"]:has-text("Thinking level")',
            'button:has-text("Thinking level")',
            '[role="button"]:has-text("Thinking level")',
            'gem-menu-item:has-text("Nivel de pensamiento")',
            '[role="menuitem"]:has-text("Nivel de pensamiento")',
            'button:has-text("Nivel de pensamiento")',
        ]
        if self._click_normalized_dom_term(("思考等级", "thinking level", "nivel de pensamiento"), "thinking level menu"):
            return True
        return self._click_gemini_control(selectors, r'[/思考等级/, /thinking level/i, /nivel de pensamiento/i]', "thinking level menu")

    def _click_extended_thinking_level(self) -> bool:
        selectors = [
            'gem-menu-item:has-text("扩展")',
            '[role="menuitem"]:has-text("扩展")',
            'button:has-text("扩展")',
            '[role="button"]:has-text("扩展")',
            'gem-menu-item:has-text("Extended")',
            '[role="menuitem"]:has-text("Extended")',
            'button:has-text("Extended")',
            '[role="button"]:has-text("Extended")',
        ]
        if self._click_normalized_dom_term(("扩展", "extended", "ampliado"), "extended thinking level"):
            return True
        return self._click_gemini_control(selectors, r'[/^扩展\b/, /^Extended\b/i]', "extended thinking level")

    def _click_thinking_control(self) -> bool:
        selectors = [
            'button[aria-label*="思考"]',
            '[role="button"][aria-label*="思考"]',
            '[role="menuitem"]:has-text("思考")',
            'gem-menu-item:has-text("思考")',
            'button:has-text("思考")',
            '[role="button"]:has-text("思考")',
            'mat-option:has-text("思考")',
            'button[aria-label*="Thinking"]',
            '[role="button"][aria-label*="Thinking"]',
            '[role="menuitem"]:has-text("Thinking")',
            'gem-menu-item:has-text("Thinking")',
            'button:has-text("Thinking")',
            '[role="button"]:has-text("Thinking")',
            'mat-option:has-text("Thinking")',
            '[role="menuitem"]:has-text("Deep Think")',
            'gem-menu-item:has-text("Deep Think")',
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

        if self._click_normalized_dom_term(("思考", "thinking", "pensamiento"), "thinking mode"):
            return True

        script = """
        () => {
            const include = [/思考模式/, /^思考$/, /思考等级/, /thinking mode/i, /^thinking$/i, /deep think/i, /pensamiento/i];
            const exclude = [/显示思路/, /显示思考/, /show thinking/i, /思路/, /thought/i];
            const candidates = [...document.querySelectorAll('button, [role="button"], [role="menuitem"], gem-menu-item, mat-option, [aria-label], [title]')];
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
        except Exception:
            self.logger.warning("Gemini: Thinking mode DOM scan was unavailable")
        return False

    def _current_model_is_flash_legacy(self) -> bool:
        script = """
        () => {
            const controls = [...document.querySelectorAll('button[aria-label*="打开模式选择器"], [role="button"][aria-label*="打开模式选择器"]')];
            const visible = (el) => {
                const rect = el.getBoundingClientRect();
                const style = getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
            };
            return controls
                .filter(visible)
                .some((el) => /Flash/i.test(el.innerText || el.textContent || '') && !/Lite/i.test(el.innerText || el.textContent || ''));
        }
        """
        try:
            if self.page.evaluate(script):
                self.logger.info("Gemini: Flash model already selected")
                return True
        except Exception:
            pass
        return False

    def _current_model_is_flash(self) -> bool:
        script = """
        () => {
            const controls = [...document.querySelectorAll('button, [role="button"], [aria-label], [title]')];
            const visible = (el) => {
                const rect = el.getBoundingClientRect();
                const style = getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
            };
            return controls
                .filter(visible)
                .some((el) => {
                    const text = [
                        el.innerText,
                        el.textContent,
                        el.getAttribute('aria-label'),
                        el.getAttribute('title'),
                    ].filter(Boolean).join(' ');
                    return /Flash/i.test(text) && !/Lite/i.test(text);
                });
        }
        """
        try:
            if self.page.evaluate(script):
                self.logger.info("Gemini: Flash model already selected")
                return True
        except Exception:
            pass
        return False

    def _current_thinking_level_is_extended(self) -> bool:
        script = """
        () => {
            const controls = [...document.querySelectorAll('button, [role="button"], [aria-label], [title]')];
            const visible = (el) => {
                const rect = el.getBoundingClientRect();
                const style = getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
            };
            return controls
                .filter(visible)
                .some((el) => {
                    const text = [
                        el.innerText,
                        el.textContent,
                        el.getAttribute('aria-label'),
                        el.getAttribute('title'),
                    ].filter(Boolean).join(' ');
                    return /Extended/i.test(text) || /\\u6269\\u5c55/.test(text);
                });
        }
        """
        try:
            if self.page.evaluate(script):
                self.logger.info("Gemini: extended thinking level already selected")
                return True
        except Exception:
            pass
        return False

    def _click_gemini_control(
        self,
        selectors: list[str],
        pattern_list: str,
        label: str,
        exclude_list: str | None = None,
    ) -> bool:
        for selector in selectors:
            try:
                control = self.page.locator(selector).first
                if control.is_visible(timeout=1200):
                    try:
                        control.click(timeout=3000, force=True)
                    except TypeError:
                        control.click(timeout=3000)
                    self.logger.info(f"Gemini: clicked {label} via {selector}")
                    return True
            except Exception:
                continue

        exclude_line = ""
        exclude_filter = ""
        if exclude_list:
            exclude_line = f"const exclude = {exclude_list};"
            exclude_filter = ".filter((item) => !exclude.some((pattern) => pattern.test(item.text)))"

        script = f"""
        () => {{
            const include = {pattern_list};
            {exclude_line}
            const candidates = [...document.querySelectorAll('button, [role="button"], [role="menuitem"], gem-menu-item, mat-option, [aria-label], [title]')];
            const visible = (el) => {{
                const rect = el.getBoundingClientRect();
                const style = getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
            }};
            const matches = candidates
                .filter(visible)
                .map((el) => ({{
                    el,
                    text: [
                        el.innerText,
                        el.getAttribute('aria-label'),
                        el.getAttribute('title'),
                        el.getAttribute('data-tooltip'),
                    ].filter(Boolean).join(' ').trim(),
                    rect: el.getBoundingClientRect(),
                }}))
                .filter((item) => item.text)
                .filter((item) => include.some((pattern) => pattern.test(item.text)))
                {exclude_filter};
            matches.sort((a, b) => (b.rect.top - a.rect.top) || (b.rect.right - a.rect.right));
            if (!matches.length) return null;
            matches[0].el.click();
            return matches[0].text;
        }}
        """
        try:
            clicked = self.page.evaluate(script)
            if clicked:
                self.logger.info(f"Gemini: clicked {label} via DOM scan")
                return True
        except Exception:
            self.logger.warning(f"Gemini: {label} DOM scan was unavailable")
        return False

    def _open_mode_menu(self) -> bool:
        selectors = [
            'button[aria-label*="打开模式选择器"]',
            '[role="button"][aria-label*="打开模式选择器"]',
            'button[aria-label*="快速"]',
            '[role="button"][aria-label*="快速"]',
            'button:has-text("快速")',
            '[role="button"]:has-text("快速")',
            'button:has-text("Flash")',
            '[role="button"]:has-text("Flash")',
            'button:has-text("Pro")',
            '[role="button"]:has-text("Pro")',
            'button[aria-label*="Fast"]',
            '[role="button"][aria-label*="Fast"]',
            'button:has-text("Fast")',
            '[role="button"]:has-text("Fast")',
            'button[aria-label*="Rápido"]',
            '[role="button"][aria-label*="Rápido"]',
            'button:has-text("Rápido")',
            '[role="button"]:has-text("Rápido")',
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

        if self._click_normalized_dom_term(MODE_TERMS, "mode menu"):
            return True

        script = """
        () => {
            const patterns = [/打开模式选择器/, /快速模式/, /^快速$/, /fast mode/i, /^fast$/i, /^flash$/i, /mode/i, /模式/, /^pro$/i, /^rapido$/i, /modo rapido/i];
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
        except Exception:
            self.logger.warning("Gemini: mode menu DOM scan was unavailable")
        return False

    def _upload_images(self, image_paths: list[str]) -> bool:
        """Upload images by clicking the add button, then setting the file input."""
        if not image_paths:
            return False

        attempts = max(1, int(self.cfg.get("upload_attempts", 3) or 3))
        for attempt in range(1, attempts + 1):
            if self._upload_images_once(image_paths):
                return True
            if attempt < attempts:
                self.logger.warning(f"Gemini: image upload attempt {attempt}/{attempts} failed; retrying")
                self.page.wait_for_timeout(2000)

        self.logger.warning("Gemini: all upload attempts failed")
        return False

    def _upload_images_once(self, image_paths: list[str]) -> bool:
        # Strategy 1: Direct set_input_files on any file input on the page.
        # Playwright can interact with hidden file inputs directly, avoiding UI clicks entirely.
        try:
            file_inputs = self.page.locator('input[type="file"]')
            if file_inputs.count() > 0:
                # Often the last one is the chat box's file input, but we can try the first visible/enabled one, or just the last.
                file_inputs.last.set_input_files(image_paths)
                if self._wait_for_uploads_complete(len(image_paths)):
                    self.logger.info(f"Gemini: uploaded {len(image_paths)} image(s) via direct input")
                    return True
        except Exception:
            self.logger.warning("Gemini: direct file input upload was unavailable")

        # Strategy 2: Click the add/upload button robustly, then expect a file chooser
        clicked = False
        selectors = [
            'button:has(img[alt="add_2"])',
            'button[aria-label*="上传和工具"]',
            'button[aria-label*="上传"]',
            'button[aria-label*="添加"]',
            'button[aria-label*="附件"]',
            'button[aria-label*="Upload"]',
            'button[aria-label*="attach"]',
            'button[aria-label*="Attach"]',
            'button[aria-label*="Adjuntar"]',
            '[role="button"][aria-label*="Adjuntar"]',
            '[role="button"][aria-label*="上传和工具"]',
        ]
        
        for selector in selectors:
            try:
                locator = self.page.locator(selector).first
                if locator.is_visible(timeout=1000):
                    try:
                        locator.click(timeout=3000, force=True)
                    except TypeError:
                        locator.click(timeout=3000)
                    self.page.wait_for_timeout(1000)
                    self.logger.info(f"Gemini: clicked add button via {selector}")
                    clicked = True
                    break
            except Exception:
                continue

        if not clicked and hasattr(self.page, "evaluate"):
            script = """
            () => {
                const include = [/上传和工具/, /上传/, /添加文件/, /添加图片/, /附件/, /upload/i, /attach/i, /adjuntar/i, /add files/i, /add image/i];
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
                    .filter((item) => include.some((pattern) => pattern.test(item.text)));
                matches.sort((a, b) => (b.rect.top - a.rect.top) || (b.rect.right - a.rect.right));
                if (!matches.length) return null;
                matches[0].el.click();
                return matches[0].text;
            }
            """
            try:
                clicked_text = self.page.evaluate(script)
                if clicked_text:
                    self.page.wait_for_timeout(1000)
                    self.logger.info("Gemini: clicked add button via DOM scan")
                    clicked = True
            except Exception:
                self.logger.warning("Gemini: add/upload DOM scan was unavailable")

        # Try to find the file chooser trigger in the menu
        for selector in [
            'li:has-text("Upload")',
            'li:has-text("上传")',
            'li:has-text("从计算机")',
            '[role="menuitem"]:has-text("上传")',
            '[role="menuitem"]:has-text("Upload")',
            '[role="menuitem"]:has-text("从电脑")',
            'div[role="menuitem"]:has-text("file")',
            'button:has-text("Upload file")',
            'gem-menu-item:has-text("上传文件")',
            'button:has-text("file")',
            'li:has-text("Adjuntar archivos")',
            '[role="menuitem"]:has-text("Adjuntar archivos")',
            'button:has-text("Adjuntar archivos")',
            'div:has-text("从您的计算机")',
        ]:
            try:
                item = self.page.locator(selector).first
                if item.is_visible(timeout=2000):
                    with self.page.expect_file_chooser(timeout=5000) as chooser:
                        item.click()
                    chooser.value.set_files(image_paths)
                    if self._wait_for_uploads_complete(len(image_paths)):
                        self.logger.info(f"Gemini: uploaded {len(image_paths)} image(s) via menu file chooser")
                        return True
            except Exception:
                continue

        # If expect_file_chooser didn't work, maybe the input[type="file"] appeared in DOM now
        try:
            file_inputs = self.page.locator('input[type="file"]')
            if file_inputs.count() > 0:
                file_inputs.last.set_input_files(image_paths)
                if self._wait_for_uploads_complete(len(image_paths)):
                    self.logger.info(f"Gemini: uploaded {len(image_paths)} image(s) via post-click direct input")
                    return True
        except Exception as exc:
            pass

        self.logger.warning("Gemini: all upload methods failed")
        return False

    def _wait_for_uploads_complete(self, expected_count: int) -> bool:
        """Fail closed unless Gemini exposes stable attachment previews after uploading."""
        timeout_ms = self.cfg.get("upload_timeout", 120) * 1000
        deadline = time.time() + timeout_ms / 1000
        stable_ready = 0

        while time.time() < deadline:
            self.page.wait_for_timeout(1000)
            try:
                state = self.page.evaluate(
                    r"""
                    () => {
                        const normalize = (value) => (value || '').normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/\s+/g, ' ').trim();
                        const text = normalize(document.body.innerText || '');
                        const busy = /(上传中|正在上传|处理中|正在处理|uploading|processing|attaching|subiendo|procesando|adjuntando)/i.test(text)
                            || [...document.querySelectorAll('[role="progressbar"], mat-progress-spinner, mat-spinner, [aria-busy="true"], [data-loading="true"]')]
                                .some((node) => { const rect = node.getBoundingClientRect(); const style = getComputedStyle(node); return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none'; });
                        const attachmentNodes = [...document.querySelectorAll('[data-attachment-id], [data-testid*="attachment"], [data-test-id*="attachment"], attachment-preview, .attachment-preview, mat-chip.attachment, .attachment-chip')]
                            .filter((el) => {
                                const rect = el.getBoundingClientRect();
                                const style = getComputedStyle(el);
                                if (rect.width <= 0 || rect.height <= 0 || style.visibility === 'hidden' || style.display === 'none') return false;
                                const label = [
                                    el.getAttribute('aria-label'),
                                    el.getAttribute('data-test-id'),
                                    el.getAttribute('data-testid'),
                                    el.getAttribute('title'),
                                    el.innerText,
                                ].filter(Boolean).join(' ');
                                return /(image|photo|picture|uploaded|attachment|图片|照片|附件|已上传|adjunto|archivo)/i.test(label);
                            });
                        const unique = [];
                        for (const node of attachmentNodes) {
                            const parent = node.closest('[data-attachment-id], attachment-preview, .attachment-preview, mat-chip.attachment, .attachment-chip') || node;
                            const id = parent.getAttribute('data-attachment-id') || node.getAttribute('data-attachment-id');
                            if (id && unique.some((item) => item.id === id)) continue;
                            if (!id && unique.some((item) => item.node === parent || item.node.contains(parent) || parent.contains(item.node))) continue;
                            unique.push({ id, node: parent });
                        }
                        return { busy, attachment_ids: unique.map((item, index) => item.id || `node-${index}`), visible_text: text };
                    }
                    """
                ) or {}
                visible_text = normalize_ui_text(str(state.get("visible_text", "")))
                text_busy = any(term in visible_text for term in ("uploading", "processing", "attaching", "subiendo", "procesando", "adjuntando", "上传中", "处理中"))
                attachment_ids = state.get("attachment_ids", [])
                unique_ids = list(dict.fromkeys(str(item) for item in attachment_ids)) if isinstance(attachment_ids, list) else []
                ready = len(unique_ids) >= expected_count and not state.get("busy") and not text_busy
                if ready:
                    stable_ready += 1
                    if stable_ready >= 2:
                        return True
                else:
                    stable_ready = 0
            except Exception:
                self.logger.warning("Gemini: upload state could not be verified")
                return False

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
            'button[aria-label*="提交"]',
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
        except Exception:
            self.logger.warning("Gemini: JS extraction was unavailable")

        try:
            body = self.page.locator("body").inner_text(timeout=10000)
            for marker in ["主标题", "副标题"]:
                idx = body.rfind(marker)
                if idx > 100:
                    return self._strip_gemini_chrome(body[idx:]).strip()
            return self._strip_gemini_chrome(body).strip()
        except Exception:
            return "(could not extract response)"

    def _save_debug_snapshot(
        self, product_id: str, label: str, attempts: int = 1, error_kind: str = "other"
    ) -> None:
        if not self.run_dir:
            return
        try:
            save_gemini_diagnostics(
                self.page, self.run_dir, product_id, label, attempts, error_kind
            )
        except Exception as exc:
            self.logger.warning("Gemini: failed to save sanitized diagnostics")

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
