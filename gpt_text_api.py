import json
import urllib.request

from model_provider import safe_provider_request_error, validate_base_url, validate_model_id
from network_retry import RetryPolicy, https_context, run_with_retry
from openai_image_api import OPENAI_IMAGE_USER_AGENT, _prepare_reference_images
from prompt_settings import normalize_prompt_settings
from utils import (
    build_design_prompt,
    build_lovart_confirmation_prompt,
    parse_lovart_confirmation_decision,
    product_output_dir,
    update_status,
)


GPT_TEXT_KEY_ENV = "SUB2API_IMAGE_API_KEY"
DEFAULT_GPT_TEXT_MODEL = "gpt-6"
# Models the Sub2API (Codex / ChatGPT account) relay accepted for chat completions.
GPT_TEXT_MODEL_CHOICES = ("gpt-6", "gpt-6.1-sol", "gpt-5.6")
# Reasoning models can think for a while before answering a long design prompt.
GPT_TEXT_TIMEOUT_SECONDS = 300


def resolve_gpt_text_settings(config: dict) -> tuple[str, str]:
    """Return (base_url, model); the base URL defaults to the Sub2API image profile."""
    gpt_cfg = config.get("gpt_text_api") if isinstance(config.get("gpt_text_api"), dict) else {}
    model = str(gpt_cfg.get("model") or DEFAULT_GPT_TEXT_MODEL).strip()
    base_url = str(gpt_cfg.get("base_url") or "").strip()
    if not base_url:
        image_cfg = config.get("openai_image") if isinstance(config.get("openai_image"), dict) else {}
        profiles = image_cfg.get("profiles") if isinstance(image_cfg.get("profiles"), dict) else {}
        sub2api = profiles.get("sub2api") if isinstance(profiles.get("sub2api"), dict) else {}
        base_url = str(sub2api.get("base_url") or "").strip()
    return base_url, model


class GPTTextAPI:
    """GPT prompt writer over the Sub2API OpenAI-compatible chat completions endpoint."""

    def __init__(self, api_key: str, model: str, base_url: str, logger=None, prompt_settings=None):
        self.api_key = api_key
        self.model = validate_model_id(model)
        self.base_url = validate_base_url(base_url)
        self.logger = logger
        self.prompt_settings = normalize_prompt_settings(prompt_settings)

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
        from utils import get_resource_path
        preamble = get_resource_path("preamble.txt").read_text(encoding="utf-8")
        prompt = f"{preamble}\n\n---\n\n{build_design_prompt(product_name_cn, language, selling_points, image_size=image_size, prompt_settings=self.prompt_settings)}"

        if self.logger:
            self.logger.info(f"GPT API: sending prompt to {self.model} with {len(image_paths)} image(s)")
        result = self._call(prompt, image_paths)

        out_dir = product_output_dir(product_id)
        (out_dir / "gemini_prompt.txt").write_text(result, encoding="utf-8")
        update_status(out_dir, "gpt_text_done", gpt_text_model=self.model, gemini_chars=len(result))
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
        if self.logger:
            self.logger.info(f"GPT API: asking Lovart confirmation decision round {round_index}")
        response = self._call(prompt, [])
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

    def _build_payload(self, text: str, image_paths: list[str] | None = None) -> dict:
        user_content = [{"type": "text", "text": text}]
        if image_paths:
            # Same conservative re-encoding as GPT Image uploads keeps large photos small.
            images, _recompressed = _prepare_reference_images(image_paths, merge=False)
            user_content.extend({"type": "image_url", "image_url": {"url": url}} for url in images)
        # Reasoning models reject custom temperature/max_tokens, so leave both to the server.
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise ecommerce design prompt generator."},
                {"role": "user", "content": user_content},
            ],
        }

    def _call(self, text: str, image_paths: list[str] | None = None) -> str:
        body = json.dumps(self._build_payload(text, image_paths)).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": OPENAI_IMAGE_USER_AGENT,
            },
        )

        def request_operation():
            with urllib.request.urlopen(req, timeout=GPT_TEXT_TIMEOUT_SECONDS, context=https_context()) as resp:
                return json.loads(resp.read().decode("utf-8"))

        failure = None
        data = None
        try:
            data = run_with_retry(request_operation, RetryPolicy(), on_retry=self._retry_notice)
        except Exception as exc:
            failure = safe_provider_request_error("gpt_text", exc, operation="chat_completion")
        if failure is not None:
            if self.logger:
                self.logger.error(f"GPT API request failed ({failure.code})")
            raise failure from None

        result = self._extract_text(data)
        if self.logger:
            self.logger.info(f"GPT API: response ({len(result)} chars)")
        return result

    def _retry_notice(self, message: str) -> None:
        if self.logger:
            self.logger.warning(f"GPT API retry: {message}")

    @staticmethod
    def _extract_text(data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        return str(content or "")
