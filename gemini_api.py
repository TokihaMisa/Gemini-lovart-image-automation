import base64
import json
import urllib.request
from pathlib import Path

from model_provider import safe_provider_request_error, validate_base_url, validate_model_id
from network_retry import RetryPolicy, run_with_retry
from prompt_settings import normalize_prompt_settings
from utils import (
    build_design_prompt,
    build_lovart_confirmation_prompt,
    parse_lovart_confirmation_decision,
    product_output_dir,
    update_status,
)


class GeminiAPI:
    """Gemini API client that does not require browser automation."""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        base_url: str = DEFAULT_BASE_URL,
        logger=None,
        prompt_settings=None,
    ):
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
            self.logger.info(f"Gemini API: sending prompt with {len(image_paths)} image(s)")
        result = self._call(prompt, image_paths)

        out_dir = product_output_dir(product_id)
        (out_dir / "gemini_prompt.txt").write_text(result, encoding="utf-8")
        update_status(out_dir, "gemini_done", gemini_chars=len(result))

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
            self.logger.info(f"Gemini API: asking Lovart confirmation decision round {round_index}")
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

    def _call(self, text: str, image_paths: list[str] | None = None) -> str:
        """Send a request to Gemini API. Returns the text response."""
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        parts = [{"text": text}]
        if image_paths:
            for path in image_paths:
                try:
                    data = Path(path).read_bytes()
                    encoded = base64.b64encode(data).decode("ascii")
                    ext = Path(path).suffix.lstrip(".").lower() or "jpeg"
                    mime = "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"
                    parts.append({"inline_data": {"mime_type": mime, "data": encoded}})
                except Exception as exc:
                    if self.logger:
                        self.logger.warning(f"Gemini API: failed to encode {path}: {exc}")

        body = json.dumps({
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192},
        }).encode("utf-8")

        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")

        def request_operation():
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))

        failure = None
        data = None
        try:
            data = run_with_retry(
                request_operation,
                RetryPolicy(),
                on_retry=self._retry_notice,
            )
        except Exception as exc:
            failure = safe_provider_request_error("gemini", exc)
        if failure is not None:
            if self.logger:
                self.logger.error(f"Gemini API request failed ({failure.code})")
            raise failure from None

        candidates = data.get("candidates", [])
        if not candidates:
            if self.logger:
                self.logger.warning("Gemini API: no candidates in response")
            return ""

        response_parts = candidates[0].get("content", {}).get("parts", [])
        result = "".join(part.get("text", "") for part in response_parts)
        if self.logger:
            self.logger.info(f"Gemini API: response ({len(result)} chars)")
        return result

    def _retry_notice(self, message: str) -> None:
        if self.logger:
            self.logger.warning(f"Gemini API retry: {message}")
