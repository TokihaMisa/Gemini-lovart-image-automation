import base64
import json
import urllib.request
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from model_provider import validate_base_url, validate_model_id
from prompt_settings import normalize_prompt_settings
from utils import (
    build_design_prompt,
    build_lovart_confirmation_prompt,
    parse_lovart_confirmation_decision,
    product_output_dir,
    update_status,
)


DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def resolve_nvidia_model(cfg: dict) -> str:
    direct = str(cfg.get("model", "") or "").strip()
    if direct:
        return direct
    choice = str(cfg.get("model_choice", "kimi") or "kimi").strip().lower()
    model = cfg.get("models", {}).get(choice)
    if not model:
        raise ValueError(f"Unknown NVIDIA model choice '{choice}'. Configure nvidia_api.model.")
    return str(model)


class NvidiaAPI:
    """NVIDIA NIM OpenAI-compatible chat completions client."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_NVIDIA_BASE_URL,
        logger=None,
        send_images: bool = True,
        prompt_settings=None,
    ):
        self.api_key = api_key
        self.model = validate_model_id(model)
        self.base_url = validate_base_url(base_url)
        self.logger = logger
        self.send_images = send_images
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
        images = image_paths if self.send_images else []

        if self.logger:
            self.logger.info(f"NVIDIA API: sending prompt to {self.model} with {len(images)} image(s)")
        result = self._call(prompt, images)

        out_dir = product_output_dir(product_id)
        (out_dir / "gemini_prompt.txt").write_text(result, encoding="utf-8")
        update_status(out_dir, "nvidia_done", nvidia_model=self.model, gemini_chars=len(result))
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
            self.logger.info(f"NVIDIA API: asking Lovart confirmation decision round {round_index}")
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
            for path in image_paths:
                data = Path(path).read_bytes()
                encoded = base64.b64encode(data).decode("ascii")
                ext = Path(path).suffix.lstrip(".").lower() or "jpeg"
                mime = "image/jpeg" if ext in {"jpg", "jpeg"} else f"image/{ext}"
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                })

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise ecommerce design prompt generator."},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.6,
            "max_tokens": 8192,
        }

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call(self, text: str, image_paths: list[str] | None = None) -> str:
        payload = self._build_payload(text, image_paths)
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            if hasattr(exc, "code") and exc.code in (400, 401, 403, 404):
                if self.logger:
                    self.logger.error(f"NVIDIA API client error {exc.code}: {exc}")
                raise
            if self.logger:
                self.logger.warning(f"NVIDIA API call failed, will retry: {exc}")
            raise

        result = self._extract_text(data)
        if self.logger:
            self.logger.info(f"NVIDIA API: response ({len(result)} chars)")
        return result

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
