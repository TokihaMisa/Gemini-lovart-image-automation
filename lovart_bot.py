import json
import os
import re
import time
from pathlib import Path

from lovart_api import AgentSkill, AgentSkillError
from utils import env_or_config, product_output_dir, update_status


def build_lovart_project_name(product_id: str, product_name_cn: str = "") -> str:
    product_id = (product_id or "").strip()
    product_name_cn = " ".join((product_name_cn or "").split())
    if product_id and product_name_cn:
        return f"{product_id}-{product_name_cn}"
    return product_id or product_name_cn


LOVART_IMAGE_MODELS = {
    "auto": None,
    "gpt_image_2": "generate_image_gpt_image_2",
    "nano_banana": "generate_image_nano_banana",
    "nano_banana_2": "generate_image_nano_banana_2",
    "nano_banana_pro": "generate_image_nano_banana_pro",
    "midjourney": "generate_image_midjourney",
    "seedream_v4": "generate_image_seedream_v4",
    "seedream_v4_5": "generate_image_seedream_v4_5",
}


def resolve_lovart_tool_config(cfg: dict) -> dict:
    """Resolve Lovart model and reasoning options into AgentSkill.send kwargs."""
    raw_model = cfg.get("image_model", "auto") or "auto"
    if isinstance(raw_model, (list, tuple)):
        requested_models = [str(item).strip().lower().replace("-", "_") for item in raw_model]
    else:
        requested_models = [
            item.strip().lower().replace("-", "_")
            for item in str(raw_model).split(",")
            if item.strip()
        ]
    if not requested_models:
        requested_models = ["auto"]
    if len(requested_models) > 1:
        requested_models = [model for model in requested_models if model != "auto"] or ["auto"]

    selection = str(cfg.get("model_selection", "prefer") or "prefer").strip().lower()
    mode = cfg.get("reasoning_mode")
    if mode:
        mode = str(mode).strip().lower()
        if mode not in {"fast", "thinking"}:
            raise ValueError("lovart.reasoning_mode must be 'fast' or 'thinking'")

    options = ", ".join(sorted(LOVART_IMAGE_MODELS))
    for model in requested_models:
        if model not in LOVART_IMAGE_MODELS:
            raise ValueError(f"Unknown lovart.image_model '{model}'. Options: {options}")

    tool_names = [LOVART_IMAGE_MODELS[model] for model in requested_models if LOVART_IMAGE_MODELS[model]]
    prefer_models = None
    include_tools = None

    if tool_names:
        if selection == "force":
            include_tools = tool_names
        elif selection == "prefer":
            prefer_models = {"IMAGE": tool_names}
        else:
            raise ValueError("lovart.model_selection must be 'prefer' or 'force'")

    return {
        "image_model": ",".join(requested_models),
        "image_models": requested_models,
        "model_selection": selection,
        "tool_name": tool_names[0] if tool_names else None,
        "tool_names": tool_names,
        "prefer_models": prefer_models,
        "include_tools": include_tools,
        "mode": mode,
    }


class LovartBot:
    """Product-level Lovart API workflow."""

    def __init__(self, config: dict, logger):
        self.cfg = config.get("lovart", {})
        self.logger = logger
        self.tool_config = resolve_lovart_tool_config(self.cfg)

        base_url = os.environ.get("LOVART_BASE_URL", self.cfg.get("base_url", "https://lgw.lovart.ai"))
        access_key = env_or_config(self.cfg, "access_key", "LOVART_ACCESS_KEY")
        secret_key = env_or_config(self.cfg, "secret_key", "LOVART_SECRET_KEY")

        if not access_key or not secret_key:
            raise ValueError(
                "Lovart API keys are required. Set LOVART_ACCESS_KEY and LOVART_SECRET_KEY "
                "as environment variables."
            )

        self.skill = AgentSkill(
            base_url=base_url,
            access_key=access_key,
            secret_key=secret_key,
            timeout=self.cfg.get("timeout", 600),
            poll_interval=self.cfg.get("poll_interval", 5),
        )
        self._fast_mode = False

    def set_fast_mode(self, fast: bool) -> None:
        """Set generation mode. True = fast/credits, False = unlimited/queue."""
        self._fast_mode = fast
        self.skill.set_mode(unlimited=not fast)
        self.logger.info(f"Lovart mode set to: {'fast' if fast else 'unlimited'}")
        if self.tool_config["tool_names"]:
            self.logger.info(
                "Lovart image model: "
                f"{self.tool_config['image_model']} ({self.tool_config['model_selection']})"
            )
        else:
            self.logger.info("Lovart image model: auto")
        if self.tool_config["mode"]:
            self.logger.info(f"Lovart reasoning mode: {self.tool_config['mode']}")

    def create_project(self, product_id: str = "", product_name_cn: str = "") -> str:
        """Create one Lovart project that can be reused across all product steps."""
        project_id = self.skill.create_project()
        self.logger.info(f"Lovart API: created project={project_id} for '{product_id}'")
        project_name = build_lovart_project_name(product_id, product_name_cn)
        if project_name:
            self._rename_project(project_id, project_name)
        return project_id

    def validate_project(self, project_id: str) -> bool:
        """Return whether an existing Lovart project can still be reused."""
        return self.skill.validate_project(project_id)

    def create_and_generate(
        self,
        product_id: str,
        prompt: str,
        image_paths: list[str],
        project_id: str = "",
        confirmation_advisor=None,
        product_name_cn: str = "",
        language: str = "",
        selling_points: str = "",
    ) -> dict | None:
        """Upload images, submit Lovart chat, poll, download artifacts, and rename."""
        self.logger.info(f"Lovart API: starting for '{product_id}'")
        product_dir = product_output_dir(product_id)

        try:
            result, project_id, thread_id = self._submit_and_poll(
                product_dir=product_dir,
                product_id=product_id,
                step_name="detail",
                project_id=project_id,
                prompt=prompt,
                image_paths=image_paths,
                confirmation_advisor=confirmation_advisor,
                product_name_cn=product_name_cn or product_id,
                language=language,
                selling_points=selling_points,
            )

            if result.get("generation_succeeded"):
                out_dir = product_dir / "lovart"
                out_dir.mkdir(parents=True, exist_ok=True)
                downloaded = self.skill.download_artifacts(result, str(out_dir))
                artifact_count = len([item for item in downloaded if item.get("local_path")])
                project_url = f"https://www.lovart.ai/canvas?projectId={project_id}"
                self.logger.info(f"Lovart API: artifacts saved to {out_dir}")
                update_status(
                    product_dir,
                    "lovart_done",
                    project_id=project_id,
                    thread_id=thread_id,
                    artifact_count=artifact_count,
                    project_url=project_url,
                    needs_manual_action=False,
                )
            elif result.get("final_status") == "pending_confirmation":
                confirmation = result.get("pending_confirmation") or {}
                confirmation_text = self._save_pending_confirmation(product_dir, confirmation)
                update_status(
                    product_dir,
                    "needs_manual_action",
                    project_id=project_id,
                    thread_id=thread_id,
                    pending_confirmation_file=str(product_dir / "pending_confirmation.json"),
                    pending_confirmation_text=confirmation_text,
                    reason="Lovart returned pending_confirmation",
                )
            elif result.get("final_status") == "timeout":
                project_url = f"https://www.lovart.ai/canvas?projectId={project_id}"
                update_status(
                    product_dir,
                    "lovart_still_running",
                    project_id=project_id,
                    thread_id=thread_id,
                    project_url=project_url,
                    needs_manual_action=False,
                    reason="Lovart was still running when the local wait timeout was reached",
                )
            else:
                project_url = f"https://www.lovart.ai/canvas?projectId={project_id}" if project_id else ""
                update_status(
                    product_dir,
                    "failed",
                    project_id=project_id,
                    thread_id=thread_id,
                    project_url=project_url,
                    needs_manual_action=False,
                    reason=result.get("warning") or result.get("final_status") or "Lovart generation failed",
                )

            if project_id:
                self._rename_project(project_id, build_lovart_project_name(product_id, product_name_cn))

            return result

        except AgentSkillError as exc:
            update_status(product_dir, "failed", reason=str(exc))
            self.logger.error(f"Lovart API error: {exc}")
            return None
        except Exception as exc:
            update_status(product_dir, "failed", reason=str(exc))
            self.logger.error(f"Lovart unexpected error: {exc}")
            return None

    def create_support_image(
        self,
        product_id: str,
        step_name: str,
        prompt: str,
        image_paths: list[str],
        project_id: str = "",
        confirmation_advisor=None,
        product_name_cn: str = "",
        language: str = "",
        selling_points: str = "",
    ) -> dict | None:
        """Generate one support image, such as white-background or scene image."""
        self.logger.info(f"Lovart API: starting support step '{step_name}' for '{product_id}'")
        product_dir = product_output_dir(product_id)
        try:
            result, project_id, thread_id = self._submit_and_poll(
                product_dir=product_dir,
                product_id=product_id,
                step_name=step_name,
                project_id=project_id,
                prompt=prompt,
                image_paths=image_paths,
                confirmation_advisor=confirmation_advisor,
                product_name_cn=product_name_cn or product_id,
                language=language,
                selling_points=selling_points,
            )

            if not result.get("generation_succeeded"):
                project_url = f"https://www.lovart.ai/canvas?projectId={project_id}" if project_id else ""
                update_status(
                    product_dir,
                    f"lovart_{step_name}_failed",
                    project_id=project_id,
                    thread_id=thread_id,
                    project_url=project_url,
                    reason=result.get("warning") or result.get("final_status") or "Lovart support image failed",
                )
                return result

            out_dir = product_dir / "lovart_steps" / step_name
            out_dir.mkdir(parents=True, exist_ok=True)
            downloaded = self.skill.download_artifacts(result, str(out_dir), prefix=step_name)
            image_files = [
                item.get("local_path")
                for item in downloaded
                if item.get("local_path") and item.get("type") in {"image", "unknown"}
            ]
            if not image_files:
                image_files = [item.get("local_path") for item in downloaded if item.get("local_path")]
            first_image = image_files[0] if image_files else ""
            result["local_path"] = first_image
            result["downloaded"] = downloaded
            update_status(
                product_dir,
                f"lovart_{step_name}_done",
                project_id=project_id,
                thread_id=thread_id,
                local_path=first_image,
                **{f"lovart_{step_name}_local_path": first_image},
                artifact_count=len(image_files),
            )
            return result
        except AgentSkillError as exc:
            update_status(product_dir, f"lovart_{step_name}_failed", reason=str(exc))
            self.logger.error(f"Lovart API error in {step_name}: {exc}")
            return None
        except Exception as exc:
            update_status(product_dir, f"lovart_{step_name}_failed", reason=str(exc))
            self.logger.error(f"Lovart unexpected error in {step_name}: {exc}")
            return None

    def _submit_and_poll(
        self,
        product_dir: Path,
        product_id: str,
        step_name: str,
        prompt: str,
        image_paths: list[str],
        confirmation_advisor,
        product_name_cn: str,
        language: str,
        selling_points: str,
        project_id: str = "",
    ) -> tuple[dict, str, str]:
        project_id = project_id or self.create_project(product_id, product_name_cn)
        return self._submit_and_poll_once(
            product_dir=product_dir,
            product_id=product_id,
            step_name=step_name,
            attempt_name="primary",
            project_id=project_id,
            prompt=prompt,
            image_paths=image_paths,
            confirmation_advisor=confirmation_advisor,
            product_name_cn=product_name_cn,
            language=language,
            selling_points=selling_points,
            tool_config=self.tool_config,
        )

    def _submit_and_poll_once(
        self,
        product_dir: Path,
        product_id: str,
        step_name: str,
        attempt_name: str,
        project_id: str,
        prompt: str,
        image_paths: list[str],
        confirmation_advisor,
        product_name_cn: str,
        language: str,
        selling_points: str,
        tool_config: dict,
    ) -> tuple[dict, str, str]:
        status = read_status(product_dir)
        is_still_running = status.get("lovart_still_running")
        last_submitted = status.get(f"lovart_{step_name}_submitted")
        thread_id = status.get("thread_id")
        
        if is_still_running and last_submitted and thread_id:
            self.logger.info(f"Lovart API: Resuming previously timed out {step_name} thread={thread_id} in project={project_id}")
            # Clear the still_running flag locally so we don't accidentally get stuck in resume loops if it fails now
            update_status(product_dir, "lovart_still_running_resumed", lovart_still_running=False)
        else:
            file_suffix = step_name if attempt_name == "primary" else f"{step_name}_{attempt_name}"
            attachment_records = self._upload_images(image_paths)
            attachment_urls = [item["url"] for item in attachment_records]
            attachment_file = product_dir / f"lovart_attachments_{file_suffix}.json"
            attachment_file.write_text(
                json.dumps(attachment_records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            update_status(
                product_dir,
                f"lovart_{step_name}_uploaded",
                attachment_count=len(attachment_urls),
                attachment_file=str(attachment_file),
                attempt=attempt_name,
            )
    
            prompt_file = product_dir / f"lovart_prompt_{file_suffix}.txt"
            prompt_file.write_text(prompt, encoding="utf-8")
            self.logger.info(
                f"Lovart API: sending {step_name} prompt ({len(prompt)} chars), "
                f"attempt={attempt_name}, model={tool_config['image_model']} ({tool_config['model_selection']})"
            )
            thread_id = self.skill.send(
                prompt=prompt,
                project_id=project_id,
                attachments=attachment_urls if attachment_urls else None,
                prefer_models=tool_config["prefer_models"],
                include_tools=tool_config["include_tools"],
                mode=tool_config["mode"],
            )
            self.logger.info(f"Lovart API: sent {step_name} project={project_id}, thread={thread_id}")
            
        update_status(
            product_dir,
            f"lovart_{step_name}_submitted",
            project_id=project_id,
            thread_id=thread_id,
            image_model=tool_config["image_model"],
            model_selection=tool_config["model_selection"],
            reasoning_mode=tool_config["mode"] or "default",
            prompt_file=str(prompt_file),
            attempt=attempt_name,
        )
        result = self._poll_with_progress(thread_id, project_id, product_dir=product_dir)
        result = self._resolve_pending_confirmations(
            result=result,
            product_dir=product_dir,
            product_id=product_id,
            product_name_cn=product_name_cn,
            language=language,
            selling_points=selling_points,
            project_id=project_id,
            thread_id=thread_id,
            confirmation_advisor=confirmation_advisor,
        )
        return result, project_id, thread_id

    def _upload_images(self, paths: list[str]) -> list[dict]:
        records = []
        attempts = max(1, int(self.cfg.get("upload_attempts", 3) or 3))
        retry_delay = float(self.cfg.get("upload_retry_delay", 2) or 0)
        for path in paths:
            for attempt in range(1, attempts + 1):
                try:
                    url = self.skill.upload_file(path)
                    records.append({
                        "local_path": str(path),
                        "filename": Path(path).name,
                        "url": url,
                    })
                    self.logger.info(f"Lovart API: uploaded {Path(path).name}")
                    break
                except Exception as exc:
                    if attempt < attempts:
                        self.logger.warning(
                            f"Lovart API: upload failed for {path} "
                            f"(attempt {attempt}/{attempts}); retrying: {exc}"
                        )
                        if retry_delay > 0:
                            time.sleep(retry_delay)
                    else:
                        self.logger.warning(f"Lovart API: upload failed for {path}: {exc}")
        return records

    def _resolve_pending_confirmations(
        self,
        result: dict,
        product_dir: Path,
        product_id: str,
        product_name_cn: str,
        language: str,
        selling_points: str,
        project_id: str,
        thread_id: str,
        confirmation_advisor,
    ) -> dict:
        max_rounds = int(self.cfg.get("max_confirmation_rounds", 5) or 5)
        max_credits = int(self.cfg.get("max_auto_confirm_credits", 10) or 10)
        lovart_mode = "fast" if self._fast_mode else "unlimited"

        for round_index in range(1, max_rounds + 1):
            if result.get("final_status") != "pending_confirmation":
                return result

            confirmation = result.get("pending_confirmation") or {}
            confirmation_text = self._save_pending_confirmation(product_dir, confirmation, round_index)
            estimated_cost = self._confirmation_estimated_cost(confirmation)
            update_status(
                product_dir,
                "needs_manual_action",
                project_id=project_id,
                thread_id=thread_id,
                pending_confirmation_file=str(product_dir / "pending_confirmation.json"),
                pending_confirmation_text=confirmation_text,
                pending_confirmation_estimated_cost=estimated_cost if estimated_cost is not None else "",
                pending_confirmation_round=round_index,
                reason="Lovart returned pending_confirmation",
            )

            if estimated_cost is not None and not self._fast_mode:
                result["warning"] = (
                    f"Lovart showed a {estimated_cost:g}-credit confirmation in unlimited mode; "
                    "left unconfirmed for manual review."
                )
                update_status(
                    product_dir,
                    "lovart_credit_prompt_waiting",
                    reason=result["warning"],
                    needs_manual_action=True,
                )
                self.logger.warning(result["warning"])
                return result

            if not confirmation_advisor or not hasattr(confirmation_advisor, "advise_lovart_confirmation"):
                result["warning"] = "Lovart requires confirmation, but no Gemini confirmation advisor is available."
                return result

            try:
                decision = confirmation_advisor.advise_lovart_confirmation(
                    product_id=product_id,
                    product_name_cn=product_name_cn,
                    language=language,
                    selling_points=selling_points,
                    confirmation_text=confirmation_text,
                    confirmation_payload=confirmation,
                    project_id=project_id,
                    thread_id=thread_id,
                    round_index=round_index,
                    max_auto_confirm_credits=max_credits,
                    lovart_mode=lovart_mode,
                )
            except Exception as exc:
                result["warning"] = f"Gemini confirmation advisor failed: {exc}"
                update_status(product_dir, "needs_manual_action", reason=result["warning"])
                self.logger.warning(result["warning"])
                return result

            update_status(
                product_dir,
                "lovart_confirmation_decided",
                lovart_confirmation_round=round_index,
                lovart_confirmation_decision=decision["decision"],
                lovart_confirmation_reason=decision["reason"],
                lovart_confirmation_message_to_lovart=decision.get("message_to_lovart", ""),
            )

            if decision["decision"] != "CONFIRM":
                result["warning"] = f"Gemini chose STOP: {decision['reason']}"
                self.logger.warning(f"Lovart confirmation stopped by Gemini: {decision['reason']}")
                return result

            if estimated_cost is not None and estimated_cost > max_credits:
                result["warning"] = (
                    f"Gemini chose CONFIRM, but estimated cost {estimated_cost:g} "
                    f"exceeds configured limit {max_credits}."
                )
                update_status(product_dir, "needs_manual_action", reason=result["warning"])
                self.logger.warning(result["warning"])
                return result

            self.logger.info(f"Lovart API: Gemini approved confirmation round {round_index}")
            confirm_response = self.skill.confirm(thread_id)
            (product_dir / f"lovart_confirm_response_{round_index}.json").write_text(
                json.dumps(confirm_response, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            update_status(
                product_dir,
                "lovart_confirmation_sent",
                lovart_confirmation_round=round_index,
                lovart_confirm_response_file=str(product_dir / f"lovart_confirm_response_{round_index}.json"),
                needs_manual_action=False,
            )
            result = self._poll_with_progress(thread_id, project_id)

        if result.get("final_status") == "pending_confirmation":
            result["warning"] = f"Lovart still requires confirmation after {max_rounds} Gemini decision round(s)."
        return result

    def _save_pending_confirmation(self, product_dir: Path, confirmation, round_index: int | None = None) -> str:
        """Save Lovart's confirmation payload and return a readable summary when possible."""
        json_path = product_dir / "pending_confirmation.json"
        text_path = product_dir / "pending_confirmation.txt"
        serialized = json.dumps(confirmation, ensure_ascii=False, indent=2)
        json_path.write_text(serialized, encoding="utf-8")
        if round_index is not None:
            (product_dir / f"pending_confirmation_{round_index}.json").write_text(serialized, encoding="utf-8")
        text = self._confirmation_text(confirmation)
        if text:
            text_path.write_text(text, encoding="utf-8")
            if round_index is not None:
                (product_dir / f"pending_confirmation_{round_index}.txt").write_text(text, encoding="utf-8")
            self.logger.warning(f"Lovart confirmation detail: {text[:500]}")
        else:
            text_path.write_text(serialized, encoding="utf-8")
            if round_index is not None:
                (product_dir / f"pending_confirmation_{round_index}.txt").write_text(serialized, encoding="utf-8")
            self.logger.warning(f"Lovart confirmation payload saved: {json_path}")
        return text

    @classmethod
    def _confirmation_estimated_cost(cls, value):
        if isinstance(value, dict):
            for key in ("estimated_cost", "cost", "credits", "credit_cost"):
                if key in value:
                    try:
                        return float(value[key])
                    except (TypeError, ValueError):
                        pass
            for item in value.values():
                found = cls._confirmation_estimated_cost(item)
                if found is not None:
                    return found
        if isinstance(value, list):
            for item in value:
                found = cls._confirmation_estimated_cost(item)
                if found is not None:
                    return found
        if isinstance(value, str):
            match = re.search(r"(\d+(?:\.\d+)?)\s*credits?", value, flags=re.I)
            if match:
                return float(match.group(1))
        return None

    @classmethod
    def _confirmation_text(cls, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            parts = [cls._confirmation_text(item) for item in value]
            return "\n".join(part for part in parts if part)
        if isinstance(value, dict):
            priority_keys = (
                "message",
                "title",
                "description",
                "content",
                "text",
                "prompt",
                "reason",
                "detail",
                "details",
                "confirm_message",
                "confirmation_message",
            )
            parts = []
            for key in priority_keys:
                if key in value:
                    text = cls._confirmation_text(value.get(key))
                    if text:
                        parts.append(text)
            if parts:
                return "\n".join(dict.fromkeys(parts))
            nested = [cls._confirmation_text(item) for item in value.values()]
            return "\n".join(part for part in nested if part)
        return ""

    def _poll_with_progress(self, thread_id: str, project_id: str, product_dir: Path | None = None) -> dict:
        timeout = self.cfg.get("wait_timeout", 10800)
        interval = self.cfg.get("poll_interval", 5)
        deadline = time.time() + timeout
        status_names = {
            "pending": "排队中...",
            "running": "生成中...",
            "done": "完成!",
            "abort": "已取消",
        }
        start = time.time()

        while time.time() < deadline:
            try:
                status_info = self.skill.get_status(thread_id)
                status = status_info.get("status", "unknown")
                elapsed = int(time.time() - start)
                dots = "." * ((elapsed // interval) % 4)

                result = self.skill.get_result(thread_id)
                pending_confirmation = result.get("pending_confirmation")
                if pending_confirmation:
                    estimated_cost = self._confirmation_estimated_cost(pending_confirmation)
                    if estimated_cost is not None and not self._fast_mode:
                        print()
                        confirmation_text = ""
                        if product_dir:
                            confirmation_text = self._save_pending_confirmation(product_dir, pending_confirmation)
                            update_status(
                                product_dir,
                                "lovart_credit_prompt_waiting",
                                project_id=project_id,
                                thread_id=thread_id,
                                pending_confirmation_file=str(product_dir / "pending_confirmation.json"),
                                pending_confirmation_text=confirmation_text,
                                pending_confirmation_estimated_cost=estimated_cost,
                                needs_manual_action=True,
                                reason=(
                                    f"Lovart showed a {estimated_cost:g}-credit confirmation in unlimited mode; "
                                    "left unconfirmed for manual review."
                                ),
                            )
                        self.logger.warning(
                            f"Lovart API: {estimated_cost:g}-credit confirmation appeared in unlimited mode; "
                            "not confirming; manual action required"
                        )
                        result["warning"] = (
                            f"Lovart showed a {estimated_cost:g}-credit confirmation in unlimited mode; "
                            "left unconfirmed for manual review."
                        )
                        return self._normalize_result(result, "pending_confirmation", project_id)
                    print()
                    self.logger.warning("Lovart API: pending confirmation")
                    return self._normalize_result(result, "pending_confirmation", project_id)

                if status == "done":
                    time.sleep(5)
                    confirm = self.skill.get_status(thread_id)
                    if confirm.get("status") in ("done", "abort"):
                        print()
                        self.logger.info(f"Lovart API: done ({elapsed}s)")
                        return self._normalize_result(self.skill.get_result(thread_id), confirm.get("status"), project_id)

                if status == "abort":
                    print()
                    self.logger.warning("Lovart API: aborted")
                    return self._normalize_result(result, "abort", project_id)

                print(f"\r  [{elapsed}s] {status_names.get(status, status)}{dots}   ", end="", flush=True)
            except Exception as exc:
                self.logger.warning(f"Lovart poll error: {exc}")

            time.sleep(interval)

        print()
        self.logger.warning(
            f"Lovart API: still running after local wait timeout ({timeout}s); "
            "the Lovart project may finish in the background"
        )
        try:
            final_status = self.skill.get_status(thread_id).get("status", "timeout")
            if final_status == "done":
                return self._normalize_result(self.skill.get_result(thread_id), "done", project_id)
        except Exception as exc:
            self.logger.warning(f"Lovart final status check failed: {exc}")
        return self._normalize_result(self.skill.get_result(thread_id), "timeout", project_id)

    @staticmethod
    def _normalize_result(result: dict, final_status: str, project_id: str) -> dict:
        result["final_status"] = final_status
        result["project_id"] = project_id

        if final_status == "pending_confirmation":
            result["generation_succeeded"] = False
            return result

        has_artifact = any((item.get("artifacts") or []) for item in (result.get("items") or []))
        result["generation_succeeded"] = final_status == "done" and has_artifact
        if final_status == "done" and not has_artifact:
            result["warning"] = "Lovart finished without returning image artifacts."
        return result

    def _rename_project(self, project_id: str, name: str) -> None:
        try:
            self.skill.rename_project(project_id, name)
            self.logger.info(f"Lovart API: renamed project {project_id} to '{name}'")
        except Exception as exc:
            try:
                self.logger.warning(f"Lovart rename failed, trying fallback: {exc}")
                self.skill._request("POST", "/v1/openapi/project/save", body={
                    "action": "rename",
                    "project_id": project_id,
                    "project_name": name,
                })
                self.logger.info("Lovart API: fallback rename OK")
            except Exception as fallback_exc:
                self.logger.warning(f"Lovart API: rename failed: {fallback_exc}")
