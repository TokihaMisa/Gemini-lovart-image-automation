from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterator, Protocol, Sequence

from PIL import Image

from gemini_browser_session import (
    GeminiPageState,
    acquire_login_helper_owner,
    build_browser_launch_options,
    inspect_gemini_page,
    login_runtime_paths,
    navigate_gemini_with_retry,
    release_login_helper_owner,
)
from network_retry import retry_policy_from_config
from utils import load_config


class CheckState(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class HealthCheckResult:
    name: str
    state: CheckState
    message: str

    def to_json(self) -> dict[str, str]:
        return {
            "name": self.name[:40],
            "state": self.state.value,
            "message": self.message[:240],
        }


class HealthCheckReporter(Protocol):
    def emit(self, result: HealthCheckResult) -> None: ...


class HealthCheckHarness(Protocol):
    def check_browser(self) -> bool: ...

    def check_profile_lock(self) -> bool: ...

    def check_page_ready(self) -> bool: ...

    def check_logged_in(self) -> bool: ...

    def check_editor(self) -> bool: ...

    def enter_temporary_chat(self) -> bool: ...

    def open_mode_menu(self) -> bool: ...

    def select_flash(self) -> bool: ...

    def select_extended_thinking(self) -> bool: ...

    def check_upload_control(self) -> bool: ...

    def upload_blank_image(self) -> bool: ...

    def check_send_button(self) -> bool: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class _CheckDefinition:
    name: str
    method: str
    pass_message: str
    fail_message: str
    exception_message: str


_CHECKS = (
    _CheckDefinition(
        "浏览器环境",
        "check_browser",
        "浏览器和配置可用",
        "浏览器或配置不可用",
        "浏览器启动失败",
    ),
    _CheckDefinition(
        "配置占用",
        "check_profile_lock",
        "浏览器配置可用",
        "Gemini 浏览器配置正在使用中",
        "控件不可用",
    ),
    _CheckDefinition(
        "页面加载",
        "check_page_ready",
        "Gemini 页面已就绪",
        "页面加载失败",
        "页面加载失败",
    ),
    _CheckDefinition(
        "登录状态",
        "check_logged_in",
        "Gemini 账户已登录",
        "Gemini 账户未登录",
        "控件不可用",
    ),
    _CheckDefinition(
        "提示词编辑器",
        "check_editor",
        "提示词编辑器可用",
        "未找到提示词编辑器",
        "控件不可用",
    ),
    _CheckDefinition(
        "临时对话",
        "enter_temporary_chat",
        "已进入临时对话",
        "未找到临时对话按钮",
        "控件不可用",
    ),
    _CheckDefinition(
        "模式菜单",
        "open_mode_menu",
        "模式菜单已打开",
        "模式菜单不可用",
        "控件不可用",
    ),
    _CheckDefinition(
        "Flash 模型",
        "select_flash",
        "已选择非 Lite Flash",
        "未找到非 Lite Flash 模型",
        "控件不可用",
    ),
    _CheckDefinition(
        "扩展思考",
        "select_extended_thinking",
        "扩展思考已启用",
        "扩展思考不可用",
        "控件不可用",
    ),
    _CheckDefinition(
        "上传控件",
        "check_upload_control",
        "上传控件可用",
        "未找到上传控件",
        "控件不可用",
    ),
    _CheckDefinition(
        "图片上传",
        "upload_blank_image",
        "空白图片附件已就绪",
        "图片上传失败",
        "上传失败",
    ),
    _CheckDefinition(
        "发送按钮",
        "check_send_button",
        "发送按钮可用（未点击）",
        "发送按钮不可用",
        "控件不可用",
    ),
)


def render_health_check(results: Sequence[HealthCheckResult]) -> str:
    icons = {
        CheckState.PASS: "✅",
        CheckState.FAIL: "❌",
        CheckState.SKIP: "⚠️",
    }
    lines = [
        f"{icons[item.state]} **{item.name}**：{item.message}"
        for item in results
    ]
    counts = {
        state: sum(item.state is state for item in results)
        for state in CheckState
    }
    lines.append(
        f"\n正常 {counts[CheckState.PASS]} · "
        f"异常 {counts[CheckState.FAIL]} · "
        f"跳过 {counts[CheckState.SKIP]}"
    )
    return "\n".join(lines)


def run_checks(harness: HealthCheckHarness) -> Iterator[HealthCheckResult]:
    prerequisite_ok = True
    try:
        for check in _CHECKS:
            if not prerequisite_ok:
                yield HealthCheckResult(
                    check.name,
                    CheckState.SKIP,
                    "前置检查失败",
                )
                continue

            try:
                passed = bool(getattr(harness, check.method)())
            except Exception:
                passed = False
                message = check.exception_message
            else:
                message = check.pass_message if passed else check.fail_message

            state = CheckState.PASS if passed else CheckState.FAIL
            yield HealthCheckResult(check.name, state, message)
            prerequisite_ok = passed
    finally:
        try:
            harness.close()
        except Exception:
            pass


class PlaywrightGeminiHealthCheckHarness:
    """Own an isolated Gemini browser while exercising production controls."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self.config = load_config(str(self.config_path))
        if not isinstance(self.config, dict):
            raise ValueError("Invalid configuration")
        self.logger = logging.getLogger("gemini_health_check")
        self.paths = None
        self.owner = None
        self.playwright = None
        self.context = None
        self.page = None
        self.status = None
        self.bot = None
        self.launch_options = None
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="gemini-health-check-"
        )
        self.blank_image = Path(self._temporary_directory.name) / "blank-health-check.png"
        Image.new("RGB", (32, 32), "white").save(self.blank_image, format="PNG")
        self._test_mode = False

    @classmethod
    def from_test_bot(cls, bot: Any) -> "PlaywrightGeminiHealthCheckHarness":
        harness = cls.__new__(cls)
        harness.config_path = Path("config.yaml")
        harness.config = {}
        harness.logger = logging.getLogger("gemini_health_check.test")
        harness.paths = None
        harness.owner = None
        harness.playwright = None
        harness.context = None
        harness.page = None
        harness.status = None
        harness.bot = bot
        harness.launch_options = {}
        harness._temporary_directory = tempfile.TemporaryDirectory(
            prefix="gemini-health-check-test-"
        )
        harness.blank_image = (
            Path(harness._temporary_directory.name) / "blank-health-check.png"
        )
        Image.new("RGB", (1, 1), "white").save(harness.blank_image, format="PNG")
        harness._test_mode = True
        return harness

    def check_browser(self) -> bool:
        if self._test_mode:
            return True
        self.launch_options = build_browser_launch_options(
            self.config,
            config_path=self.config_path,
        )
        executable = self.launch_options.get("executable_path")
        if executable:
            return Path(str(executable)).is_file()
        from playwright.sync_api import sync_playwright

        playwright = sync_playwright().start()
        try:
            return Path(playwright.chromium.executable_path).is_file()
        finally:
            playwright.stop()

    def check_profile_lock(self) -> bool:
        if self._test_mode:
            return True
        self.paths = login_runtime_paths(self.config_path)
        self.owner = acquire_login_helper_owner(self.paths)
        return self.owner is not None

    def check_page_ready(self) -> bool:
        if self._test_mode:
            return True
        from playwright.sync_api import sync_playwright

        if self.owner is None or self.launch_options is None:
            return False
        self.playwright = sync_playwright().start()
        self.context = self.playwright.chromium.launch_persistent_context(
            **self.launch_options
        )
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        gemini_config = self.config.get("gemini", {})
        base_url = (
            gemini_config.get("base_url", "https://gemini.google.com/app")
            if isinstance(gemini_config, dict)
            else "https://gemini.google.com/app"
        )
        self.status = navigate_gemini_with_retry(
            self.page,
            str(base_url),
            retry_policy_from_config(self.config),
            logger=self.logger,
        )
        if self.status.state not in (GeminiPageState.READY, GeminiPageState.WAITING_LOGIN):
            return False
        if self.status.state is GeminiPageState.READY:
            policy = retry_policy_from_config(self.config)
            self.page.wait_for_function(
                """
                () => {
                    const visible = (node) => {
                        const rect = node.getBoundingClientRect();
                        const style = getComputedStyle(node);
                        return rect.width > 0 && rect.height > 0
                            && style.visibility !== 'hidden' && style.display !== 'none';
                    };
                    const editors = [...document.querySelectorAll(
                        '[contenteditable="true"], textarea, [role="textbox"]'
                    )].filter(visible);
                    const controls = [...document.querySelectorAll(
                        'button, [role="button"]'
                    )].filter(visible);
                    return editors.length > 0 && controls.length >= 3;
                }
                """,
                timeout=int(policy.page_ready_timeout * 1000),
            )
        from gemini_bot import GeminiBot

        self.bot = GeminiBot(self.page, self.config, self.logger)
        return True

    def check_logged_in(self) -> bool:
        if self._test_mode:
            return True
        if self.page is None:
            return False
        self.status = inspect_gemini_page(self.page)
        return self.status.state is GeminiPageState.READY and self.status.ready

    def check_editor(self) -> bool:
        return bool(self.bot and self.bot.health_check_editor())

    def enter_temporary_chat(self) -> bool:
        return bool(
            self.bot
            and self.bot._start_temporary_chat()
            and self.bot.health_check_temporary_chat_active()
        )

    def open_mode_menu(self) -> bool:
        return bool(self.bot and self.bot._open_mode_menu())

    def select_flash(self) -> bool:
        return bool(self.bot and self.bot._click_flash_model())

    def select_extended_thinking(self) -> bool:
        return bool(self.bot and self.bot._select_extended_thinking_option())

    def check_upload_control(self) -> bool:
        return bool(self.bot and self.bot.health_check_upload_control())

    def upload_blank_image(self) -> bool:
        if not self.bot or not self.blank_image.is_file():
            return False
        original_attempts = self.bot.cfg.get("upload_attempts")
        original_timeout = self.bot.cfg.get("upload_timeout")
        self.bot.cfg["upload_attempts"] = 1
        try:
            configured_timeout = float(original_timeout or 120)
        except (TypeError, ValueError):
            configured_timeout = 120
        self.bot.cfg["upload_timeout"] = min(max(configured_timeout, 120), 180)
        try:
            return bool(self.bot._upload_images([str(self.blank_image)]))
        finally:
            if original_attempts is None:
                self.bot.cfg.pop("upload_attempts", None)
            else:
                self.bot.cfg["upload_attempts"] = original_attempts
            if original_timeout is None:
                self.bot.cfg.pop("upload_timeout", None)
            else:
                self.bot.cfg["upload_timeout"] = original_timeout

    def check_send_button(self) -> bool:
        return bool(self.bot and self.bot.health_check_send_button())

    def close(self) -> None:
        if self.context is not None:
            try:
                self.context.close()
            except Exception:
                pass
            self.context = None
        if self.playwright is not None:
            try:
                self.playwright.stop()
            except Exception:
                pass
            self.playwright = None
        if self.paths is not None and self.owner is not None:
            release_login_helper_owner(self.paths, self.owner)
            self.owner = None
        temporary_directory = getattr(self, "_temporary_directory", None)
        if temporary_directory is not None:
            temporary_directory.cleanup()
            self._temporary_directory = None


def _default_harness_factory(config_path: Path) -> HealthCheckHarness:
    return PlaywrightGeminiHealthCheckHarness(config_path)


def run_gemini_health_check(
    config_path: str | Path,
    reporter: HealthCheckReporter,
    *,
    harness_factory: Callable[[Path], HealthCheckHarness] | None = None,
) -> int:
    factory = harness_factory or _default_harness_factory
    try:
        harness = factory(Path(config_path))
    except Exception:
        results = [
            HealthCheckResult(
                _CHECKS[0].name,
                CheckState.FAIL,
                "浏览器启动失败",
            ),
            *[
                HealthCheckResult(check.name, CheckState.SKIP, "前置检查失败")
                for check in _CHECKS[1:]
            ],
        ]
        for result in results:
            reporter.emit(result)
        return 1

    failed = False
    for result in run_checks(harness):
        reporter.emit(result)
        failed = failed or result.state is CheckState.FAIL
    return 1 if failed else 0


class _JsonlReporter:
    def __init__(self, handle):
        self.handle = handle

    def write(self, value: dict[str, object]) -> None:
        self.handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def emit(self, result: HealthCheckResult) -> None:
        self.write({"event": "result", "result": result.to_json()})


def run_health_check_cli(
    config_path: str | Path,
    status_file: str | Path,
) -> int:
    target = Path(status_file).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        reporter = _JsonlReporter(handle)
        reporter.write({"event": "running", "results": []})
        try:
            exit_code = int(run_gemini_health_check(config_path, reporter))
        except Exception:
            reporter.emit(
                HealthCheckResult(
                    "体检进程",
                    CheckState.FAIL,
                    "体检进程异常结束",
                )
            )
            exit_code = 1
        reporter.write({"event": "complete", "exit_code": exit_code})
    return exit_code


def run_health_check_cli_from_argv(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gemini-health-check", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--status-file", required=True)
    args, _unknown = parser.parse_known_args(list(argv))
    return run_health_check_cli(args.config, args.status_file)
