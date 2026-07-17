from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

import yaml
from playwright.sync_api import sync_playwright

from network_retry import (
    RetryKind,
    RetryPolicy,
    classify_network_error,
    retry_policy_from_config,
    run_with_retry,
)


class GeminiLoginRequiredError(RuntimeError):
    """Raised when Gemini redirects to an interactive Google login page."""

    def __init__(self, message: str = "Gemini 未登录，请先在软件内打开登录浏览器完成登录。"):
        super().__init__(message)


class GeminiPageNotReadyError(RuntimeError):
    """Raised when Gemini cannot reach a ready editor before processing begins."""

    def __init__(self, message: str = "Gemini 页面未准备完成，请检查网络后重试。"):
        super().__init__(message)


class GeminiPermanentTlsError(RuntimeError):
    """Raised for certificate failures that must not be retried or bypassed."""

    def __init__(self, message: str = "Gemini TLS 证书验证失败，请检查系统时间、代理或受信任证书。"):
        super().__init__(message)


class GeminiPageState(str, Enum):
    STARTING = "starting"
    PAGE_LOADING = "page_loading"
    WAITING_LOGIN = "waiting_login"
    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


def _sanitized_url(url: str) -> str:
    parts = urlsplit(str(url or ""))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


@dataclass(frozen=True)
class LoginStatus:
    pid: int | None
    state: GeminiPageState
    ready: bool
    url: str
    language: str
    message: str
    updated_at: float

    @classmethod
    def create(
        cls,
        state: GeminiPageState,
        ready: bool,
        url: str,
        language: str,
        message: str,
        pid: int | None = None,
    ) -> "LoginStatus":
        return cls(
            pid=os.getpid() if pid is None else pid,
            state=GeminiPageState(state),
            ready=bool(ready),
            url=_sanitized_url(url),
            language=str(language or ""),
            message=str(message or ""),
            updated_at=time.time(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "state": self.state.value,
            "ready": self.ready,
            "url": _sanitized_url(self.url),
            "language": self.language,
            "message": self.message,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LoginStatus":
        return cls(
            pid=int(value["pid"]) if value.get("pid") is not None else None,
            state=GeminiPageState(str(value["state"])),
            ready=bool(value["ready"]),
            url=_sanitized_url(str(value.get("url", ""))),
            language=str(value.get("language", "")),
            message=str(value.get("message", "")),
            updated_at=float(value["updated_at"]),
        )


@dataclass(frozen=True)
class LoginRuntimePaths:
    status_path: Path
    close_request_path: Path
    owner_lock_path: Path


@dataclass(frozen=True)
class LoginHelperOwner:
    pid: int
    token: str


def login_runtime_paths(config_path: str | Path) -> LoginRuntimePaths:
    root = Path(config_path).parent / "runs" / "gemini_login"
    config = _read_helper_config(config_path)
    browser_cfg = config.get("browser", {})
    browser_cfg = browser_cfg if isinstance(browser_cfg, Mapping) else {}
    profile = resolve_user_data_dir(browser_cfg, config_path)
    return LoginRuntimePaths(
        root / "status.json",
        root / "close.request",
        profile / ".gemini_login_helper.owner",
    )


_PAGE_INSPECTION_SCRIPT = r"""
() => {
  const visible = (node) => {
    if (!node) return false;
    const style = window.getComputedStyle(node);
    const rect = node.getBoundingClientRect();
    return style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0;
  };
  const normalize = (value) => (value || '').normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '').toLowerCase();
  const text = normalize(document.body ? document.body.innerText : '');
  const hasVisible = (selector) => Array.from(document.querySelectorAll(selector)).some(visible);
  const controls = Array.from(document.querySelectorAll('button, [role=button], [role=menuitem]'))
    .filter(visible).map((node) => (node.innerText || node.getAttribute('aria-label') || '').trim())
    .filter(Boolean).slice(0, 30);
  return {
    language: document.documentElement.lang || navigator.language || '',
    has_editor: hasVisible('textarea, [contenteditable=true], [role=textbox]'),
    has_login_prompt: /sign in|log in|login|iniciar sesion|verificar|verification|cuenta|account|\u767b\u5f55|\u767b\u5165|\u9a8c\u8bc1|\u9a57\u8b49/.test(text),
    has_loading: /loading|cargando|\u52a0\u8f7d\u4e2d|offline|sin conexion|\u79bb\u7ebf/.test(text)
      || hasVisible('[role=progressbar], mat-progress-bar'),
    controls,
  };
}
"""


def _status_for_page(
    state: GeminiPageState,
    ready: bool,
    page: Any,
    payload: Mapping[str, object] | None = None,
    message: str = "",
) -> LoginStatus:
    payload = payload or {}
    return LoginStatus.create(
        state,
        ready,
        getattr(page, "url", ""),
        str(payload.get("language", "")),
        message,
    )


def inspect_gemini_page(page: Any) -> LoginStatus:
    """Classify the visible Gemini page without reading account data."""
    try:
        raw_payload = page.evaluate(_PAGE_INSPECTION_SCRIPT)
    except Exception:
        return _status_for_page(
            GeminiPageState.ERROR,
            False,
            page,
            message="Unable to inspect the Gemini page.",
        )

    payload = raw_payload if isinstance(raw_payload, Mapping) else {}
    url_parts = urlsplit(_sanitized_url(str(getattr(page, "url", ""))))
    hostname = (url_parts.hostname or "").lower()
    path_segments = {segment.lower() for segment in url_parts.path.split("/") if segment}
    is_account_or_login_url = (
        hostname in {"accounts.google.com", "myaccount.google.com", "account.google.com"}
        or bool(path_segments & {"login", "signin", "verify", "verification", "challenge"})
    )
    has_login_prompt = bool(payload.get("has_login_prompt"))
    if is_account_or_login_url or has_login_prompt:
        return _status_for_page(
            GeminiPageState.WAITING_LOGIN,
            False,
            page,
            payload,
            "Sign in to Gemini in the browser window.",
        )
    if bool(payload.get("has_editor")):
        return _status_for_page(
            GeminiPageState.READY,
            True,
            page,
            payload,
            "Gemini is ready.",
        )
    if bool(payload.get("has_loading")):
        return _status_for_page(
            GeminiPageState.PAGE_LOADING,
            False,
            page,
            payload,
            "Gemini is loading.",
        )
    return _status_for_page(
        GeminiPageState.PAGE_LOADING,
        False,
        page,
        payload,
        "Waiting for the Gemini page.",
    )


def _log_status(logger: Any, status: LoginStatus) -> None:
    if logger:
        logger.info(f"Gemini page state: {status.state.value}")


def wait_for_gemini_ready(
    page: Any, policy: RetryPolicy, logger: Any = None
) -> LoginStatus:
    deadline = time.monotonic() + policy.page_ready_timeout
    while True:
        status = inspect_gemini_page(page)
        _log_status(logger, status)
        if status.state in (GeminiPageState.READY, GeminiPageState.WAITING_LOGIN, GeminiPageState.ERROR):
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError("Gemini did not become ready before the configured timeout.")
        waiter = getattr(page, "wait_for_timeout", None)
        if callable(waiter):
            waiter(250)
        else:
            time.sleep(0.25)


def navigate_gemini_with_retry(
    page: Any, url: str, policy: RetryPolicy, logger: Any = None
) -> LoginStatus:
    def navigate() -> LoginStatus:
        page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=int(policy.page_ready_timeout * 1000),
        )
        return wait_for_gemini_ready(page, policy, logger=logger)

    def on_retry(message: str) -> None:
        if logger:
            logger.warning(message)

    try:
        return run_with_retry(navigate, policy, on_retry=on_retry)  # type: ignore[return-value]
    except Exception as exc:
        if classify_network_error(exc) is RetryKind.PERMANENT_TLS:
            raise GeminiPermanentTlsError() from exc
        raise


def write_login_status(path: str | Path, status: LoginStatus) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    try:
        temp.write_text(
            json.dumps(status.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def read_login_status(path: str | Path) -> LoginStatus | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return LoginStatus.from_dict(value) if isinstance(value, Mapping) else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def request_login_helper_close(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp")
    try:
        temp.write_text("close\n", encoding="utf-8")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def process_is_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED
            try:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return True
                return exit_code.value == 259  # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except OSError:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_login_helper_owner(path: str | Path) -> LoginHelperOwner | None:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        pid = int(value["pid"])
        token = str(value["token"])
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return LoginHelperOwner(pid=pid, token=token) if pid > 0 and token else None


def acquire_login_helper_owner(paths: LoginRuntimePaths) -> LoginHelperOwner | None:
    """Atomically claim a persistent browser profile without signalling any process."""
    lock_path = paths.owner_lock_path
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(2):
        owner = LoginHelperOwner(pid=os.getpid(), token=os.urandom(16).hex())
        try:
            descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            existing = _read_login_helper_owner(lock_path)
            if existing is None or process_is_alive(existing.pid):
                return None
            # Only a positively identified dead owner is recoverable. Re-read the
            # metadata before removal so a replacement owner is never cleaned up.
            if _read_login_helper_owner(lock_path) != existing:
                continue
            try:
                lock_path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                return None
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {"pid": owner.pid, "token": owner.token, "created_at": time.time()},
                handle,
                ensure_ascii=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        return owner
    return None


def release_login_helper_owner(paths: LoginRuntimePaths, owner: LoginHelperOwner) -> None:
    """Remove only the lock metadata created by this helper instance."""
    if _read_login_helper_owner(paths.owner_lock_path) != owner:
        return
    try:
        paths.owner_lock_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def login_helper_is_active(paths: LoginRuntimePaths) -> bool:
    owner = _read_login_helper_owner(paths.owner_lock_path)
    if owner is not None and process_is_alive(owner.pid):
        return True
    status = read_login_status(paths.status_path)
    return bool(status and process_is_alive(status.pid))


def clear_stale_login_runtime(paths: LoginRuntimePaths) -> bool:
    status = read_login_status(paths.status_path)
    if status is None or process_is_alive(status.pid):
        return False
    removed = False
    for path in (paths.status_path, paths.close_request_path):
        try:
            Path(path).unlink()
            removed = True
        except FileNotFoundError:
            pass
    return removed


def _default_browser_candidates() -> list[str]:
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    local_app_data = Path.home() / "AppData" / "Local"
    candidates.extend(
        [
            str(local_app_data / "Google" / "Chrome" / "Application" / "chrome.exe"),
            str(local_app_data / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
        ]
    )
    return candidates


def resolve_browser_executable(
    browser_cfg: Mapping[str, object], candidate_paths: list[str] | None = None
) -> str | None:
    configured = str(browser_cfg.get("chrome_exe", "") or "").strip()
    if configured and Path(configured).exists():
        return configured
    for candidate in candidate_paths if candidate_paths is not None else _default_browser_candidates():
        if candidate and Path(candidate).exists():
            return candidate
    return None


def resolve_user_data_dir(
    browser_cfg: Mapping[str, object], config_path: str | Path | None = None
) -> Path:
    configured = Path(str(browser_cfg.get("user_data_dir", "browser_profile") or "browser_profile"))
    if configured.is_absolute():
        return configured
    root = Path(config_path).parent if config_path is not None else Path.cwd()
    return root / configured


def build_browser_launch_options(
    config: Mapping[str, object],
    *,
    config_path: str | Path | None = None,
    candidate_paths: list[str] | None = None,
) -> dict[str, object]:
    browser_cfg = config.get("browser", config)
    browser_cfg = browser_cfg if isinstance(browser_cfg, Mapping) else {}
    launch_options: dict[str, object] = {
        "user_data_dir": str(resolve_user_data_dir(browser_cfg, config_path)),
        "headless": False,
        "no_viewport": True,
        "args": [
            "--disable-blink-features=AutomationControlled",
            "--disable-features=ImprovedCookieControls",
        ],
    }
    executable = resolve_browser_executable(browser_cfg, candidate_paths=candidate_paths)
    if executable:
        launch_options["executable_path"] = executable
    return launch_options


def build_login_helper_command(
    config_path: str | Path,
    *,
    executable: str | None = None,
    frozen: bool | None = None,
) -> list[str]:
    """Build the source or packaged entry point for the isolated login helper."""
    config = str(Path(config_path).resolve())
    is_frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    program = executable or sys.executable
    if is_frozen:
        return [program, "--gemini-login-helper", "--config", config]
    return [program, str(Path(__file__).with_name("app.py").resolve()), "--gemini-login-helper", "--config", config]


def _read_helper_config(config_path: str | Path) -> dict[str, object]:
    try:
        value = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_helper_status(
    paths: LoginRuntimePaths,
    state: GeminiPageState,
    ready: bool,
    message: str,
    *,
    page: Any = None,
) -> LoginStatus:
    return_status = LoginStatus.create(
        state,
        ready,
        getattr(page, "url", ""),
        "",
        message,
    )
    write_login_status(paths.status_path, return_status)
    return return_status


def run_login_helper(config_path: str | Path) -> int:
    """Own one configured Gemini profile until the ready helper is asked to close."""
    paths = login_runtime_paths(config_path)
    owner = acquire_login_helper_owner(paths)
    if owner is None:
        return 1
    context = None
    page = None
    try:
        clear_stale_login_runtime(paths)
        _write_helper_status(paths, GeminiPageState.STARTING, False, "Starting Gemini login helper.")
        config = _read_helper_config(config_path)
        policy = retry_policy_from_config(config)
        gemini_config = config.get("gemini", {})
        gemini_config = gemini_config if isinstance(gemini_config, Mapping) else {}
        target_url = str(gemini_config.get("base_url", "https://gemini.google.com") or "https://gemini.google.com")
        with sync_playwright() as playwright:
            launch_options = build_browser_launch_options(config, config_path=config_path)
            context = playwright.chromium.launch_persistent_context(**launch_options)
            pages = getattr(context, "pages", [])
            page = pages[0] if pages else context.new_page()
            status = navigate_gemini_with_retry(page, target_url, policy)
            write_login_status(paths.status_path, status)

            while True:
                if paths.close_request_path.exists() and status.ready:
                    _write_helper_status(paths, GeminiPageState.CLOSING, False, "Closing Gemini login helper.", page=page)
                    context.close()
                    context = None
                    try:
                        paths.close_request_path.unlink()
                    except FileNotFoundError:
                        pass
                    _write_helper_status(paths, GeminiPageState.CLOSED, False, "Gemini login helper closed.")
                    return 0

                is_closed = getattr(page, "is_closed", None)
                if callable(is_closed) and is_closed():
                    _write_helper_status(paths, GeminiPageState.CLOSED, False, "Gemini browser was closed.")
                    return 0

                status = inspect_gemini_page(page)
                write_login_status(paths.status_path, status)
                if status.state == GeminiPageState.ERROR:
                    return 1
                time.sleep(1)
    except Exception:
        _write_helper_status(paths, GeminiPageState.ERROR, False, "Gemini login helper stopped safely.", page=page)
        return 1
    finally:
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
        release_login_helper_owner(paths, owner)


__all__ = [
    "GeminiPageState",
    "GeminiLoginRequiredError",
    "GeminiPageNotReadyError",
    "GeminiPermanentTlsError",
    "LoginRuntimePaths",
    "LoginHelperOwner",
    "acquire_login_helper_owner",
    "LoginStatus",
    "build_browser_launch_options",
    "build_login_helper_command",
    "clear_stale_login_runtime",
    "inspect_gemini_page",
    "login_helper_is_active",
    "login_runtime_paths",
    "navigate_gemini_with_retry",
    "process_is_alive",
    "read_login_status",
    "release_login_helper_owner",
    "request_login_helper_close",
    "resolve_browser_executable",
    "resolve_user_data_dir",
    "retry_policy_from_config",
    "run_login_helper",
    "wait_for_gemini_ready",
    "write_login_status",
]
