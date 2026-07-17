from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
import socket
import ssl
import time
from urllib.error import HTTPError, URLError


class RetryKind(str, Enum):
    TRANSIENT = "transient"
    PERMANENT_TLS = "permanent_tls"
    AUTH = "auth"
    NOT_FOUND = "not_found"
    OTHER = "other"


@dataclass(frozen=True)
class RetryPolicy:
    network_attempts: int = 5
    page_ready_timeout: float = 90.0
    product_attempts: int = 2
    retry_delays: tuple[float, ...] = (3.0, 6.0, 12.0, 20.0)

    def delay_after(self, failed_attempt: int) -> float:
        index = min(max(failed_attempt - 1, 0), len(self.retry_delays) - 1)
        return self.retry_delays[index]


def retry_policy_from_config(config: Mapping[str, object]) -> RetryPolicy:
    browser = config.get("browser", {}) if isinstance(config, Mapping) else {}
    browser = browser if isinstance(browser, Mapping) else {}
    delays = tuple(float(value) for value in browser.get("retry_delays", (3, 6, 12, 20)))
    return RetryPolicy(
        network_attempts=max(1, int(browser.get("network_attempts", 5))),
        page_ready_timeout=max(1.0, float(browser.get("page_ready_timeout", 90))),
        product_attempts=max(1, int(browser.get("product_attempts", 2))),
        retry_delays=delays or (3.0, 6.0, 12.0, 20.0),
    )


_PERMANENT_TLS_MARKERS = (
    "certificate verify failed",
    "net::err_cert_authority_invalid",
    "net::err_cert_common_name_invalid",
    "net::err_cert_date_invalid",
)


def classify_network_error(exc: BaseException) -> RetryKind:
    message = str(exc).lower()
    if isinstance(exc, ssl.SSLCertVerificationError) or any(
        marker in message for marker in _PERMANENT_TLS_MARKERS
    ):
        return RetryKind.PERMANENT_TLS

    if isinstance(exc, ssl.SSLError):
        return RetryKind.TRANSIENT

    if isinstance(exc, HTTPError):
        if exc.code in (401, 403):
            return RetryKind.AUTH
        if exc.code == 404:
            return RetryKind.NOT_FOUND
        if exc.code in (408, 429) or 500 <= exc.code < 600:
            return RetryKind.TRANSIENT
        return RetryKind.OTHER

    if isinstance(exc, (TimeoutError, socket.timeout, ConnectionError, URLError)):
        return RetryKind.TRANSIENT

    if "net::err_" in message:
        return RetryKind.TRANSIENT

    return RetryKind.OTHER


def safe_retry_message(kind: RetryKind, attempt: int, attempts: int, delay: float) -> str:
    return f"{kind.value}：第 {attempt}/{attempts} 次失败，{delay:g} 秒后重试。"


def run_with_retry(
    operation: Callable[[], object],
    policy: RetryPolicy,
    *,
    on_retry: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> object:
    sleeper = sleep if sleep is not None else time.sleep
    for attempt in range(1, policy.network_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            kind = classify_network_error(exc)
            if kind is not RetryKind.TRANSIENT or attempt >= policy.network_attempts:
                raise
            delay = policy.delay_after(attempt)
            if on_retry:
                on_retry(safe_retry_message(kind, attempt, policy.network_attempts, delay))
            sleeper(delay)
    raise RuntimeError("retry loop exhausted")
