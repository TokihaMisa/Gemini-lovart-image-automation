from dataclasses import dataclass


RETRY_MODE_OFF = "off"
RETRY_MODE_FINITE = "finite"
RETRY_MODE_INFINITE = "infinite"
RETRY_MODES = (RETRY_MODE_OFF, RETRY_MODE_FINITE, RETRY_MODE_INFINITE)

RETRY_ERROR_TYPE_CHOICES = (
    ("Lovart 服务不可用或 API 异常", "lovart_service"),
    ("网络连接或 SSL 失败", "network"),
    ("请求或页面等待超时", "timeout"),
    ("Gemini 页面控件或加载失败", "gemini_page"),
    ("Gemini 图片上传失败", "gemini_upload"),
    ("Lovart 完成但未返回图片", "lovart_no_artifacts"),
    ("其他未知错误", "other"),
)
RETRY_ERROR_TYPES = tuple(value for _label, value in RETRY_ERROR_TYPE_CHOICES)
DEFAULT_RETRY_ERROR_TYPES = (
    "lovart_service",
    "network",
    "timeout",
    "gemini_page",
    "gemini_upload",
)

_PERMANENT_ERROR_MARKERS = (
    "invalid access key",
    "invalid secret key",
    "api key authentication failed",
    "unauthorized",
    "未登录",
    "请先登录",
    "pending confirmation",
    "manual action",
    "needs_manual_action",
    "已停止自动重试",
    "no main product image",
    "没有商品主图",
)

_CATEGORY_MARKERS = (
    (
        "lovart_no_artifacts",
        (
            "without returning image artifacts",
            "完成但未返回图片",
            "未返回图片产物",
        ),
    ),
    (
        "gemini_upload",
        (
            "gemini image upload did not complete",
            "gemini 图片上传",
            "gemini upload",
        ),
    ),
    (
        "gemini_page",
        (
            "gemini page structure changed",
            "gemini page did not finish loading",
            "gemini temporary chat control not found",
            "gemini 页面结构",
            "gemini 页面未完成加载",
            "gemini 临时聊天",
        ),
    ),
    (
        "timeout",
        (
            "timed out",
            "timeout",
            "超时",
        ),
    ),
    (
        "lovart_service",
        (
            "lovart 服务暂时不可用",
            "无法连接 lovart",
            "lovart api 失败",
            "lovart service unavailable",
            "unknown api error",
        ),
    ),
    (
        "network",
        (
            "connection failed",
            "temporarily unavailable",
            "connection reset",
            "network",
            "ssl/tls connection failed",
            "failed to fetch",
            "网络连接",
            "连接被重置",
        ),
    ),
)

_PERMANENT_FAILURE_CODES = frozenset(
    {
        "ambiguous_submission",
        "task_still_running",
        "authentication",
        "missing_key",
        "invalid_base_url",
        "invalid_request",
        "missing_input_image",
        "reference_image_too_large",
        "reference_total_too_large",
        "create_body_too_large",
        "tls_certificate",
    }
)

_FAILURE_CODE_CATEGORIES = {
    "network": "network",
    "server_error": "network",
    "rate_limit": "network",
    "timeout": "timeout",
    "task_failed": "other",
}


def normalize_retry_mode(value) -> str:
    mode = str(value or RETRY_MODE_FINITE).strip().lower()
    if mode not in RETRY_MODES:
        raise ValueError("失败任务重试模式无效")
    return mode


def normalize_retry_rounds(value) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("补偿轮次必须是整数") from exc
    if not number.is_integer() or number < 1:
        raise ValueError("有限重试的补偿轮次必须是大于或等于 1 的整数")
    return int(number)


def normalize_retry_delay(value) -> float:
    try:
        delay = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("重试间隔必须是数字") from exc
    if delay < 0:
        raise ValueError("重试间隔不能小于 0 秒")
    return delay


def normalize_retry_error_types(values) -> tuple[str, ...]:
    if values is None:
        values = DEFAULT_RETRY_ERROR_TYPES
    if isinstance(values, str):
        values = [values]
    unknown = [value for value in values if value not in RETRY_ERROR_TYPES]
    if unknown:
        raise ValueError(f"存在未知的重试错误类型：{', '.join(map(str, unknown))}")
    selected = set(values)
    return tuple(value for value in RETRY_ERROR_TYPES if value in selected)


@dataclass(frozen=True)
class FailedRetryPolicy:
    mode: str = RETRY_MODE_FINITE
    rounds: int = 2
    delay: float = 15.0
    error_types: tuple[str, ...] = DEFAULT_RETRY_ERROR_TYPES

    @classmethod
    def from_config(cls, lovart_config: dict | None):
        config = lovart_config if isinstance(lovart_config, dict) else {}
        mode = normalize_retry_mode(config.get("failed_retry_mode", RETRY_MODE_FINITE))
        rounds = normalize_retry_rounds(config.get("failed_retry_rounds", 2))
        delay = normalize_retry_delay(config.get("failed_retry_delay", 15))
        error_types = normalize_retry_error_types(config.get("failed_retry_error_types"))
        return cls(mode=mode, rounds=rounds, delay=delay, error_types=error_types)

    @property
    def enabled(self) -> bool:
        return self.mode != RETRY_MODE_OFF and bool(self.error_types)

    @property
    def infinite(self) -> bool:
        return self.mode == RETRY_MODE_INFINITE


def classify_retry_failure(row: dict) -> str | None:
    if row.get("status") != "failed":
        return None
    failure_code = str(row.get("failure_code") or "").strip().casefold()
    if failure_code:
        if failure_code in _PERMANENT_FAILURE_CODES:
            return None
        return _FAILURE_CODE_CATEGORIES.get(failure_code, "other")
    error = str(row.get("error") or "").casefold()
    if any(marker in error for marker in _PERMANENT_ERROR_MARKERS):
        return None
    for category, markers in _CATEGORY_MARKERS:
        if any(marker in error for marker in markers):
            return category
    return "other"
