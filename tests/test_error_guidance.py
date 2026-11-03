import pytest

from error_guidance import explain_failure


@pytest.mark.parametrize(
    ("raw", "cause"),
    [
        (
            "screen 1: GPT Image 异步任务提交后未收到可验证任务 ID；为避免重复扣费，已停止自动重试。",
            "提交结果没能确认",
        ),
        ("GPT Image task failed. Provider message: Request body read timed out", "平台读取上传内容超时"),
        ("GPT Image API is rate limiting requests (HTTP 429).", "平台限流"),
        ("GPT Image API authentication failed (HTTP 401).", "密钥无效或没有权限"),
        ("GPT Image API is temporarily unavailable (HTTP 503).", "平台暂时不可用"),
        ("Could not reach GPT Image API.", "连不上出图平台"),
        ("Gemini Thinking mode could not be selected", "Gemini 页面控件没找到"),
        ("Detail screen 1 is empty", "Gemini 返回了空内容"),
    ],
)
def test_known_failures_get_a_chinese_cause_and_next_step(raw, cause):
    guidance = explain_failure(raw)

    assert guidance is not None
    assert guidance.cause == cause
    assert guidance.action


def test_unknown_failure_has_no_guidance():
    assert explain_failure("something nobody has seen before") is None
