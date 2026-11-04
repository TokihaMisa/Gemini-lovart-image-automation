"""Plain-language cause and next step for failure messages shown to colleagues.

Raw provider/engine messages stay in the logs; this table only adds a Chinese
explanation. Order matters: specific markers come before generic ones.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureGuidance:
    cause: str
    action: str


_GUIDANCE: tuple[tuple[tuple[str, ...], FailureGuidance], ...] = (
    (
        ("未收到可验证任务 id", "未收到可验证图片", "no validated task id", "submission_unknown"),
        FailureGuidance(
            "提交结果没能确认",
            "请到出图平台后台查看这张图是否已扣费：没有扣费就重新开始任务；已扣费请等平台出图，不要重复提交。",
        ),
    ),
    (
        ("request body read timed out",),
        FailureGuidance(
            "平台读取上传内容超时",
            "通常是网络上传慢或参考图较大，系统会按失败重试设置自动重试；反复出现请换个网络或减少参考图。",
        ),
    ),
    (
        ("rate limiting", "http 429"),
        FailureGuidance("平台限流", "请求太频繁，系统会稍后自动重试；经常出现请减少同时运行的任务。"),
    ),
    (
        ("authentication failed", "http 401", "http 403", "invalid access key", "invalid secret key", "unauthorized"),
        FailureGuidance("密钥无效或没有权限", "请到「API 与模型」页检查密钥是否填对、是否过期、账户余额是否充足。"),
    ),
    (
        ("overloaded", "服务器过载"),
        FailureGuidance("平台繁忙", "系统会按失败重试设置自动重试，无需操作。"),
    ),
    (
        ("temporarily unavailable", "服务暂时不可用"),
        FailureGuidance("平台暂时不可用", "系统会按失败重试设置自动重试；长时间出现请联系平台。"),
    ),
    (
        ("certificate", "tls", "证书"),
        FailureGuidance("安全证书校验失败", "多半是代理或抓包软件干扰，请关闭后重试；不要关闭证书校验。"),
    ),
    (
        ("could not reach", "无法连接", "connection reset", "connection failed", "网络连接"),
        FailureGuidance("连不上出图平台", "请检查网络、代理或 VPN 后重新开始任务。"),
    ),
    (
        ("chatgpt 未登录",),
        FailureGuidance("ChatGPT 未登录", "请在「API 与模型」页点击「打开 ChatGPT 登录浏览器」登录后再开始。"),
    ),
    (
        ("人机验证", "没有进入临时聊天"),
        FailureGuidance(
            "ChatGPT 页面没准备好",
            "请在「API 与模型」页打开 ChatGPT 登录浏览器，完成验证、确认能进入临时聊天后再开始。",
        ),
    ),
    (
        ("chatgpt 回复过短", "可能拒绝了请求"),
        FailureGuidance("ChatGPT 拒绝了这个商品", "可以把提示词来源换成「GPT API」或其他来源后重新开始。"),
    ),
    (
        ("gemini requires login", "未登录", "请先登录"),
        FailureGuidance("Gemini 未登录", "请在「API 与模型」页点击「打开 Gemini 登录浏览器」登录后再开始。"),
    ),
    (
        ("thinking mode could not be selected", "mode menu", "temporary chat", "页面控件", "page structure changed"),
        FailureGuidance(
            "Gemini 页面控件没找到",
            "请先在「API 与模型」页运行「Gemini 一键完整体检」；如果是选不上模型或思考等级，"
            "可以把「Gemini 档位」改成「按我在 Gemini 里选好的档位」，在登录浏览器里亲手选好后点「读取当前档位并保存」。",
        ),
    ),
    (
        ("is empty", "返回为空", "空内容"),
        FailureGuidance("Gemini 返回了空内容", "这是 Gemini 那边的问题，稍后重新开始任务即可。"),
    ),
    (
        ("timed out", "timeout", "超时"),
        FailureGuidance("等待超时", "系统会按失败重试设置自动重试；反复出现请检查网络。"),
    ),
)


def explain_failure(reason: object) -> FailureGuidance | None:
    text = str(reason or "").casefold()
    for markers, guidance in _GUIDANCE:
        if any(marker in text for marker in markers):
            return guidance
    return None
