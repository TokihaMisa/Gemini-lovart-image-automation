# LK888 异步 GPT Image API 专用模式设计

## 背景

当前 `openai_image` 传输层同时包含 OpenAI Images、HAPI 专用异步端点、同步回退和 LK888 的少量字段适配。实际运行表明，`api.lk888.ai` 的同步兼容端点可能已经在服务端完成并扣费，却迟迟不结束 HTTP 响应，导致本地界面停留在旧阶段，也无法取得结果。

LK888 的 GPT Image 2 文档提供了完整的异步任务协议：创建任务后返回 `task_id`，客户端轮询任务状态，最终从 `result_url` 下载图片。参考图支持公网 URL 或 `data:<mime>;base64,<data>` 内联数据。因此本地 Excel 图片不需要外部图床。

## 目标

- 将 GPT Image 传输层收窄为 LK888 异步媒体任务协议。
- 保留可配置 Base URL，为以后使用相同协议的其他网关留出空间。
- 创建付费任务只提交一次；取得 `task_id` 后只重试安全的查询请求。
- 白底图、场景图和逐屏详情图均支持持久化任务、实时进度和跨进程恢复。
- 删除 HAPI 专用识别、同步 `/images/edits` 请求和自动回退。

## 非目标

- 不同时维护 OpenAI、HAPI、LK888 多种协议选择器。
- 不自动探测端点或在异步失败后回退到同步付费请求。
- 不为参考图引入外部图床。
- 不反查或认领没有 `task_id` 的历史同步任务。
- 不改变 Gemini 提示词生成、商品详情屏解析或 Lovart 路由。

## 方案选择

采用“可配置 Base URL + 单一 LK888 异步媒体协议”。

未采用以下方案：

- 多协议设置：会继续扩大设置、测试和运行分支，不符合当前只维护一种图像 API 协议的要求。
- 自动探测和同步回退：端点探测存在重复创建付费任务的风险。
- 固定 `api.lk888.ai`：用户希望未来可以填写采用相同协议的其他网关。

## 配置和界面

保留以下设置：

- `OPENAI_IMAGE_API_KEY` 环境变量中的 API Key。
- 可编辑 Base URL。
- 可编辑模型，默认 `gpt-image-2`。
- 1K、2K、4K 分辨率档位。
- “将多张参考图合并为一张上传”开关。

Base URL 允许带或不带末尾 `/v1`。端点构造器只移除一个末尾 `/v1`，随后拼接文档端点，避免产生 `/v1/v1`：

- `POST {normalized_base}/v1/media/generate`
- `GET {normalized_base}/v1/media/status?task_id={task_id}`

设置说明和付费测试文案改为“异步媒体任务协议”。删除 HAPI 名称、HAPI 默认合并行为和 OpenAI Images `/images/edits` 说明。API 测试按钮也必须通过创建任务、轮询和结果下载完成。

## 创建任务请求

请求使用 `Authorization: Bearer <API key>` 和 `Content-Type: application/json`。请求体为：

```json
{
  "model": "gpt-image-2",
  "prompt": "...",
  "params": {
    "images": [
      "data:image/png;base64,..."
    ],
    "size": "1024x1536",
    "quality": "auto",
    "n": 1
  }
}
```

`size` 沿用现有的图片比例和 1K／2K／4K 映射逻辑，传递文档允许的精确像素尺寸。每个提示词固定 `n=1`；详情套图仍是一屏一个任务，以保留逐屏断点语义。

创建响应接受根对象或 `data` 对象中的字符串／整数 `task_id`。在没有 `task_id` 的情况下不得轮询，也不得自动重新提交。

## 参考图编码和限制

关闭合并开关时，每个本地文件根据实际解码格式生成独立的 `data:<mime>;base64,<data>`。开启开关时先调用现有本地参考拼图逻辑，再编码为一个 Data URL。

创建付费任务前必须完成所有限制检查：

- 最多 14 张参考图。
- 单张解码数据默认不超过 10MB。
- 全部 Base64 图片解码数据合计默认不超过 30MB。
- UTF-8 JSON 请求体默认不超过 50MB。

关闭合并开关且超限时，任务在本地失败，并明确提示用户开启合并或压缩原图；程序不得擅自合并。开启合并后仍超限时同样本地失败。限制失败不得发起网络请求。

只接受 Pillow 可以完整解码的 PNG、JPEG 或 WebP 参考图，MIME 由实际格式决定，不能只相信文件扩展名。

## 轮询状态机

取得 `task_id` 后立即持久化，再发起第一次查询。轮询节奏遵循文档建议：

- 已等待不超过 120 秒：每 5 秒查询一次。
- 已等待超过 120 秒：每 10 秒查询一次。
- 本轮本地等待上限：600 秒。

任务状态以 `is_final` 和 `state` 为逻辑字段；中文 `status`、`status_group` 仅作为界面展示，不参与完成判断。

允许的状态流程：

- `is_final=false` 且 `state` 为 `pending` 或 `running`：继续轮询。
- `is_final=true` 且 `state=success`：要求非空 `result_url`，进入安全下载。
- `is_final=true` 且 `state=failed`：记录 `error`，该任务终止。
- 其他组合或缺失关键字段：协议响应错误，不创建新任务。

WebUI 显示服务端 `progress`、展示状态、已等待时间和任务 ID 的脱敏尾号。轮询 GET 的网络超时、429 和 5xx可以按现有退避策略重试，但不得改变 `task_id`。

## 持久化和恢复

任务身份至少包含：

- 规范化 Base URL。
- 模型。
- 精确尺寸。
- 合并设置。
- 上游输入指纹。
- 最终付费提示词哈希。
- `task_id`。
- 创建时间和最近一次服务端状态。

详情图复用现有 `detail_checkpoints`，为 `running` 状态增加 `task_id`、`task_created_at`、`progress` 和服务端状态字段。支持图在 `status.json` 中新增按 `white_bg`／`scene` 分开的 `support_task_checkpoints`。

恢复规则：

- 身份完全匹配且有 `task_id`：只恢复轮询。
- 任务成功且本地图片有效：直接复用图片。
- 任务成功但本地文件缺失或损坏：使用保存的 `result_url` 重新下载，不重新创建任务。
- 服务端最终失败：按现有失败重试策略决定是否在下一轮创建新任务。
- 本地 600 秒超时但任务仍未终止：保留 running 断点，当前商品记为仍在运行；下一次只继续轮询。
- 旧同步断点没有 `task_id`：清理一次并创建新的异步任务。
- Base URL、模型、尺寸、提示词或上游参考图变化：旧任务不得复用；新任务创建前清理旧的 running 身份和陈旧本地目标文件。
- `--no-resume`：明确忽略旧任务并创建新任务，但仍保证本次 POST 只发送一次。

## 付费和错误安全

创建任务 POST 没有自动网络重试。即使发生本地连接超时，也可能已经被服务端接收；没有拿到 `task_id` 时返回“提交结果未知”，现有无限失败重试必须把它视为永久停止条件。

明确 HTTP 4xx、响应 JSON 损坏或缺少 `task_id` 也不自动重发创建请求。用户可以检查平台后台后手动决定是否重试。

取得 `task_id` 后，查询 GET 可以安全重试。最终 `state=failed` 按文档属于已结束任务，保留服务端错误和费用信息；是否创建新任务由商品级失败重试策略控制。

`result_url` 下载继续使用现有 SSRF 防护：单次 DNS 快照、只允许公网地址、连接到固定 IP、TLS 证书和 SNI 验证、对端 IP 校验、禁止重定向、响应 MIME 与 Pillow 完整解码检查，以及原子保存。下载重试不会创建新任务。

## HAPI 和同步协议移除

删除或停止使用：

- HAPI 主机识别和端点拼接。
- `/images/edits/async`、`/images/tasks/...` 和 HAPI 同步回退。
- 通用 `/images/edits` multipart 创建请求。
- `async_edits` 配置开关和 HAPI 默认合并规则。
- “OpenAI-compatible Images edits”相关设置文案。

Base URL 不做 LK888 域名白名单；其他网关只有在实现相同 JSON 创建、`task_id` 轮询和 `result_url` 响应协议时才兼容。

## 测试策略

所有生产改动遵循测试先行的 RED→GREEN 循环。

传输层测试覆盖：

- Base URL 带／不带 `/v1` 的精确端点。
- JSON 结构、授权头、模型、提示词、尺寸、`quality=auto` 和 `n=1`。
- PNG／JPEG／WebP Data URL 与实际 MIME。
- 14 张、单张 10MB、合计 30MB、请求体 50MB 边界，以及限制失败时 POST 次数为零。
- 根对象和 `data` 对象中的 `task_id`。
- POST 永不自动重试。
- 5 秒转 10 秒轮询节奏。
- running、success、failed、600 秒超时、缺字段和非法状态组合。
- 安全结果下载和已有 SSRF 回归。

管线和恢复测试覆盖：

- 白底图／场景图任务 ID 持久化和重启恢复。
- 详情图逐屏任务 ID、部分成功、重启补齐和动态目标数量。
- 运行中任务恢复时 POST 次数为零。
- 成功但本地文件丢失时只重新下载。
- 输入指纹变化和 `--no-resume` 创建新任务。
- 本地轮询超时不进入无限商品重试。
- 旧无任务 ID 断点的一次性迁移。

WebUI 和配置测试覆盖：

- Base URL 仍可编辑并可带／不带 `/v1`。
- 删除 HAPI 文案和行为。
- API 测试按钮显示创建、轮询进度和结果。
- 商品卡片显示任务进度、已等待时间和最终状态。
- API Key 继续只保存在 `.env`，不写入配置文件或日志。

最终验证包括聚焦测试、完整测试集、Python 编译检查、打包后 EXE `--help` 自检和 OTA ZIP 完整性校验。

## 发布要求

该变更作为 `v1.3.21` 发布。Release 创建前必须确保：

- 所有测试通过。
- 旧版本安装包和现有用户文件未被覆盖。
- 远端版本清单中的大小和 SHA-256 与实际 `update.zip` 一致。
- Git 标签、Release 资产和 `master` 版本清单指向同一提交。
