# GPT Image API 生图模式设计

## 背景与目标

当前流程依次使用 Lovart 生成白底图和场景图，再由用户选择的提示词来源生成详情页方案，最后由 Lovart 生成完整套图。本设计增加一个 OpenAI Images API 兼容通道，使用户可以分别决定：

1. 白底图和场景图使用 Lovart 还是 GPT Image；
2. 最终套图使用 Lovart 还是 GPT Image。

提示词来源保持不变，仍支持 Gemini 浏览器、Gemini API 和 NVIDIA。三个图片阶段全部选择 GPT Image 时，任务不依赖 Lovart 凭据，也不创建 Lovart 项目。

HAPI 的公开接入文档确认 OpenAI SDK Base URL 为 `https://hapiopen.cc/v1`，公开模型广场列出了 `gpt-image-2`，但没有公开图片编辑端点的请求与响应细节。因此实现以标准 OpenAI Images API 为边界，并提供用户主动触发的真实图生图测试来验证网关兼容性。

## 范围

本次包含：

- OpenAI 兼容 GPT Image 客户端；
- 白底/场景和最终套图的独立 provider 选择；
- GPT Image 设置、密钥保存和付费测试入口；
- 动态套图数量、逐屏生成、检查点与恢复；
- 旧 Lovart 状态与中间图兼容；
- CLI、WebUI、日志、结果汇总和测试更新。

本次不包含：

- 用户自定义任意请求模板；
- HAPI 私有协议或未公开端点；
- 自动从一个付费 provider 切换到另一个；
- 并行生成多屏图片；
- 修改提示词来源的选择逻辑。

## 方案选择

采用“标准 OpenAI Images API 适配器 + 主动兼容性测试”。

不硬编码 HAPI 私有实现，避免供应商锁定；也不提供任意端点和请求模板，避免设置复杂度、不可测试行为和密钥泄露风险。标准适配器使用 Bearer 鉴权，Base URL 以 `/v1` 结尾，带参考图的任务调用 `/images/edits`。

## 架构

### Provider 边界

新增中性的图片生成接口，主流程不再直接把所有阶段绑定到 `LovartBot`。接口至少提供：

- `generate_support_image(...)`：生成一张白底图或场景图；
- `generate_detail_image(...)`：生成一张指定序号的详情图；
- `validate_configuration()`：执行不产生图片费用的本地配置校验；
- `test_image_edit(...)`：由用户主动调用的一次真实图生图测试。

`LovartImageProvider` 包装现有 `LovartBot` 行为，保持项目、轮询、确认和 artifact 下载逻辑。`OpenAIImageProvider` 负责 multipart 请求、重试、响应解析、图片校验和本地落盘。

Provider 必须延迟初始化：只有实际阶段选择 Lovart 时才读取 Lovart 凭据并创建/验证 project；只有选择 GPT Image 时才读取其密钥。

### 运行路由

配置保存两个独立选择：

- `support_provider`: `lovart` 或 `openai_image`；
- `detail_provider`: `lovart` 或 `openai_image`。

组合行为：

| 白底/场景 | 最终套图 | 行为 |
|---|---|---|
| Lovart | Lovart | 完全保持现有链路 |
| GPT Image | Lovart | GPT Image 生成中间图；提示词完成后才创建 Lovart project |
| Lovart | GPT Image | Lovart 仅生成中间图；最终结果保存到 GPT Image 输出目录 |
| GPT Image | GPT Image | 完全不连接 Lovart |

## 数据流

### 白底图与场景图

1. 解析 Excel 商品图片角色。
2. 根据 `support_provider` 生成或复用白底图。
3. 使用白底图作为参考生成或复用场景图。
4. 两张中间图与可选参考拼图交给所选提示词来源。

GPT Image 路径的两个步骤均使用图片编辑语义，不能在 `/images/edits` 不可用时降级为无参考图的纯文字生成。白底图或场景图失败后停止当前商品，避免继续产生提示词或最终图片费用。

### 动态最终套图

最终张数读取任务启动时的 `prompt_settings.detail_page_count`。`12` 只是默认值，不是固定规则。

任务启动时把目标数量写入 `detail_page_count_snapshot`。同一任务和后续恢复始终使用该快照；运行中修改全局设置不会改变已启动商品的目标数量。

提示词规则增加稳定的屏级分隔标记，同时兼容旧的“屏 01”至“屏 NN”标题。解析结果必须恰好等于快照数量。数量不符时，在调用任何最终图片 API 前停止并写出诊断信息。

GPT Image 最终阶段按屏顺序串行生成：

1. 组合全局商品约束、当前屏提示词和参考图说明；
2. 上传白底图、场景图、配件图、尺寸图和参考拼图中实际存在的文件；
3. 调用 `/images/edits` 生成当前屏；
4. 校验并原子保存图片；
5. 更新当前屏检查点后再处理下一屏。

串行执行优先保证成本可控、日志清晰和恢复确定性。本次不增加屏级并发。

## 配置与 WebUI

### 本地配置

建议配置结构：

```yaml
image_generation:
  support_provider: lovart
  detail_provider: lovart

openai_image:
  base_url: https://hapiopen.cc/v1
  model: gpt-image-2
  resolution: 1K
  timeout: 600
  max_attempts: 4
  retry_delays: [3, 6, 12]
```

密钥使用 `OPENAI_IMAGE_API_KEY`，仅保存到本地 `.env`。示例文件只写占位符。

Base URL 规范化规则：

- 去除首尾空白和末尾 `/`；
- 只允许包含主机名的 HTTP/HTTPS URL；
- 纯主机地址自动补 `/v1`；
- 已以 `/v1` 结尾时保持不变；
- 防止形成 `/v1/v1`；
- 包含换行、无主机名或不支持 scheme 时拒绝保存。

### 设置界面

“API 与模型设置”新增“GPT Image（OpenAI 兼容格式）”：

- API 地址；
- API 密钥密码框；
- 模型名，默认 `gpt-image-2`；
- 分辨率：1K、2K、4K；
- “测试图生图”按钮，并明确提示可能产生一次图片费用。

密钥输入留空表示保留已有值；清除密钥必须是明确操作。保存继续使用现有 `config.yaml` 与 `.env` 补偿式事务，任何一步失败都回滚两份文件。

“运行任务”区域新增白底/场景 provider 和最终套图 provider。选择会保存为下次默认值，但每次启动前可以修改。启动前只校验本次实际选择的 provider。

## API 请求与响应

`OpenAIImageProvider` 使用：

- `Authorization: Bearer <key>`；
- `POST {base_url}/images/edits`；
- multipart 表单包含模型、提示词、分辨率/尺寸参数和一张或多张参考图。

客户端兼容两类结果：

- `data[].b64_json`：解码后保存；
- `data[].url`：下载后保存。

下载结果必须验证状态码、Content-Type、非空内容及图片可解码性。响应 URL 只允许 HTTP/HTTPS，并拒绝本机、环回和私有网段目标，降低恶意或错误响应引发本地网络访问的风险。

如果 HAPI 对标准字段或端点不兼容，“测试图生图”返回清晰错误，不尝试猜测私有协议。保存设置本身不产生图片费用。

## 状态、输出与兼容

### 新输出目录

建议使用中性目录：

```text
output/<商品ID>/image_steps/white_bg/
output/<商品ID>/image_steps/scene/
output/<商品ID>/gpt_image/detail/01.png
output/<商品ID>/gpt_image/detail/02.png
...
```

Lovart 原有目录保持不变。查找可复用中间图时，先读取新中性字段，再兼容旧 `lovart_white_bg_local_path`、`lovart_scene_local_path` 和 `lovart_steps/*`。

### 状态字段

新增或统一记录：

- `support_provider`；
- `detail_provider`；
- `detail_page_count_snapshot`；
- `white_bg_local_path`；
- `scene_local_path`；
- `detail_images`，按屏号记录路径、状态、尝试次数和错误；
- `detail_completed_count`；
- `detail_generation_complete`；
- `partial_complete`。

Lovart 路径继续双写必要的旧字段，保证已有 WebUI、结果汇总和 v1.3.x 恢复逻辑可用。旧商品没有数量快照时，从已有提示词设置和已生成状态推导一次并写入；不得在每次恢复时重新读取可能已经变化的全局数量。

### 成功与恢复

- 只有快照要求的全部屏都存在且通过图片校验时，GPT Image 最终阶段才成功。
- 某屏失败时保留所有成功图片，商品标记为部分完成。
- 恢复时只生成缺失、失败或校验不通过的屏，不覆盖已经成功的文件。
- 不自动切换 provider，不因重试重复生成已经成功的付费图片。

## 错误处理与重试

- `401/403`：密钥、分组或权限错误，立即失败，不重试；
- `400/404`：请求字段、模型或端点不兼容，立即失败并提示检查模型/网关协议；
- `429`、可恢复 `5xx`、连接重置和临时超时：按退避策略重试；
- TLS 证书、主机名错误等永久安全错误：立即失败，不关闭证书校验；
- 无图片、Base64 无效、URL 下载失败或图片不可解码：当前屏失败并保留诊断信息。

日志和用户错误消息不得包含 API key、Authorization 头、multipart 原文或可能携带凭据的 URL 查询参数。每屏耗尽重试后停止当前商品，由现有失败队列或后续恢复补齐。

## 测试与验收

自动测试全部使用模拟服务，不调用真实付费 API。

### 单元测试

- Base URL 规范化、补 `/v1` 和重复路径防护；
- 配置、密钥保留/清除和双文件回滚；
- multipart 多参考图请求；
- Base64 与 URL 响应解析及图片校验；
- 动态 `detail_page_count`、任务快照和屏级提示词拆分；
- 每屏检查点、部分失败、恢复只补缺失屏；
- 401、403、400、404、429、5xx、超时和永久 TLS 分类；
- 日志与异常脱敏。

### 流程测试

- 四种 provider 组合均路由到正确实现；
- 全 GPT Image 时不实例化 Lovart；
- 混合模式只在需要时创建 Lovart project；
- 白底或场景失败后不进入后续付费阶段；
- 提示词屏数不符时不调用最终图片 API；
- 设置数量为 1、非默认值和较大允许值时，输出及成功判断一致；
- 旧 Lovart 状态可以继续恢复。

### WebUI 验收

- 密钥字段被遮蔽且不会回显到日志；
- 保存、回滚和运行时选择正常；
- 缺少所选 provider 配置时在启动前提示；
- 付费测试必须由用户点击触发并显示费用提示；
- 任务进度显示当前屏、目标数量、已完成数量和失败编号。

### 完成标准

1. 用户可以独立选择两段图片 provider；
2. 提示词来源行为不变；
3. 最终张数严格跟随任务快照中的 `detail_page_count`；
4. GPT Image 中断后不会重复生成已完成图片；
5. 全 GPT Image 流程不要求 Lovart；
6. 密钥不进入仓库、日志、状态和测试产物；
7. 全量自动测试通过，真实付费调用仅由用户主动测试。
