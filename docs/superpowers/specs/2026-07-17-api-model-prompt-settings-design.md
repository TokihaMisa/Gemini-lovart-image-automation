# API 模型发现与提示词设置设计

## 目标

为现有 Gradio 软件增加以下能力：

1. 检测 Gemini API 与 NVIDIA API 的连通性，并返回可选择的模型列表。
2. 在软件内选择并长期保存 Gemini API 与 NVIDIA API 模型。
3. 在软件内编辑并长期保存 Excel 未提供的提示词参数。
4. 展示不可编辑的流程约束和安全规则，让使用者知道最终生效的限制。

本次不引入 OpenSpec。仓库现有的 `docs/superpowers/specs/` 与实现计划足以描述和执行这次改动，避免增加新的规范工具链。

## 范围

### 包含

- Gemini API 密钥检测、模型列表获取、模型筛选和所选模型测试。
- NVIDIA API 密钥检测、模型列表获取、模型筛选和所选模型测试。
- 工作台按提示词来源显示对应模型下拉框。
- 所选模型写入 `config.yaml` 并在下次启动时恢复。
- 结构化提示词参数、额外要求、默认值恢复和配置校验。
- Excel 商品级要求、软件长期参数、程序默认值和锁定规则之间的明确优先级。
- 三种提示词来源：Gemini API、Gemini 浏览器、NVIDIA API。
- 白底图、场景图、详情页设计和 Lovart 最终执行提示词的一致参数传递。
- 只读展示所有锁定规则及当前生效规则预览。
- 自动兼容缺少新字段的旧版 `config.yaml`。

### 不包含

- 开放完整原始提示词模板编辑。
- 允许用户编辑图片角色、上传顺序或 Lovart 付费确认规则。
- 自动替用户选择价格最高或能力最强的模型。
- 对动态返回的每个模型逐一发送付费测试请求。
- 修改 Excel 的列结构或商品数据格式。

## 设计原则与优先级

配置优先级从高到低固定为：

1. 流程锁定规则。
2. Excel 商品级要求。
3. 软件内长期提示词参数。
4. 程序默认值。

因此：

- 锁定规则始终生效，任何额外要求都不能覆盖。
- Excel 中存在的商品名、语言、图片尺寸/比例、卖点和参考图属性始终优先。
- 只有 Excel 没有提供的内容才使用软件长期参数或默认值。
- 旧配置缺少新增字段时使用安全默认值，而不是启动失败。

## 模块边界

### `model_provider.py`

新增统一的模型商适配层，负责：

- Gemini 与 NVIDIA 密钥检测。
- 模型列表请求和分页处理。
- 响应解析、通用模型信息归一化和过滤。
- 所选模型的极小多模态测试。
- HTTP 状态、网络异常和模型兼容错误转换为不含密钥的用户消息。

对外提供稳定接口，WebUI 不直接拼装供应商 HTTP 请求。统一数据结构为：

```python
@dataclass(frozen=True)
class DiscoveredModel:
    provider: str
    model_id: str
    display_name: str
    supports_generation: bool
    supports_thinking: bool | None
    image_input_status: str  # "verified", "reported", "unknown", "failed"
    recommendation: str      # "recommended", "available", "unsupported"
```

适配层提供：

```python
def discover_models(provider: str, api_key: str, base_url: str) -> list[DiscoveredModel]: ...
def test_selected_model(provider: str, api_key: str, base_url: str, model_id: str) -> ModelTestResult: ...
```

### `prompt_settings.py`

新增提示词设置层，负责：

- 定义默认配置。
- 从 `config.yaml` 读取、归一化和校验配置。
- 应用 Excel 优先级。
- 构造白底图、场景图和详情页设计参数片段。
- 提供只读锁定规则和生效规则预览。

现有 `utils.py` 中的提示词构造函数继续作为业务入口，但改为消费归一化后的提示词设置，避免三种提示词来源各自拼装要求。

### `webui.py`

WebUI 负责：

- 收集密钥、服务地址、模型选择和提示词设置。
- 调用适配层执行检测、刷新与测试。
- 显示状态、模型标签、只读规则与最终预览。
- 经校验后将长期设置保存到 `config.yaml`。
- 启动任务前保存当前提示词来源和对应模型。

## API 与模型发现

### Gemini

使用 Gemini Models API 获取模型列表，处理 `nextPageToken`，只保留支持 `generateContent` 的生成模型。过滤 embedding、Live、语音、纯图像生成和其他明显不适合当前文字提示词任务的模型。

Gemini 返回的 `thinking`、输入/输出 token 上限等元数据用于标签展示。若返回数据没有明确说明图片输入能力，则标记为“图片能力未验证”，而不是猜测为可用。

### NVIDIA

使用配置的 OpenAI-compatible `base_url` 请求 `/models`。过滤 embedding、rerank、语音、纯图像生成等明显不适用模型。由于 OpenAI-compatible 模型列表通常只包含模型 ID，图片输入与 Thinking 能力默认标记为“未知”。

如果供应商不支持 `/models`，检测结果明确显示“服务可访问但不支持模型列表接口”，保留已保存模型，不清空下拉框。

### 检测并刷新模型

Gemini 和 NVIDIA 各自提供“检测 API 并刷新模型”按钮。点击后：

1. 校验密钥与服务地址。
2. 请求模型列表。
3. 过滤不适用模型。
4. 返回可读状态、模型标签和下拉选项。
5. 保留当前选中模型；仅当它不存在且列表非空时选择第一个推荐模型。

动态模型列表不写入 `config.yaml`，避免把易过期的供应商目录当作长期配置。仅长期保存用户选中的模型。刷新失败时不得覆盖或清空已保存模型。

### 测试所选模型

“测试所选模型”发送一个极小的文字加内置测试图片请求，并限制输出长度，只验证以下条件：

- 认证和模型 ID 有效。
- 模型接受当前项目使用的图片消息格式。
- 模型能返回非空文字。

测试前显示“可能产生极少量 API 用量”。测试结果只更新当前 UI 状态，不自动删除模型。失败模型标记为“测试失败”，用户仍可保留选择。

## 界面设计

### 工作台

在“提示词引擎”旁新增统一“模型”下拉框：

- `gemini_api`：显示 Gemini 模型。
- `nvidia`：显示 NVIDIA 模型。
- `gemini_browser`：显示“由浏览器页面选择”，不可编辑。

切换引擎时切换下拉选项，并恢复该供应商上次保存的模型。启动任务时保存选中模型。

### API 与模型页

将现有“密钥配置”扩展为 Gemini 与 NVIDIA 两个区域。每个区域包含：

- 密钥输入。
- 服务地址。
- “保存密钥”。
- “检测 API 并刷新模型”。
- 模型列表/下拉框。
- “测试所选模型”。
- 连接、刷新、限流和兼容性状态。

Lovart 密钥继续保留在同一页，但不参与提示词模型发现。

### 提示词设置页

新增独立标签页，长期保存：

- 详情页屏数/成品图数量，默认 `12`，允许 `1-50`。一屏对应一张详情成品图，不表示同一商品的多套设计版本。
- 整体设计风格。
- 每屏必须包含的内容。
- 图片画质。
- Logo 规则。
- 文案风格。
- 文案详细程度。
- 产品还原强调程度。
- 白底图精修要求。
- 场景图生成要求。
- 是否允许模型反问，默认关闭。
- Excel 未填写语言时的默认语言。
- Excel 未填写图片尺寸时的处理规则。
- 自定义额外要求。

页面提供“保存设置”和“恢复默认值”。恢复默认值先更新表单，只有用户点击保存后才写入磁盘，避免误操作立即覆盖长期配置。

### 只读规则

提示词设置页展示不可编辑的锁定规则和当前生效规则预览：

- 所有提示词生成模型只输出可交给 Lovart 的文字设计提示词，不直接生成图片。
- Excel 已有商品要求优先，软件设置不得覆盖。
- 商品图片角色、上传顺序和参考图属性由程序与 Excel 决定。
- 不得改变商品真实形态或虚构不存在的部件、颜色和结构。
- Lovart 付费确认与安全规则不可编辑。
- 最终图片只能在 Lovart 阶段生成。

只读区域采用不可编辑文本框或代码预览，而不是隐藏字段。它既帮助使用者理解约束，也使问题排查时能够确认实际规则。

## 长期配置

`config.yaml` 新增：

```yaml
prompt_settings:
  detail_page_count: 12
  design_style: "温馨感、高级感"
  required_sections:
    - 主标题
    - 副标题
    - 信息布局
    - 排版形式
  image_quality: "1K"
  logo_policy: "不出现 Logo"
  copy_style: "适合跨境电商，具体、不空泛"
  copy_detail_level: "详细"
  product_fidelity: "严格还原"
  white_background_requirements: "白底、超清摄影、突出高级感"
  scene_requirements: "场景重新设计，产品特征保持一致"
  allow_questions: false
  default_language: "巴西葡萄牙语"
  missing_image_size_policy: "不使用默认固定比例"
  extra_requirements: ""

gemini_api:
  base_url: "https://generativelanguage.googleapis.com/v1beta"
  model: "gemini-2.5-flash-lite"

nvidia_api:
  base_url: "https://integrate.api.nvidia.com/v1"
  model: "moonshotai/kimi-k2.5"
  send_images: true
```

为兼容旧配置，`nvidia_api.model_choice` 与 `nvidia_api.models` 暂时继续支持读取；保存新选择时写入直接的 `nvidia_api.model`。后续可在单独版本中移除旧字段。

API 密钥继续保存在 `.env`，不得写入 `config.yaml`、日志、状态文件或模型目录缓存。

## 提示词数据流

归一化后的设置进入所有相关阶段：

1. 白底图提示词使用 `white_background_requirements` 和 `image_quality`，再追加 Excel 图片尺寸。
2. 场景图提示词使用 `scene_requirements` 和 `image_quality`，再追加 Excel 图片尺寸。
3. 详情页设计提示词使用屏数、风格、每屏结构、画质、Logo、文案和额外要求。
4. Gemini API、Gemini 浏览器和 NVIDIA 调用同一个详情页提示词构造函数。
5. Lovart 最终提示词再次带入屏数与关键要求，避免提示词生成阶段和图片执行阶段不一致。

当 Excel 提供语言或图片尺寸时，使用 Excel 值；只有字段为空时才使用 `default_language` 或 `missing_image_size_policy`。

额外要求位于可编辑参数之后。最终发送前由构造函数追加锁定规则，因此额外要求无法在文本顺序上覆盖锁定约束。冲突内容仍可能被用户写入额外要求，但锁定规则会明确声明冲突时忽略额外要求。

## 错误处理

- 缺少密钥：不发起请求，提示需要先填写并保存。
- `401/403`：显示密钥无效或无权限，不回显响应中的敏感认证信息。
- `404`：区分服务地址错误、模型不存在和供应商不支持模型列表端点。
- `429`：显示限流状态，不清空模型选择。
- 网络、DNS、TLS 或超时：显示简短可操作信息，保留现有配置。
- 模型列表为空：显示“连接成功但没有发现适用模型”。
- 模型测试返回空文本：标记测试失败并保留模型。
- 设置校验失败：显示具体字段错误，不写入任何部分更新。

所有网络操作设置有限超时。WebUI 回调捕获异常并返回状态，不允许异常终止 Gradio 服务。密钥在日志中只允许显示是否存在，不允许显示前缀、后缀或完整值。

## 校验规则

- `detail_page_count` 必须是 `1-50` 的整数。
- 列表字段删除空项并去重。
- 单行文本去除首尾空白并限制合理长度。
- 多行额外要求限制最大长度，防止配置误粘贴超大文档。
- 服务地址必须为 `http` 或 `https` URL。
- 模型 ID 不能为空且不包含换行。
- `config.yaml` 保存采用完整校验后的单次写入，失败时保留旧文件。

## 测试策略

### 单元测试

- Gemini 模型分页、解析、过滤和标签。
- NVIDIA/OpenAI-compatible 模型解析与过滤。
- `401/403/404/429`、网络异常和超时映射。
- 极小多模态测试请求的 payload 与响应解析。
- 提示词设置默认值、校验和旧配置兼容。
- Excel 字段优先级。
- 屏数、风格、画质、Logo、每屏内容和额外要求进入最终提示词。
- 三种提示词来源共享同一核心详情页要求。
- 锁定规则始终位于最终提示词中，额外要求冲突时明确以锁定规则为准。

### WebUI 行为测试

- 引擎切换更新模型下拉框。
- 刷新失败保留已选模型。
- 保存后重新加载恢复模型和提示词设置。
- 恢复默认值在保存前不修改磁盘。
- 只读规则组件不可编辑且内容完整。

### 回归测试

- 现有 Gemini API、Gemini 浏览器、NVIDIA 和 Lovart 流程测试全部通过。
- 旧版 `config.yaml` 无需人工迁移即可运行。
- 浏览器模式不依赖 API 模型列表即可继续使用。

## 验收标准

- 用户可分别检测 Gemini 与 NVIDIA API，并看到明确结果。
- 检测成功后模型下拉框显示经过基础过滤的模型列表。
- 用户可对所选模型执行极小多模态测试。
- 用户选择的 Gemini/NVIDIA 模型在重启后保留。
- 用户可编辑并长期保存所有约定的非 Excel 提示词参数。
- Excel 中的商品级要求始终优先。
- 三种提示词来源均遵守“只输出文字提示词、不直接生成图片”。
- 所有锁定规则在界面可见但不可编辑。
- API 或模型列表服务失败不会清空配置或使 WebUI 崩溃。
- 现有自动化主流程和测试套件无回归。
