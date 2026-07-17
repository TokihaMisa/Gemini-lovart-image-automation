# Gemini 浏览器登录与弱网络恢复设计

## 背景与根因

同事电脑使用 `gemini_browser` 时，任务日志依次出现：

- `temporary chat control not found`
- `mode menu control not found`
- `Gemini Thinking mode could not be selected`

当前 Gemini 浏览器流程的实际顺序是：打开 Gemini 页面、固定等待 4 秒、尝试临时会话、选择 Thinking 模式、发送 preamble、上传图片、发送商品提示词。因此本次失败发生在图片上传之前，不能归因于文件没有上传完成。

现有实现存在四个直接风险：

1. WebUI 明确选择 `gemini_browser` 时不会等待用户确认登录完成。
2. 登录判断只检查 URL，无法识别仍停留在 `gemini.google.com` 的未登录落地页。
3. 页面就绪依赖固定等待，没有等待真实聊天控件。
4. 临时会话与 Thinking 选择器主要覆盖中文和英文，西班牙语界面会增加失败概率。

弱网络会放大上述问题。浏览器导航目前只对一次特殊的导航中断做重试，Playwright 超时、连接重置、DNS、`net::ERR_*` 和部分 SSL 协议错误没有统一恢复策略。

## 目标

1. 在软件内提供专用 Gemini 登录入口，使用正式任务相同的持久化浏览器账号目录。
2. 用户可以明确检查登录是否就绪，并在就绪后安全关闭登录浏览器。
3. 正式任务开始前再次验证登录和页面就绪状态，失败时不处理商品。
4. 以页面结构为主，兼容中文、英文和西班牙语 Gemini 界面。
5. 对可恢复的弱网络、限流、服务器错误和临时 SSL 协议中断进行有限重试。
6. 保持 TLS 校验，区分可恢复错误与永久证书/认证错误。
7. 提供可操作的中文状态和诊断材料，避免所有问题都显示为 Thinking 模式失败。

## 不包含

- 不自动填写 Google 账号、密码、验证码或双因素认证。
- 不读取、记录或显示完整 Google 账号邮箱。
- 不绕过 Google 登录安全检查或浏览器风控。
- 不默认关闭 TLS 证书验证。
- 不为任意未知 Gemini 语言建立无限文本词典；本次明确覆盖中文、英文和西班牙语，并以结构选择器降低语言依赖。
- 不在 Gemini 重试期间重复发起最终 Lovart 详情页绘图。

## 方案比较与决策

### 方案 A：独立登录助手进程（采用）

登录按钮启动一个独立助手进程。助手使用正式任务相同的 `browser_profile`，打开 Gemini、持续检测状态，并通过本地原子状态文件与 WebUI 通信。

优点：

- 适合 PyInstaller 无控制台客户端。
- Playwright 页面对象始终只在创建它的进程内使用。
- WebUI 回调不需要长时间占用工作线程。
- 登录过程与正式商品任务隔离，崩溃和关闭边界清晰。

代价：需要管理 PID、状态文件、关闭请求和 profile 互斥。

### 方案 B：直接启动 Chrome，用户手动关闭（不采用）

实现简单，但用户忘记关闭时会锁住 profile，检查按钮也无法可靠控制正在运行的页面。

### 方案 C：正式任务内暂停等待登录（不采用）

步骤少，但会占用任务进程；打包客户端无法安全依赖控制台 `input()`，也容易再次出现未登录即执行。

## 架构

### `gemini_browser_session.py`

新增独立模块，作为浏览器会话的唯一公共入口，负责：

- 解析并创建持久化 `browser.user_data_dir`。
- 解析 Chrome/Edge 可执行文件并构造统一 Playwright 启动参数。
- Gemini 页面导航、页面就绪等待和登录状态判断。
- 登录助手状态机、PID、关闭请求与原子 JSON 状态文件。
- profile 占用检测与陈旧状态清理。
- 网络/SSL/HTTP 错误分类和均衡重试策略。
- 页面语言、URL、可见控件摘要和诊断信息采集。

该模块不处理 Excel 商品、提示词或 Lovart 任务。

### `app.py`

新增内部启动参数 `--gemini-login-helper`：

- 冻结版使用当前 EXE 启动助手。
- 源码版使用当前 Python 和 `app.py` 启动助手。
- 助手进程只打开 Gemini 并维护状态，不启动 Gradio 或商品流程。

### `webui.py`

在“API 与模型”页增加“Gemini 浏览器账号”区域：

- `打开 Gemini 登录浏览器`
- `检查登录并关闭浏览器`
- 只读状态区域

WebUI 回调只启动助手、读取状态、提交关闭请求和更新状态，不直接持有 Playwright 对象。

正式任务启动前，如果提示词来源为 `gemini_browser`：

- 登录助手仍活跃时阻止启动，并提示先完成检查与关闭。
- 陈旧状态可安全清理。
- 未登录状态不启动商品子进程。

### `main.py`

正式浏览器流程改为复用 `gemini_browser_session.py`：

- 使用与登录助手一致的 profile、浏览器和页面就绪判定。
- 移除打包 WebUI 路径对控制台 `input()` 的依赖。
- 页面未就绪时输出明确错误并在商品处理前退出。

### `gemini_bot.py`

保留商品级自动化职责，并增强：

- 中文、英文、西班牙语的临时会话、模式、Thinking 和上传文本回退。
- 结构选择器优先于文本选择器。
- Thinking 失败前重新确认页面就绪，必要时刷新并重试。
- 图片只有在确认上传完成后才发送商品提示词。
- Gemini 浏览器阶段可完整重试当前商品，但不重复调用后续最终 Lovart 绘图。

## 登录助手协议

运行时文件位于本机忽略目录，例如 `runs/gemini_login/`：

- `status.json`：助手当前状态。
- `close.request`：WebUI 请求助手安全关闭。

`status.json` 使用临时文件加 `os.replace` 原子写入，至少包含：

```json
{
  "pid": 1234,
  "state": "waiting_login",
  "ready": false,
  "url": "https://gemini.google.com/app",
  "language": "es",
  "message": "等待完成 Google 登录",
  "updated_at": "2026-07-17T18:30:00"
}
```

允许状态：

- `starting`
- `page_loading`
- `waiting_login`
- `ready`
- `closing`
- `closed`
- `error`

### 打开按钮

1. 检查状态文件与 PID。
2. 活跃助手已存在时不重复启动，返回当前状态。
3. 陈旧 PID 或状态文件被清理。
4. 使用相同 profile 启动助手并等待初始状态，最长只阻塞 WebUI 回调约 15 秒。
5. 浏览器继续保持打开，供用户完成账号、验证码和双因素登录。

### 检查并关闭按钮

1. 助手状态为 `ready` 时写入关闭请求。
2. 助手在自己的进程内关闭 Playwright context，写入 `closed`。
3. 未就绪时不关闭浏览器，返回当前原因。
4. 助手已异常退出时返回错误并清理陈旧状态。

### profile 互斥

- 登录助手存活时，正式任务不得启动同一 profile。
- 正式任务运行时，登录助手不得启动。
- 只对明确属于该 profile 且已确认陈旧的进程进行清理，不结束用户普通 Chrome/Edge 会话。

## 登录与页面就绪判定

判定采用组合证据，不依赖单个 URL 或单段文字。

### 明确未登录

- URL 包含 Google 登录、账号选择或验证路径。
- 页面存在可见登录、继续登录、账号选择或验证提示。
- 聊天编辑器不可用且登录入口可见。

### 页面就绪

- URL 位于 Gemini 应用域名。
- 可见且可编辑的聊天输入控件存在，例如 `textarea` 或带 `contenteditable`/`role=textbox` 的编辑器。
- 页面不处于登录、账号选择或验证状态。
- 页面不处于明显的加载遮罩或离线状态。

### 状态不确定

页面尚在跳转、输入框尚未出现或只有部分控件时，继续按页面就绪超时等待，不立即判定登录失败。

## 多语言策略

1. 优先使用 `role`、`contenteditable`、菜单层级、稳定属性和控件关系。
2. 对文本进行 Unicode 规范化、大小写折叠、首尾空格清理和重音符号兼容比较。
3. 文本回退覆盖：
   - 中文：临时、快速、扩展思考、上传等。
   - 英文：Temporary、Fast、Extended thinking、Upload 等。
   - 西班牙语：Temporal、Rápido、Pensamiento ampliado/extendido、Subir/Adjuntar 等常见界面文本。
4. `Flash`、`Pro` 等模型名继续作为跨语言辅助信号，但不作为唯一条件。
5. 从 `document.documentElement.lang` 记录检测语言；失败时保存可见控件摘要，便于后续补充界面变化。

Excel 中的目标图片语言只影响生成提示词和图片文案，不受 Gemini 网页自身显示语言影响。

## 均衡重试策略

默认参数：

```yaml
browser:
  network_attempts: 5
  page_ready_timeout: 90
  product_attempts: 2
  retry_delays: [3, 6, 12, 20]
```

第 5 次尝试前继续使用最大 20 秒间隔。界面和日志显示 `第 N/5 次` 与下一次等待秒数。

### 可恢复错误

- Playwright 导航超时。
- `ERR_CONNECTION_RESET`
- `ERR_CONNECTION_CLOSED`
- `ERR_NETWORK_CHANGED`
- `ERR_TIMED_OUT`
- 临时 DNS/连接中断。
- 部分 `ERR_SSL_PROTOCOL_ERROR` 或 TLS 握手中断。
- HTTP 408、429 和 5xx。
- API 层的 `URLError`、`socket.timeout`、连接重置与可恢复 `SSLError`。

### 不可恢复错误

- 401/403 认证和权限错误。
- 404 模型或端点配置错误。
- 明确未登录、账号验证或人工挑战。
- `ERR_CERT_AUTHORITY_INVALID`
- `ERR_CERT_COMMON_NAME_INVALID`
- `ERR_CERT_DATE_INVALID`
- 主机名不匹配和已确认的永久证书错误。

不可恢复错误立即停止并提供中文解决建议。TLS 验证默认开启；不会因为重试失败而自动切换到不安全模式。

### 页面导航

- 每次失败后按退避时间等待。
- 重新导航前确认 page/context 仍存活。
- 必要时新建 page，但保持同一持久化 context。
- 导航成功后继续等待真实聊天控件，不能只依赖 `domcontentloaded`。

### Thinking 与上传

- Thinking 菜单首次找不到时重新检查登录与页面就绪。
- 页面仍正常时刷新并重新尝试选择，最多受网络尝试上限约束。
- 图片上传使用现有 3 次上传重试，并以真实附件状态确认完成。
- 未确认上传完成时不发送商品提示词。

### 商品级重试边界

Gemini 浏览器生成阶段最多完整尝试 2 次。重试发生在 `GeminiBot.generate_prompt()` 内，只有成功返回文字提示词后才进入最终 Lovart 详情页步骤，因此不会重复提交最终 Lovart 绘图。白底图和场景图等已完成的前置产物继续复用。

## API 与 Lovart 重试一致性

- Gemini/NVIDIA 正式提示词 API 保留指数退避，并改用统一错误分类和日志格式。
- 模型发现与模型测试对可恢复网络错误增加有限重试，但认证、模型不存在和输入校验错误不重试。
- Lovart API 保留现有请求签名和重签逻辑，统一重试次数、退避显示和永久 TLS 错误处理。
- API 密钥、Cookie、认证头和原始敏感响应不得出现在重试日志中。

## 错误与诊断

错误类型至少区分：

- `Gemini 未登录`
- `Gemini 页面仍在加载`
- `Gemini 页面结构发生变化`
- `Gemini 网络连接失败`
- `SSL/TLS 临时连接失败`
- `SSL 证书无效，需要检查系统时间、代理或证书`
- `图片上传未完成`
- `Thinking 模式不可用`

最终失败时保存到当前 run 目录：

- 页面截图。
- 当前 URL。
- `document.documentElement.lang`。
- 可见按钮、菜单和输入控件的去敏摘要。
- 尝试次数和最后一个错误分类。

WebUI 商品卡显示中文主错误和诊断文件位置；底层英文错误可保留在运行日志，但不得泄露凭据或 Cookie。

## 安全与隐私

- Google Cookie、账号会话和浏览器存储只保存在本机 `browser_profile`。
- 不把账号邮箱、Cookie、token、请求头或 profile 内容写入状态 JSON 和日志。
- 登录助手状态只记录 URL、页面语言、状态和非敏感错误摘要。
- 不自动关闭用户普通浏览器进程。
- 不默认启用 `LOVART_INSECURE_SSL=1` 或 Chromium 忽略证书错误参数。

## 配置与兼容

- 旧配置缺少新字段时使用上述均衡默认值。
- `config.example.yaml` 和 WebUI 内嵌默认配置保持一致。
- `browser.user_data_dir` 继续兼容相对和绝对路径。
- 源码运行与 PyInstaller 冻结版使用同一状态协议和回调逻辑。
- Gemini API 与 NVIDIA 模式不依赖浏览器登录助手，现有流程不受阻塞。

## 测试策略

### 单元测试

- 登录 URL、账号选择、验证页、正常聊天页和状态不确定判定。
- 结构输入框与中英西文字回退。
- Unicode、大小写和西班牙语重音规范化。
- 可恢复/不可恢复 Playwright、HTTP、TLS 和连接错误分类。
- 5 次退避序列与最大等待值。
- 状态 JSON 原子写入、陈旧 PID、关闭请求和 profile 互斥。

### WebUI 行为测试

- 两个按钮及状态组件存在。
- 打开按钮不重复启动助手。
- 未就绪时检查按钮不关闭。
- 就绪后提交关闭并显示成功。
- `gemini_browser` 启动时助手仍活跃会被阻止。
- Gemini API/NVIDIA 来源不受浏览器状态影响。

### 浏览器流程测试

- 慢加载后出现聊天输入框并继续。
- 瞬时网络/SSL 错误重试后成功。
- 永久证书错误立即失败。
- Thinking 首次找不到、刷新后成功。
- 图片上传未完成时不发送商品提示词。
- 当前商品 Gemini 重试不会重复进入最终 Lovart 绘图。

### 回归与打包验证

- 完整自动化测试通过。
- 无真实账号和 API 配额的结构化测试通过。
- 本地手动登录按钮使用测试 profile 冒烟。
- PyInstaller EXE 中登录助手可启动、状态可更新并安全关闭。
- 更新版本号并制作新的 OTA 包后，解压包 EXE 冒烟通过。

## 验收标准

- 同事可以先在软件内打开 Gemini 登录浏览器，完成登录后检查并关闭。
- 再次启动软件或任务时复用同一账号会话。
- 未登录、登录助手未关闭或页面未就绪时不会处理商品。
- 慢网络下页面按均衡策略重试并显示进度。
- 西班牙语界面不再仅因按钮文本不同而立即失败。
- 图片确认上传完成后才发送商品提示词。
- 永久 SSL 证书错误不会被无限重试或静默忽略。
- 最终失败能区分登录、页面、网络、SSL、上传和 Thinking 问题。
- Gemini API、NVIDIA 和 Lovart 现有流程无回归。
