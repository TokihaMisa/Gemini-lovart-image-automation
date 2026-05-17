# image-automation 项目说明

## 项目作用

这个项目用于把本地商品表格里的商品信息和商品图片，自动转成可用于 Lovart 生成电商详情图的设计提示词，并进一步调用 Lovart 生成图片结果。

整体目标是减少人工在多个工具之间反复复制粘贴的工作：

1. 从 Excel 读取商品 ID、中文商品名、目标语言、卖点和商品图片。
2. 把商品图片和商品信息交给 Gemini，生成电商详情页设计提示词。
3. 把 Gemini 生成的提示词和商品图片交给 Lovart。
4. 等待 Lovart 生成结果，下载生成图片，并记录项目链接。

项目支持三条提示词生成路径：

- **Gemini API 路径**：直接调用 Gemini API，速度更稳，不依赖浏览器登录。
- **Gemini 浏览器路径**：用 Playwright 打开 Gemini 网页，复用本地 Chrome profile 登录态。
- **NVIDIA API 路径**：调用 NVIDIA NIM 的 OpenAI-compatible Chat Completions API，当前只保留支持商品图片输入的 Kimi K2.5。

Lovart 当前主要走 OpenAPI/AgentSkill 方式，不再主要依赖手动浏览器点击。

## 输入与输出

### 输入

主要配置在 `config.yaml`：

- `excel.path`：商品 Excel 文件路径。
- `excel.sheet`：工作表，支持索引或 sheet 名称。
- `excel.columns`：商品字段列，包括商品 ID、中文名、语言、卖点。
- `excel.image_columns.start`：图片列开始位置。
- `browser`：浏览器路径和持久化 profile 目录。
- `gemini` / `gemini_api`：Gemini 网页或 API 配置。
- `nvidia_api`：NVIDIA API 配置，当前只配置支持商品图片输入的 Kimi 模型 ID。
- `lovart`：Lovart OpenAPI 访问配置。

Excel 默认字段含义：

| 字段 | 默认列 | 说明 |
| --- | --- | --- |
| 商品 ID | A | 用作输出目录名和 Lovart 项目名 |
| 中文商品名 | B | 传给 Gemini 的商品名 |
| 目标语言 | C | 详情页输出语言 |
| 商品卖点 | D | 商品信息和卖点描述 |
| 商品图片 | E 起 | 支持 WPS/Excel `DISPIMG(...)` 嵌入图片 |

### 输出

运行后主要生成在 `output/`：

- `output/<商品ID>/image_*.jpeg`：从 Excel 提取出的商品图片。
- `output/<商品名>/gemini_prompt.txt`：Gemini 生成的设计提示词。
- `output/<商品ID>/lovart/`：Lovart 返回并下载的生成结果。
- `output/results.csv`：成功生成时追加商品 ID、商品名、Lovart 项目链接。

日志生成在 `logs/run_时间.log`。

## 核心流程

### 1. 启动入口

入口文件是 `main.py`。

启动后会：

1. 读取 `config.yaml`。
2. 初始化日志。
3. 调用 `excel_reader.read_products()` 解析商品。
4. 在控制台列出解析到的商品。
5. 让用户选择 Gemini 来源：
   - `1`：浏览器自动化。
   - `2`：Gemini API，默认。
6. 让用户选择 Lovart 模式：
   - `1`：Fast，消耗 credits。
   - `2`：Unlimited，可能排队，默认。
7. 逐个商品执行 Gemini 生成和 Lovart 生成。

程序支持 `Ctrl+C` 中断。中断时会设置停止标记，尽量等当前商品处理完再退出。

### 2. Excel 解析

`excel_reader.py` 负责读取商品表和图片。

它不是只依赖 `openpyxl` 的普通图片对象，而是专门兼容 WPS/Kingsoft 风格的 `DISPIMG(...)` 图片存储：

1. 把 `.xlsx` 当 zip 打开。
2. 读取 `xl/cellimages.xml`，获得图片 ID 和关系 ID。
3. 读取 `xl/_rels/cellimages.xml.rels`，把关系 ID 映射到 `xl/media/*`。
4. 扫描商品行图片列，把对应图片解压到 `output/<商品ID>/image_*.jpeg`。
5. 如果没有找到 `DISPIMG` 图片，再回退尝试 `ws._images`。

只有同时具备商品名和至少一张图片的行才会进入后续流程。

### 3. Gemini 生成提示词

#### API 路径

`gemini_api.py` 负责直接调用 Gemini API：

1. 读取 `preamble.txt`。
2. 先发送 preamble，让 Gemini 建立提示词专家角色。
3. 再把商品信息和图片作为多模态请求发送给 Gemini。
4. 保存返回文本到 `output/<商品名>/gemini_prompt.txt`。

#### 浏览器路径

`gemini_bot.py` 负责 Playwright 网页自动化：

1. 打开 `https://gemini.google.com/app`。
2. 尝试点击右上角的临时会话/Temporary chat。
3. 切换到思考模式/Thinking mode；如果切换失败，会保存 debug 快照并停止当前商品，避免误走快速模式。
4. 发送 `preamble.txt`，并等待 Gemini 的前置回复真正完成。
5. 上传商品图片，并等待图片上传完成。
6. 发送商品设计请求。
7. 根据 Gemini 页面自身状态等待回复完成：生成中的停止按钮、进度条、`aria-busy` 等状态消失并连续稳定后，才进入提取结果；不再通过回复文字关键词判断是否完成。
8. 从页面 DOM 中提取最后一条 Gemini 回复。
9. 保存为 `gemini_prompt.txt`。

历史验证中，浏览器路径最稳定的方式是复用独立的 `browser_profile`，并避免依赖 `networkidle` 这类网页后台活动很重时不稳定的等待条件。

### 4. Lovart 生成

`lovart_bot.py` 包装了 `lovart_api.AgentSkill`：

1. 根据用户选择调用 Lovart 模式设置：
   - `fast=True`：快速模式，可能消耗 credits。
   - `fast=False`：unlimited 模式，可能排队。
2. 上传本地商品图到 Lovart，得到 CDN URL。
3. 创建 Lovart project。
4. 将 Gemini prompt 和图片附件发送到 Lovart chat。
5. 按配置轮询状态直到完成、失败或超时。
6. 如果生成成功，下载 artifacts 到 `output/<商品ID>/lovart/`。
7. 将项目重命名为商品 ID。

`lovart_api.py` 是较完整的 Lovart OpenAPI 客户端，包含：

- HMAC-SHA256 鉴权。
- 项目创建/重命名/校验。
- 文件上传。
- chat 发送、轮询、获取结果。
- artifact 下载。
- 本地项目和 thread 状态管理。
- 命令行子命令。

## 当前文件职责

| 文件 | 职责 |
| --- | --- |
| `main.py` | 主入口，串联 Excel、Gemini、Lovart，并提供交互选择 |
| `excel_reader.py` | 读取 Excel 商品数据，提取 `DISPIMG` 或普通嵌入图片 |
| `gemini_api.py` | 直接调用 Gemini API 生成设计提示词 |
| `gemini_bot.py` | 通过 Playwright 自动操作 Gemini 网页 |
| `lovart_bot.py` | 封装商品级 Lovart 生成流程 |
| `lovart_api.py` | Lovart OpenAPI 客户端和 CLI |
| `utils.py` | 配置读取、日志、Excel 列号转换、输出目录辅助函数 |
| `preamble.txt` | Gemini 的角色/提示词生成专家预设 |
| `requirements.txt` | Python 依赖 |
| `.gitignore` | 忽略输出、日志、浏览器 profile、配置密钥等本地文件 |

## 运行方式

先安装依赖：

```powershell
uv pip install -r requirements.txt
uv run playwright install chromium
```

运行：

```powershell
uv run python main.py
```

非交互运行示例：

```powershell
uv run python main.py --gemini api --lovart unlimited --limit 5
```

也可以用 NVIDIA API 生成提示词：

```powershell
uv run python main.py --prompt-source nvidia --nvidia-model kimi --lovart unlimited --limit 1
```

dry-run 只解析 Excel、提取图片、写状态和运行报告，不调用 Gemini/Lovart：

```powershell
uv run python main.py --dry-run --limit 5
```

常用参数：

- `--prompt-source ask|gemini_api|gemini_browser|nvidia`：选择提示词生成来源，默认 `ask`。
- `--gemini ask|api|browser`：旧参数，仍可用，会映射到 `--prompt-source`。
- `--nvidia-model kimi`：当 `--prompt-source nvidia` 时使用 NVIDIA Kimi K2.5。因为当前只有它适合这个带商品图片的流程，所以不再提供 GLM/Kimi Thinking 选择。
- `--lovart ask|fast|unlimited`：选择 Lovart 模式，默认 `ask`。
- `--lovart-image-model auto|gpt_image_2|nano_banana|nano_banana_2|nano_banana_pro|midjourney|seedream_v4|seedream_v4_5`：临时指定 Lovart 图片模型。
- `--lovart-model-selection prefer|force`：`prefer` 是偏好该模型但允许 Lovart 自动规划；`force` 是强制只用该图片工具。
- `--lovart-reasoning fast|thinking`：临时指定 Lovart chat 推理模式。
- `--limit N`：只处理前 N 个解析到的商品。
- `--dry-run`：只检查输入和输出计划，不调用外部生成服务。
- `--no-resume`：即使商品已有 `lovart_done=true` 也重新处理。
- `--config PATH`：指定配置文件。

如果选择 NVIDIA API，程序会直接使用 `kimi`，对应 `moonshotai/kimi-k2.5`。它支持商品图片输入并默认启用 Thinking 模式，更适合“看商品图 + 写电商详情图提示词”的流程。

正式执行时，如果没有通过命令行指定 Lovart 图片模型，程序会让你在运行中选择：

```text
Lovart image model:
  [1] Auto
  [2] GPT Image 2
  [3] Nano Banana
  [4] Nano Banana 2
  [5] Nano Banana Pro
  [6] Midjourney
  [7] Seedream 4
  [8] Seedream 4.5

图片模型支持多选，输入逗号分隔即可，例如 `4,5` 表示同时选择 Nano Banana 2 和 Nano Banana Pro。

Lovart model selection:
  [1] Prefer
  [2] Force

Lovart reasoning mode:
  [1] Fast
  [2] Thinking
```

也可以在 `config.yaml` 中设置默认值：

```yaml
lovart:
  image_model: "nano_banana_2,nano_banana_pro"
  model_selection: "force"
  reasoning_mode: "thinking"
```

如果 `image_model: "auto"`，程序不会指定具体图片模型，由 Lovart Agent 自动选择。

如果不用 `uv`，也可以使用本机 Python：

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
python main.py
```

运行前需要确认：

1. `config.yaml` 中的 Excel 路径存在。
2. 如果选 Gemini API，`GEMINI_API_KEY` 已配置。
3. 如果选 NVIDIA API，`NVIDIA_API_KEY` 已配置。
4. `LOVART_ACCESS_KEY` / `LOVART_SECRET_KEY` 已配置。
5. 如果选浏览器 Gemini，首次运行可能需要手动登录 Gemini。

可以把密钥放在本地 `.env` 文件中，程序启动时会自动读取；`.env` 已在 `.gitignore` 中，不应提交。参考 `.env.example`。

## 已知风险和可优化点

### 高优先级（已处理）

1. **密钥不应写在 `config.yaml` 明文里**

   已从 `config.yaml` 移除明文 Gemini/Lovart 密钥，并新增 `config.example.yaml` 和 `.env.example`。真实密钥改为从环境变量或本地 `.env` 读取：

   - `GEMINI_API_KEY`
   - `NVIDIA_API_KEY`
   - `LOVART_ACCESS_KEY`
   - `LOVART_SECRET_KEY`

2. **部分源码和 `preamble.txt` 出现明显乱码**

   已重写 `preamble.txt`、`gemini_api.py`、`gemini_bot.py`、`lovart_bot.py` 中影响运行的中文提示词和日志文案。Gemini API 和浏览器路径现在共用 `utils.build_design_prompt()` 生成 UTF-8 中文商品提示词。

3. **Gemini API 调用可以合并成一次更清晰的多模态请求**

   已改为一次请求发送 `preamble.txt`、商品任务和图片，不再先请求 preamble 再把 Gemini 对 preamble 的回复拼回商品 prompt，减少噪音。

4. **输出目录命名不一致**

   已统一为按 `商品ID` 保存：

   ```text
   output/<商品ID>/
     image_*.jpeg
     gemini_prompt.txt
     lovart/
     status.json
   ```

5. **缺少断点续跑和状态文件**

   已为每个商品写入 `status.json`。主流程启动时会跳过已有 `lovart_done=true` 的商品。当前状态字段包括：

   - `parsed`
   - `gemini_done`
   - `lovart_uploaded`
   - `lovart_submitted`
   - `lovart_done`
   - `needs_manual_action`
   - `failed`
   - `reason`

   `results.csv` 也已改为用 Python `csv` 模块写入，包含表头并支持逗号、引号等字符转义。

### 中优先级

6. **Lovart pending confirmation 处理不完整（已顺手处理）**

   `lovart_bot.py` 现在会把 `pending_confirmation` 写入 `status.json` 的 `needs_manual_action`，并避免把它当作成功生成。

7. **日志 handler 可能重复添加（已顺手处理）**

   `setup_logging()` 现在会先清空已有 handler，避免同一进程多次调用时重复打印。

8. **`results.csv` 没有表头，也没有 CSV 转义（已顺手处理）**

   已通过 `utils.append_result()` 统一写入 CSV。

9. **浏览器路径的选择器容易随网页变化失效（已处理留证）**

   `gemini_bot.py` 现在会在图片上传失败、回复过短或异常时保存浏览器调试证据到：

   ```text
   runs/<时间>/browser-debug/<商品ID>/
     <时间戳>-<原因>.png
     <时间戳>-<原因>.html
   ```

   这样 Gemini 页面更新或选择器失效时，可以直接看当时页面状态。

10. **Excel 图片扫描规则比较隐式（已处理）**

    `excel.image_columns` 现在支持：

    - `start`：图片起始列。
    - `end`：可选，固定扫描到某列。
    - `max_columns`：可选，从起始列最多扫描多少列。
    - `empty_streak`：可选，连续多少个空图片单元格后停止，默认 `2`。

11. **缺少结构化运行报告（已处理）**

    每次运行现在会生成：

    ```text
    runs/<时间>/summary.json
    runs/<时间>/summary.csv
    ```

    报告记录商品 ID、商品名、状态、Lovart 链接、Gemini 字符数、artifact 数量、耗时和错误原因。

### 低优先级（已处理）

12. **CLI 参数可以替代运行时 input（已处理）**

    已支持：

    ```powershell
    uv run python main.py --gemini api --lovart unlimited --limit 5
    ```

    默认仍保留 `ask` 交互模式，适合人工确认；脚本化运行可以显式传 `--gemini` 和 `--lovart`。

13. **可以增加 dry-run 模式（已处理）**

    已支持 `--dry-run`。dry-run 只解析 Excel、提取图片、写 `status.json` 和 `runs/<时间>/summary.*`，不调用 Gemini/Lovart。

14. **可以增加最小测试集（已处理）**

    已补充 `unittest` 覆盖：

    - Excel 列字母转换。
    - 图片扫描配置解析。
    - `DISPIMG` 映射解析。
    - Lovart result 中 artifact 是否判定成功。
    - `results.csv` 写入。
    - `.env` 加载。
    - dry-run 状态和运行报告。
    - 浏览器 debug 快照。

15. **依赖和 Python 版本应固定（已处理）**

    已固定：

    - `requirements.txt` 使用精确版本。
    - `pyproject.toml` 指定 Python 范围和依赖版本。
    - `uv.lock` 锁定解析结果。

## 建议的下一步改造顺序

1. 跑一轮真实 dry-run，确认当前 Excel 能解析到预期商品和图片。
2. 用 `--gemini api --lovart unlimited --limit 1` 跑一条真实链路。
3. 如果真实链路稳定，再考虑把运行入口封装成一键 `.ps1` 脚本。
