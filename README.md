# Gemini Lovart Image Automation Pro

This project reads product data and images from an Excel workbook, sends the product context to Gemini or NVIDIA Kimi to generate ecommerce image prompts, and then uses Lovart to create product detail images.

**🔥 New:** Now features an ultra-sleek, enterprise-grade dark mode WebUI!

## Features

- **Sleek Web Interface (`webui.py`)**: A modern Gradio-based WebUI with a gorgeous dark theme, pill-shaped dropdowns, glassmorphism panels, and a dynamic Command Center.
- **Auto Folder Picker**: Directly browse your local PC for output folders using a native dialog window (no typing required!).
- **Multi-Model Support**: Automatically choose between `gemini_api`, `gemini_browser`, and `nvidia` for prompt generation.
- **Lovart Automation**: Submit generated prompts directly to Lovart's image generation engines with real-time tracking.
- **Smart Data Handling**: Upload your `.xlsx` task tables directly via the web interface.

## First-Time Setup

Clone the repository, then run the Windows setup helper from the project root:

```powershell
setup_windows.bat
```

If Python 3.12 or newer is not installed, the script will stop and show the Python install command. After installing Python, run `setup_windows.bat` again. It will install Python dependencies, install Playwright Chromium, create `.env`, create `config.yaml`, and create the `data/` folder.

Manual setup:

```powershell
copy .env.example .env
copy config.example.yaml config.yaml
uv pip install -r requirements.txt
uv run playwright install chromium
```

## Running the WebUI (Recommended)

Start the beautiful graphical interface:

```powershell
python webui.py
```
> **Note:** If the UI becomes unresponsive on Windows, ensure you haven't clicked inside the black terminal window (disabling "Quick Edit Mode" in the terminal properties prevents this).

## Running via CLI

Interactive mode:

```powershell
uv run python main.py
```

Dry run, useful for checking Excel parsing without calling Gemini or Lovart:

```powershell
uv run python main.py --dry-run --limit 5
```

Non-interactive example:

```powershell
uv run python main.py --prompt-source nvidia --nvidia-model kimi --lovart unlimited --limit 1
```

## Required Configuration

These files are intentionally not uploaded to GitHub. Create them locally before running the project:

- `.env`: API keys and secrets. Start from `.env.example`.
- `config.yaml`: local workbook path, column settings, browser path, and Lovart/Gemini settings. Start from `config.example.yaml`.

The `.env` file should contain values like:

```text
GEMINI_API_KEY=your_gemini_api_key
NVIDIA_API_KEY=your_nvidia_api_key
LOVART_ACCESS_KEY=your_lovart_access_key
LOVART_SECRET_KEY=your_lovart_secret_key
```

You can also input and save these keys directly via the **"⚙️ 系统设置" (System Settings)** tab in the WebUI.

## Prompt-model and prompt-settings workflow

1. Save your Gemini and/or NVIDIA API key in `.env` or in the WebUI system settings.
2. In the WebUI, click **“检测 API 并刷新模型”** for the selected Gemini or NVIDIA prompt source.
3. Select a discovered model. You may then run the minimal multimodal model test; it can use a very small amount of API quota.
4. Open **“提示词设置”** and save the persistent prompt settings you want to use.
5. Review the visible, read-only locked rules in the effective-rule preview.
6. Start the task. Values supplied by Excel override these software defaults.

Browser mode does not require API model discovery. The saved API model fields are `gemini_api.model` and `nvidia_api.model`; the legacy NVIDIA `model_choice` and `models` fields remain compatibility-only for older configurations.

Prompt precedence is: Excel product values (name, language, image size/aspect ratio, selling points, and reference-image attributes) → saved prompt settings → built-in defaults. Locked rules always apply and cannot be changed in the form.

## Gemini 浏览器登录、生成与网络排查

使用 `gemini_browser` 前，请在 WebUI 中按以下顺序操作：

1. 打开 **API 与模型**。
2. 点击 **打开 Gemini 登录浏览器**。
3. 在打开的窗口中完成 Google 登录和账号验证。
4. 点击 **检查登录并关闭浏览器**，等待显示登录成功状态。
5. 再启动 `gemini_browser` 任务。

浏览器登录使用本机的 `browser_profile` 保存会话；不要共享或提交该目录。Gemini 页面控件支持中文、英文和西班牙语界面。`gemini_browser` 只负责根据商品资料生成提示词文本，最终图片由后续 Lovart 流程生成；输出的语言和内容仍以产品文档（Excel）及已保存的提示词设置为准。

弱网络下会进行有界重试：默认最多 5 次网络尝试，页面就绪最多等待 90 秒，每个商品最多 2 次 Gemini 浏览器尝试，重试间隔为 3、6、12、20 秒。超时、临时连接/DNS 变化、部分 SSL 协议错误，以及 408、429 和 5xx 响应可重试；登录验证、权限问题和不存在的模型或端点不会被反复重试。

证书颁发机构、域名或日期错误属于永久 TLS 问题。请先检查系统时间、代理/VPN、防病毒软件的 TLS 拦截，以及企业根证书配置。关闭 TLS 证书校验不是默认解决方案，也不应作为常规修复手段。

常见状态/错误类别包括：等待登录、页面加载/未就绪、永久 TLS、上传未完成、Thinking/页面结构、认证/权限，以及端点/模型未找到。界面会给出可操作的状态信息，但网络、账号验证和第三方服务状态仍可能需要人工处理。

## Files Generated Automatically

These paths are created during normal use and do not need to be uploaded:

- `output/`: extracted workbook images, Gemini prompts, Lovart outputs, `status.json`, and `results.csv`.
- `logs/`: run logs.
- `runs/`: per-run summaries and browser debug snapshots.
- `browser_profile/`: persistent browser login state for the Gemini browser flow.

`output/results.csv` is a local summary table. The project updates one row per `product_id`, so reruns can preserve the latest status and Lovart project URL.
