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

## Files Generated Automatically

These paths are created during normal use and do not need to be uploaded:

- `output/`: extracted workbook images, Gemini prompts, Lovart outputs, `status.json`, and `results.csv`.
- `logs/`: run logs.
- `runs/`: per-run summaries and browser debug snapshots.
- `browser_profile/`: persistent browser login state for the Gemini browser flow.

`output/results.csv` is a local summary table. The project updates one row per `product_id`, so reruns can preserve the latest status and Lovart project URL.
