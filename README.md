# Gemini Lovart Image Automation

This project reads product data and images from an Excel workbook, sends the product context to Gemini or NVIDIA Kimi to generate ecommerce image prompts, and then uses Lovart to create product detail images.

## First-Time Setup

Clone the repository, then run these commands from the project root:

```powershell
copy .env.example .env
copy config.example.yaml config.yaml
uv pip install -r requirements.txt
uv run playwright install chromium
```

If you do not use `uv`, install dependencies with Python directly:

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Required Local Files

These files are intentionally not uploaded to GitHub. Create them locally before running the project:

- `.env`: API keys and secrets. Start from `.env.example`.
- `config.yaml`: local workbook path, column settings, browser path, and Lovart/Gemini settings. Start from `config.example.yaml`.
- Excel workbook, usually `data/products.xlsx`: your product table and embedded product images.

The `.env` file should contain values like:

```text
GEMINI_API_KEY=your_gemini_api_key
NVIDIA_API_KEY=your_nvidia_api_key
LOVART_ACCESS_KEY=your_lovart_access_key
LOVART_SECRET_KEY=your_lovart_secret_key
```

## Running

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

## Files Generated Automatically

These paths are created during normal use and do not need to be uploaded:

- `logs/`: run logs.
- `runs/`: per-run summaries and browser debug snapshots.
- `output/`: extracted workbook images, Gemini prompts, Lovart outputs, `status.json`, and `results.csv`.
- `browser_profile/`: persistent browser login state for the Gemini browser flow.
- `.venv/`: local Python virtual environment, if you create one.
- `__pycache__/`: Python cache files.

`output/results.csv` is a local summary table. The project updates one row per `product_id`, so reruns can preserve the latest status and Lovart project URL.

## What Should Be Uploaded

GitHub should contain source code, tests, templates, and documentation:

- Python source files such as `main.py`, `excel_reader.py`, `gemini_bot.py`, `lovart_bot.py`, and `utils.py`.
- Template files such as `.env.example` and `config.example.yaml`.
- `requirements.txt`, `pyproject.toml`, and `uv.lock`.
- Tests under `tests/`.
- Documentation such as this `README.md`.

Do not upload real API keys, browser profiles, generated images, run logs, or local Excel data unless you intentionally want to share that data.

## Notes for New Users

The first time you use the Gemini browser flow, you may need to log in manually. After that, the login state is stored in `browser_profile/`.

If the Excel path in `config.yaml` points to `data/products.xlsx`, create the `data/` folder locally and put your workbook there.

By default, column `H` is `reference_images_are_product`. Fill `是` when the later reference images are the same product/spec/color/style, or `否` when they should only be used as style references.
