import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_ENV_KEYS = [
    "GEMINI_API_KEY",
    "NVIDIA_API_KEY",
    "LOVART_ACCESS_KEY",
    "LOVART_SECRET_KEY",
]


def _is_placeholder(value: str) -> bool:
    text = (value or "").strip().strip("\"'")
    lowered = text.lower()
    return not text or lowered.startswith("your_") or lowered in {"changeme", "replace_me", "todo", "xxx"}


def _read_env_values(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def missing_or_placeholder_env_keys(env_path: Path, required_keys: list[str] | None = None) -> list[str]:
    values = _read_env_values(env_path)
    return [
        key
        for key in (required_keys or REQUIRED_ENV_KEYS)
        if key not in values or _is_placeholder(values[key])
    ]


def ensure_local_setup_files(root: Path) -> list[str]:
    actions: list[str] = []
    env_example = root / ".env.example"
    env_file = root / ".env"
    config_example = root / "config.example.yaml"
    config_file = root / "config.yaml"
    data_dir = root / "data"

    if not env_file.exists() and env_example.exists():
        shutil.copyfile(env_example, env_file)
        actions.append("created .env from .env.example")
    elif env_file.exists():
        actions.append(".env already exists")
    else:
        actions.append("missing .env.example; could not create .env")

    if not config_file.exists() and config_example.exists():
        shutil.copyfile(config_example, config_file)
        actions.append("created config.yaml from config.example.yaml")
    elif config_file.exists():
        actions.append("config.yaml already exists")
    else:
        actions.append("missing config.example.yaml; could not create config.yaml")

    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        actions.append("created data directory")
    else:
        actions.append("data directory already exists")

    return actions


def install_dependencies(root: Path, python_exe: str = sys.executable) -> None:
    requirements = root / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError("requirements.txt not found")
    subprocess.check_call([python_exe, "-m", "pip", "install", "-r", str(requirements)])
    subprocess.check_call([python_exe, "-m", "playwright", "install", "chromium"])


def _print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="First-time setup helper for image automation")
    parser.add_argument("--check-only", action="store_true", help="Check files and keys without installing dependencies")
    parser.add_argument("--skip-install", action="store_true", help="Create local files but skip pip and Playwright install")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    _print_header("Gemini Lovart Image Automation setup")

    if sys.version_info < (3, 12):
        print("Python 3.12 or newer is required. Please install Python 3.12/3.13 and run this setup again.")
        return 1

    print(f"Python OK: {sys.version.split()[0]}")
    for action in ensure_local_setup_files(root):
        print(f"- {action}")

    if not args.check_only and not args.skip_install:
        _print_header("Installing Python dependencies")
        install_dependencies(root)
        print("- dependencies installed")
        print("- Playwright Chromium installed")

    missing_keys = missing_or_placeholder_env_keys(root / ".env")
    workbook = root / "data" / "products.xlsx"

    _print_header("Next steps")
    if missing_keys:
        print("Fill these values in .env:")
        for key in missing_keys:
            print(f"- {key}")
    else:
        print("- .env API keys look filled")

    if workbook.exists():
        print("- data/products.xlsx found")
    else:
        print("- Put your Excel workbook at data/products.xlsx, or change excel.path in config.yaml")

    print("- In Excel, column H is reference_images_are_product.")
    print("  Fill Chinese YES for same-product references, or Chinese NO for style-only references.")
    print("- Run: python main.py --dry-run --limit 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
