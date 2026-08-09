import base64
from pathlib import Path


VALID_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
)


def write_valid_png(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(VALID_PNG_BASE64))
    return str(path)


def write_truncated_png(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(VALID_PNG_BASE64)[:-12])
    return str(path)
