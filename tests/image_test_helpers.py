import base64
from pathlib import Path


VALID_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk/x8AAusB9Y9Z4WQAAAAASUVORK5CYII="
)


def write_valid_png(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(VALID_PNG_BASE64))
    return str(path)
