import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from setup_wizard import (
    ensure_local_setup_files,
    missing_or_placeholder_env_keys,
    optional_or_placeholder_env_keys,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class SetupWizardTests(unittest.TestCase):
    def test_ensure_local_setup_files_creates_templates_and_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env.example").write_text("GEMINI_API_KEY=your_gemini_api_key\n", encoding="utf-8")
            shutil.copyfile(REPOSITORY_ROOT / "config.example.yaml", root / "config.example.yaml")

            actions = ensure_local_setup_files(root)

            self.assertTrue((root / ".env").exists())
            self.assertTrue((root / "config.yaml").exists())
            self.assertTrue((root / "data").is_dir())
            created_config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
            self.assertIn("prompt_settings", created_config)
            self.assertEqual(created_config["gemini_api"]["model"], "gemini-2.5-flash-lite")
            self.assertEqual(created_config["nvidia_api"]["model"], "moonshotai/kimi-k2.5")
            self.assertIn("created .env from .env.example", actions)
            self.assertIn("created config.yaml from config.example.yaml", actions)

    def test_missing_or_placeholder_env_keys_requires_only_lovart_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join([
                    "GEMINI_API_KEY=your_gemini_api_key",
                    "NVIDIA_API_KEY=",
                    "LOVART_ACCESS_KEY=",
                    "LOVART_SECRET_KEY=real-secret",
                ]),
                encoding="utf-8",
            )

            missing = missing_or_placeholder_env_keys(env_path)

            self.assertEqual(missing, ["LOVART_ACCESS_KEY"])

    def test_optional_or_placeholder_env_keys_reports_prompt_source_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join([
                    "GEMINI_API_KEY=your_gemini_api_key",
                    "NVIDIA_API_KEY=",
                    "LOVART_ACCESS_KEY=real-access",
                    "LOVART_SECRET_KEY=real-secret",
                ]),
                encoding="utf-8",
            )

            optional = optional_or_placeholder_env_keys(env_path)

            self.assertEqual(optional, ["GEMINI_API_KEY", "NVIDIA_API_KEY"])


if __name__ == "__main__":
    unittest.main()
