import tempfile
import unittest
from pathlib import Path

from setup_wizard import (
    ensure_local_setup_files,
    missing_or_placeholder_env_keys,
)


class SetupWizardTests(unittest.TestCase):
    def test_ensure_local_setup_files_creates_templates_and_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env.example").write_text("GEMINI_API_KEY=your_gemini_api_key\n", encoding="utf-8")
            (root / "config.example.yaml").write_text("excel:\n  path: data\\\\products.xlsx\n", encoding="utf-8")

            actions = ensure_local_setup_files(root)

            self.assertTrue((root / ".env").exists())
            self.assertTrue((root / "config.yaml").exists())
            self.assertTrue((root / "data").is_dir())
            self.assertIn("created .env from .env.example", actions)
            self.assertIn("created config.yaml from config.example.yaml", actions)

    def test_missing_or_placeholder_env_keys_reports_values_to_fill(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join([
                    "GEMINI_API_KEY=your_gemini_api_key",
                    "NVIDIA_API_KEY=real-nvidia-key",
                    "LOVART_ACCESS_KEY=",
                    "LOVART_SECRET_KEY=real-secret",
                ]),
                encoding="utf-8",
            )

            missing = missing_or_placeholder_env_keys(env_path)

            self.assertEqual(missing, ["GEMINI_API_KEY", "LOVART_ACCESS_KEY"])


if __name__ == "__main__":
    unittest.main()
