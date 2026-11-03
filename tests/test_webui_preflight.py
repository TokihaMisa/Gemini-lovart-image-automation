from pathlib import Path
from unittest.mock import Mock, patch

import openpyxl
import pytest

import webui

pytestmark = pytest.mark.real_preflight


def _excel(path: Path, rows) -> Path:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["商品ID", "中文名", "图片尺寸", "语言", "卖点"])
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    return path


def _config(excel_path: Path) -> dict:
    return {
        "excel": {
            "path": str(excel_path),
            "sheet": 0,
            "columns": {"id": "A", "name_cn": "B", "language": "D", "selling_points": "E"},
        }
    }


def _problems(tmp_path, config, **overrides):
    values = {
        "prompt_source": "gemini_browser",
        "support_provider": "openai_image",
        "detail_provider": "openai_image",
        "gemini_key": "",
        "nvidia_key": "",
        "lovart_access": "",
        "lovart_secret": "",
        "output_dir": str(tmp_path / "output"),
    }
    values.update(overrides)
    return webui._preflight_problems(config, **values)


def test_preflight_passes_with_usable_excel_and_needed_keys(tmp_path):
    excel = _excel(tmp_path / "products.xlsx", [["SKU-1", "保温杯", "1:1", "英语", ""]])

    assert _problems(tmp_path, _config(excel)) == []


def test_preflight_reports_missing_excel(tmp_path):
    problems = _problems(tmp_path, _config(tmp_path / "missing.xlsx"))

    assert len(problems) == 1
    assert "找不到商品表格" in problems[0]


def test_preflight_reports_excel_without_usable_products(tmp_path):
    excel = _excel(tmp_path / "products.xlsx", [["SKU-1", "", "1:1", "英语", ""]])

    problems = _problems(tmp_path, _config(excel))

    assert len(problems) == 1
    assert "没有可处理的商品" in problems[0]


def test_preflight_reports_unreadable_excel(tmp_path):
    broken = tmp_path / "products.xlsx"
    broken.write_text("not a workbook", encoding="utf-8")

    problems = _problems(tmp_path, _config(broken))

    assert len(problems) == 1
    assert "打不开" in problems[0]


def test_preflight_reports_missing_keys_for_selected_services(tmp_path):
    excel = _excel(tmp_path / "products.xlsx", [["SKU-1", "保温杯", "1:1", "英语", ""]])

    problems = _problems(
        tmp_path,
        _config(excel),
        prompt_source="gemini_api",
        support_provider="lovart",
        lovart_access="ak",
    )

    assert len(problems) == 2
    assert "GEMINI_API_KEY" in problems[0]
    assert "LOVART" in problems[1]


def test_preflight_reports_unwritable_output_folder(tmp_path):
    excel = _excel(tmp_path / "products.xlsx", [["SKU-1", "保温杯", "1:1", "英语", ""]])
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("file", encoding="utf-8")

    problems = _problems(tmp_path, _config(excel), output_dir=str(blocker))

    assert len(problems) == 1
    assert "输出文件夹" in problems[0]


def test_run_process_stops_before_launch_when_preflight_fails(tmp_path):
    config = _config(tmp_path / "missing.xlsx")
    with (
        patch("webui.subprocess.Popen") as popen,
        patch("webui.load_config", return_value=config),
        patch("webui._save_config_and_env_transaction"),
    ):
        updates = list(
            webui.run_process(
                None,
                str(tmp_path / "output"),
                "gemini_api",
                "gemini-model",
                "unlimited",
                "auto",
                "https://gemini.test/v1beta",
                "https://nvidia.test/v1",
                "gemini-key",
                "nvidia-key",
                "ak",
                "sk",
            )
        )

    popen.assert_not_called()
    assert "开始前检查未通过" in updates[-1]
    assert "找不到商品表格" in updates[-1]
