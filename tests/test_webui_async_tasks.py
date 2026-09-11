import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

import webui
from openai_image_api import GeneratedImage, ImageTaskSnapshot, OpenAIImageAPI


def _task(
    task_id="provider-private-task-12345678",
    *,
    state="running",
    is_final=False,
    progress="42%",
    status="rendering",
):
    return ImageTaskSnapshot(
        task_id=task_id,
        state=state,
        is_final=is_final,
        task_created_at=1_787_200_000.0,
        progress=progress,
        status=status,
        result_url="https://cdn.example/result.png" if is_final else "",
    )


def test_transport_test_edit_forwards_callbacks_to_async_generate_pipeline(tmp_path):
    client = object.__new__(OpenAIImageAPI)
    client.generate_edit = Mock(
        return_value=GeneratedImage(
            local_path=str(tmp_path / "result.png"),
            model="gpt-image-2",
            task=_task(state="success", is_final=True, progress="100%"),
        )
    )
    status_callback = Mock()
    task_callback = Mock()
    display_callback = Mock()

    result = client.test_edit(
        tmp_path,
        status_callback=status_callback,
        task_callback=task_callback,
        display_callback=display_callback,
    )

    assert result.local_path.endswith("result.png")
    assert client.generate_edit.call_args.kwargs["status_callback"] is status_callback
    assert client.generate_edit.call_args.kwargs["task_callback"] is task_callback
    assert client.generate_edit.call_args.kwargs["display_callback"] is display_callback


@patch("webui.OpenAIImageAPI")
def test_paid_test_streams_async_task_progress_and_masks_sensitive_values(api_cls):
    secret = "paid-test-secret"
    full_task_id = "provider-private-task-12345678"

    def run_test(_output_dir, *, status_callback, task_callback, **_kwargs):
        task_callback(_task(full_task_id, progress="0%", status="accepted"))
        status_callback(
            "⏳ GPT Image rendering · 42% · 已等待 17 秒 · 任务 …12345678"
        )
        status_callback("✅ GPT Image 生成完成，正在安全下载图片")
        return GeneratedImage(
            local_path="output/.api-tests/openai-image-test.png",
            model="gpt-image-2",
            task=_task(
                full_task_id,
                state="success",
                is_final=True,
                progress="100%",
                status="success",
            ),
        )

    api_cls.return_value.test_edit.side_effect = run_test

    accepted, disabled = webui.begin_openai_image_test()
    updates = list(
        webui.test_openai_image_edit(
            secret, "https://api.lk888.ai", "gpt-image-2", "1K"
        )
    )
    enabled = webui.reset_openai_image_test_button()
    rendered = "\n".join([accepted, *updates])

    assert "正在上传" in accepted
    assert disabled["interactive"] is False
    assert "任务 …12345678" in rendered
    assert "42%" in rendered
    assert "rendering" in rendered
    assert "已等待 17 秒" in rendered
    assert "正在安全下载图片" in rendered
    assert "测试成功" in updates[-1]
    assert enabled["interactive"] is True
    assert full_task_id not in rendered
    assert secret not in rendered


@patch("webui.OpenAIImageAPI")
def test_paid_test_sanitizes_terminal_error_and_button_can_be_reenabled(api_cls):
    secret = "paid-test-secret"
    full_task_id = "provider-private-task-12345678"

    def run_test(_output_dir, *, status_callback, task_callback, **_kwargs):
        task_callback(_task(full_task_id, progress="10%", status="queued"))
        raise ValueError(f"gateway rejected {secret} for {full_task_id}")

    api_cls.return_value.test_edit.side_effect = run_test

    updates = list(
        webui.test_openai_image_edit(
            secret, "https://api.lk888.ai/v1", "gpt-image-2", "1K"
        )
    )
    enabled = webui.reset_openai_image_test_button()
    rendered = "\n".join(updates)

    assert "测试失败" in updates[-1]
    assert "任务 …12345678" in rendered
    assert secret not in rendered
    assert full_task_id not in rendered
    assert enabled["interactive"] is True


@patch("webui.OpenAIImageAPI")
def test_selected_paid_test_uses_sub2api_profile_and_sync_protocol(api_cls):
    api_cls.return_value.test_edit.return_value = GeneratedImage(
        local_path="output/.api-tests/sub2api-test.png",
        model="gpt-image-2",
        task=_task("", state="success", is_final=True, progress="100%"),
    )

    updates = list(webui.test_selected_openai_image_edit(
        "sub2api",
        "wending-key",
        "https://wending.example/v1",
        "wending-model",
        "1K",
        False,
        False,
        "sub2-key",
        "https://sub2.example/v1",
        "sub2-model",
        "2K",
        True,
        False,
    ))

    selected = api_cls.call_args.args[0]
    assert selected.profile == "sub2api"
    assert selected.protocol == "sub2api_sync"
    assert selected.base_url == "https://sub2.example/v1"
    assert selected.model == "sub2-model"
    assert selected.api_key == "sub2-key"
    assert "测试成功" in updates[-1]
    assert "异步任务已提交" not in "\n".join(updates)


def _run_dashboard_frames(lines, output_dir):
    child = Mock()
    child.stdout = io.StringIO("".join(lines))
    child.poll.return_value = 0
    with (
        patch("webui.subprocess.Popen", return_value=child),
        patch("webui.load_config", return_value={
            "gemini_api": {"model": "gemini-model"},
            "nvidia_api": {"model": "nvidia-model"},
        }),
        patch("webui._save_config_and_env_transaction"),
    ):
        return list(
            webui.run_process(
                None,
                str(output_dir),
                "gemini_api",
                "gemini-model",
                "unlimited",
                "auto",
                "https://gemini.test/v1beta",
                "https://nvidia.test/v1",
                "gemini-key",
                "nvidia-key",
                "",
                "",
            )
        )


def test_dashboard_uses_bounded_thumbnail_payload_for_large_product_image(tmp_path):
    image_path = tmp_path / "large-product.bmp"
    Image.new("RGB", (1400, 1000), "#5a7d9a").save(image_path)
    assert image_path.stat().st_size > 4_000_000

    frames = _run_dashboard_frames(
        [
            "[UI_PRODUCT] "
            + json.dumps(
                {"id": "SKU-1", "name": "测试商品", "image": str(image_path)},
                ensure_ascii=False,
            )
            + "\n",
            '[UI_STATUS] {"id":"SKU-1","stage":"product","message":"处理中"}\n',
            '[UI_STATUS] {"id":"SKU-1","stage":"prompt","message":"生成提示词"}\n',
        ],
        tmp_path,
    )

    image_frames = [frame for frame in frames if "data:image/" in frame]
    assert image_frames
    assert max(len(frame.encode("utf-8")) for frame in image_frames) < 250_000


def test_dashboard_visible_set_is_bounded_and_keeps_current_product_first():
    products = {
        f"SKU-{index:04d}": {"status": "等待处理"}
        for index in range(1_000)
    }

    visible = webui._dashboard_visible_product_ids(
        products,
        current_pid="SKU-0999",
        recent_product_ids=[f"SKU-{index:04d}" for index in range(900, 999)],
    )

    assert len(visible) == 50
    assert visible[0] == "SKU-0999"
    assert len(set(visible)) == 50


def test_dashboard_batches_catalog_events_and_renders_only_card_limit(tmp_path):
    lines = [
        "[UI_PRODUCT] "
        + json.dumps(
            {"id": f"SKU-{index:03d}", "name": f"商品 {index}", "image": ""},
            ensure_ascii=False,
        )
        + "\n"
        for index in range(1_000)
    ]
    lines.append(
        '[UI_STATUS] {"id":"SKU-999","stage":"product","message":"处理中"}\n'
    )

    frames = _run_dashboard_frames(lines, tmp_path)
    final_frame = frames[-1]

    assert final_frame.count("class='status-card'") == 50
    assert "ID: SKU-999" in final_frame
    assert "显示 50 / 1000" in final_frame
    assert len(frames) <= 6


def test_product_logs_are_bounded_to_latest_entries():
    product = {}

    for index in range(100):
        webui._append_product_log(product, f"log-{index}")

    assert product["logs"] == [f"log-{index}" for index in range(80, 100)]


def test_source_mode_launches_through_resilient_app_wrapper(tmp_path):
    child = Mock()
    child.stdout = io.StringIO("")
    child.poll.return_value = 0
    with (
        patch("webui.subprocess.Popen", return_value=child) as popen,
        patch("webui.load_config", return_value={
            "gemini_api": {"model": "gemini-model"},
            "nvidia_api": {"model": "nvidia-model"},
        }),
        patch("webui._save_config_and_env_transaction"),
        patch.object(sys, "frozen", False, create=True),
    ):
        list(
            webui.run_process(
                None,
                str(tmp_path),
                "gemini_api",
                "gemini-model",
                "unlimited",
                "auto",
                "https://gemini.test/v1beta",
                "https://nvidia.test/v1",
                "gemini-key",
                "nvidia-key",
                "",
                "",
            )
        )

    argv = popen.call_args.args[0]
    assert Path(argv[1]).name == "app.py"
    assert argv[2] == "--run-main"


@pytest.mark.skipif(os.name != "nt", reason="Windows pipe cancellation regression")
def test_dashboard_disconnect_stops_output_reader_and_closes_pipe(tmp_path):
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        with (
            patch("webui.subprocess.Popen", return_value=child),
            patch("webui.load_config", return_value={
                "gemini_api": {"model": "gemini-model"},
                "nvidia_api": {"model": "nvidia-model"},
            }),
            patch("webui._save_config_and_env_transaction"),
        ):
            dashboard = webui.run_process(
                None,
                str(tmp_path),
                "gemini_api",
                "gemini-model",
                "unlimited",
                "auto",
                "https://gemini.test/v1beta",
                "https://nvidia.test/v1",
                "gemini-key",
                "nvidia-key",
                "",
                "",
            )
            next(dashboard)
            next(dashboard)
            next(dashboard)
            dashboard.close()
            deadline = time.monotonic() + 1.0
            while not child.stdout.closed and time.monotonic() < deadline:
                time.sleep(0.01)
            assert child.stdout.closed
            assert child.poll() is None
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)
        if child in webui.active_processes:
            webui.active_processes.remove(child)


def test_thumbnail_rotates_only_after_source_is_reduced(tmp_path):
    image_path = tmp_path / "large.jpg"
    Image.new("RGB", (1200, 900), "#5a7d9a").save(image_path)
    transpose_sizes = []

    def record_transpose_size(image):
        transpose_sizes.append(image.size)
        return image

    with patch("PIL.ImageOps.exif_transpose", side_effect=record_transpose_size):
        uri = webui._dashboard_thumbnail_data_uri(str(image_path))

    assert uri.startswith("data:image/jpeg;base64,")
    assert transpose_sizes
    assert max(transpose_sizes[0]) <= max(webui._DASHBOARD_THUMBNAIL_SIZE)


def test_dashboard_status_counts_include_background_running_and_skipped():
    counts = webui._dashboard_status_counts(
        {
            "RUNNING": {"stage": "support_scene", "still_running": True},
            "SKIPPED": {"stage": "skipped", "status": "已跳过"},
            "WAITING": {"status": "等待处理"},
        },
        current_pid="WAITING",
    )

    assert counts == {
        "total": 3,
        "waiting": 0,
        "running": 2,
        "success": 0,
        "failed": 0,
        "skipped": 1,
    }


@pytest.mark.parametrize(
    ("stage", "stage_label"),
    [
        ("support_white", "白底图"),
        ("support_scene", "场景图"),
        ("detail_screen_3", "详情图 3"),
    ],
)
def test_product_card_immediately_renders_real_async_stage_and_progress(
    tmp_path, stage, stage_label
):
    full_task_id = "provider-private-task-12345678"
    frames = _run_dashboard_frames(
        [
            '[UI_PRODUCT] {"id":"SKU-1","name":"测试商品","image":""}\n',
            "[UI_STATUS] "
            + json.dumps(
                {
                    "id": "SKU-1",
                    "stage": stage,
                    "message": "⏳ GPT Image rendering",
                    "progress": "42%",
                    "status": "rendering",
                    "elapsed_seconds": 17,
                    "task_suffix": "12345678",
                    "task_id": full_task_id,
                },
                ensure_ascii=False,
            )
            + "\n",
        ],
        tmp_path,
    )

    active_frames = [frame for frame in frames if stage_label in frame]
    assert active_frames
    assert "42%" in active_frames[0]
    assert "rendering" in active_frames[0]
    assert "已等待 17 秒" in active_frames[0]
    assert "任务 …12345678" in active_frames[0]
    assert full_task_id not in "\n".join(frames)


def test_product_card_replaces_live_poll_row_for_the_same_async_task(tmp_path):
    def status(progress, elapsed):
        return (
            "[UI_STATUS] "
            + json.dumps(
                {
                    "id": "SKU-1",
                    "stage": "support_scene",
                    "message": "⏳ GPT Image 异步任务正在处理",
                    "progress": progress,
                    "status": "running",
                    "elapsed_seconds": elapsed,
                    "task_suffix": "12345678",
                    "live_task": True,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    frames = _run_dashboard_frames(
        [
            '[UI_PRODUCT] {"id":"SKU-1","name":"测试商品","image":""}\n',
            status("10%", 5),
            status("45%", 10),
            status("90%", 15),
        ],
        tmp_path,
    )
    final_card = frames[-1]

    assert "平台进度 90%" in final_card
    assert "平台进度 10%" not in final_card
    assert "平台进度 45%" not in final_card
    assert final_card.count("data-live-task-status=") == 1


def test_product_card_keeps_one_refreshable_row_per_live_stage(tmp_path):
    def status(stage, progress):
        return (
            "[UI_STATUS] "
            + json.dumps(
                {
                    "id": "SKU-1",
                    "stage": stage,
                    "message": "GPT Image 异步任务正在处理",
                    "progress": progress,
                    "status": "running",
                    "live_task": True,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    frames = _run_dashboard_frames(
        [
            '[UI_PRODUCT] {"id":"SKU-1","name":"测试商品","image":""}\n',
            status("support_white", "10%"),
            status("support_scene", "20%"),
            status("support_white", "90%"),
        ],
        tmp_path,
    )
    final_card = frames[-1]

    assert "平台进度 10%" not in final_card
    assert "平台进度 90%" in final_card
    assert "平台进度 20%" in final_card
    assert final_card.count("data-live-task-status='support_white'") == 1
    assert final_card.count("data-live-task-status='support_scene'") == 1


def test_terminal_event_ends_live_row_refresh_for_a_later_retry(tmp_path):
    frames = _run_dashboard_frames(
        [
            '[UI_PRODUCT] {"id":"SKU-1","name":"测试商品","image":""}\n',
            '[UI_STATUS] {"id":"SKU-1","stage":"support_scene","message":"处理中","progress":"10%","live_task":true}\n',
            '[UI_FAIL] {"id":"SKU-1","reason":"第一次失败"}\n',
            '[UI_STATUS] {"id":"SKU-1","stage":"support_scene","message":"重新处理","progress":"20%","live_task":true}\n',
        ],
        tmp_path,
    )
    final_card = frames[-1]

    assert "平台进度 10%" in final_card
    assert "第一次失败" in final_card
    assert "平台进度 20%" in final_card
    assert final_card.count("data-live-task-status='support_scene'") == 2


def test_malformed_detail_screen_stage_is_not_treated_as_live(tmp_path):
    frames = _run_dashboard_frames(
        [
            '[UI_PRODUCT] {"id":"SKU-1","name":"测试商品","image":""}\n',
            '[UI_STATUS] {"id":"SKU-1","stage":"detail_screen_bad","message":"第一条","live_task":true}\n',
            '[UI_STATUS] {"id":"SKU-1","stage":"detail_screen_bad","message":"第二条","live_task":true}\n',
        ],
        tmp_path,
    )
    final_card = frames[-1]

    assert "第一条" in final_card
    assert "第二条" in final_card
    assert "data-live-task-status='detail_screen_bad'" not in final_card


def test_provider_neutral_stage_messages_keep_existing_append_behavior(tmp_path):
    frames = _run_dashboard_frames(
        [
            '[UI_PRODUCT] {"id":"SKU-1","name":"测试商品","image":""}\n',
            '[UI_STATUS] {"id":"SKU-1","stage":"support_scene","message":"Lovart 已提交"}\n',
            '[UI_STATUS] {"id":"SKU-1","stage":"support_scene","message":"Lovart 正在下载"}\n',
        ],
        tmp_path,
    )
    final_card = frames[-1]

    assert "Lovart 已提交" in final_card
    assert "Lovart 正在下载" in final_card
    assert "data-live-task-status=" not in final_card


def test_detail_polling_refreshes_generic_and_screen_rows_independently(tmp_path):
    def status(stage, progress):
        return (
            "[UI_STATUS] "
            + json.dumps(
                {
                    "id": "SKU-1",
                    "stage": stage,
                    "message": "GPT Image 异步任务正在处理",
                    "progress": progress,
                    "live_task": True,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    frames = _run_dashboard_frames(
        [
            '[UI_PRODUCT] {"id":"SKU-1","name":"测试商品","image":""}\n',
            status("detail", "10%"),
            status("detail_screen_1", "10%"),
            status("detail", "45%"),
            status("detail_screen_1", "45%"),
        ],
        tmp_path,
    )
    final_card = frames[-1]

    assert "平台进度 10%" not in final_card
    assert "平台进度 45%" in final_card
    assert final_card.count("data-live-task-status=") == 2
    assert final_card.count("data-live-task-status='detail'") == 1
    assert final_card.count("data-live-task-status='detail_screen_1'") == 1


def test_async_task_status_labels_provider_progress_without_sync_wording():
    rendered = webui._format_openai_image_ui_status(
        {
            "stage": "detail_screen_2",
            "display_status": "running",
            "progress": "45%",
            "elapsed_seconds": 17,
            "task_suffix": "12345678",
        },
        "GPT Image 异步任务正在处理",
    )

    assert "平台进度 45%" in rendered
    assert "同步请求" not in rendered


def test_product_card_reads_still_running_snapshot_as_resumable_not_failed(tmp_path):
    product_dir = tmp_path / "SKU-1"
    product_dir.mkdir()
    (product_dir / "status.json").write_text(
        json.dumps(
            {
                "openai_image_still_running": True,
                "openai_image_active_stage": "detail_screen_4",
                "openai_image_task_suffix": "12345678",
                "openai_image_next_poll_at": 105.0,
                "failed": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with patch("webui.time.time", return_value=100.0):
        frames = _run_dashboard_frames(
            ['[UI_PRODUCT] {"id":"SKU-1","name":"测试商品","image":""}\n'],
            tmp_path,
        )
    final_card = next(
        frame for frame in reversed(frames) if "5 秒后自动复查" in frame
    )

    assert "详情图 4" in final_card
    assert "任务 …12345678" in final_card
    assert "下次将继续查询" not in final_card
    assert "❌" not in final_card
    assert "pulse-glow 2s infinite" not in final_card


def test_product_card_redacts_full_task_id_from_fallback_status_message(tmp_path):
    full_task_id = "provider-private-task-12345678"
    frames = _run_dashboard_frames(
        [
            '[UI_PRODUCT] {"id":"SKU-1","name":"测试商品","image":""}\n',
            "[UI_STATUS] "
            + json.dumps(
                {
                    "id": "SKU-1",
                    "stage": "support_white",
                    "message": f"provider is still processing {full_task_id}",
                    "task_id": full_task_id,
                },
                ensure_ascii=False,
            )
            + "\n",
        ],
        tmp_path,
    )
    rendered = "\n".join(frames)

    assert full_task_id not in rendered
    assert "白底图" in rendered


def test_ui_status_redacts_encoded_and_case_transformed_key_and_task_id():
    full_task_id = "Private-Task-12345678"
    rendered = webui._format_openai_image_ui_status(
        {
            "stage": "detail_screen_1",
            "api_key": "TeSt-Key",
            "task_id": full_task_id,
            "status": "key=TEST%2DKEY task=Private%252DTask%252D12345678",
        },
        "provider=PRIVATE-TASK-12345678",
    )
    assert "TEST%2DKEY" not in rendered
    assert "Private%252DTask%252D12345678" not in rendered
    assert "PRIVATE-TASK-12345678" not in rendered


@patch("webui.OpenAIImageAPI")
def test_paid_test_resolves_saved_env_key_without_echoing_it(api_cls, tmp_path):
    saved_key = "saved-paid-test-secret"
    env_path = tmp_path / ".env"
    env_path.write_text(f"OPENAI_IMAGE_API_KEY={saved_key}\n", encoding="utf-8")
    api_cls.return_value.test_edit.return_value = GeneratedImage(
        local_path="output/.api-tests/openai-image-test.png",
        model="gpt-image-2",
        task=_task(state="success", is_final=True, progress="100%"),
    )

    updates = list(
        webui.test_openai_image_edit(
            "",
            "https://api.lk888.ai",
            "gpt-image-2",
            "1K",
            env_path=env_path,
        )
    )

    resolved_config = api_cls.call_args.args[0]
    assert resolved_config.api_key == saved_key
    assert "测试成功" in updates[-1]
    assert saved_key not in "\n".join(updates)


@patch("webui.OpenAIImageAPI")
def test_paid_test_explicit_clear_does_not_reuse_saved_env_key(api_cls, tmp_path):
    saved_key = "saved-paid-test-secret"
    env_path = tmp_path / ".env"
    env_path.write_text(f"OPENAI_IMAGE_API_KEY={saved_key}\n", encoding="utf-8")

    updates = list(
        webui.test_openai_image_edit(
            "",
            "https://api.lk888.ai",
            "gpt-image-2",
            "1K",
            False,
            True,
            env_path=env_path,
        )
    )

    assert "测试失败" in updates[-1]
    assert "密钥" in updates[-1]
    assert saved_key not in "\n".join(updates)
    api_cls.assert_not_called()
