import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from failed_retry import (
    RETRY_MODE_FINITE,
    RETRY_MODE_INFINITE,
    RETRY_MODE_OFF,
    FailedRetryPolicy,
    classify_retry_failure,
)
from utils import write_run_summary


class _Product:
    def __init__(self, product_id):
        self.id = product_id


class _Lovart:
    def __init__(self, policy):
        self.failed_retry_policy = policy


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


class FailedRetryQueueTests(unittest.TestCase):
    def setUp(self):
        main._shutdown_requested = False

    def tearDown(self):
        main._shutdown_requested = False

    def test_finite_policy_retries_selected_error_after_full_queue(self):
        products = [_Product("SKU-1"), _Product("SKU-2")]
        calls = []

        def process_once(current, _gemini, _lovart, _logger, run_dir, **_kwargs):
            calls.append([product.id for product in current])
            if len(calls) == 1:
                rows = [
                    {"product_id": "SKU-1", "status": "success", "error": ""},
                    {
                        "product_id": "SKU-2",
                        "status": "failed",
                        "error": "Lovart 服务暂时不可用，请稍后重试。",
                    },
                ]
            elif len(calls) == 2:
                rows = [{
                    "product_id": "SKU-2",
                    "status": "failed",
                    "error": "Lovart API 失败：Unknown API error",
                }]
            else:
                rows = [{"product_id": "SKU-2", "status": "success", "error": ""}]
            write_run_summary(run_dir, rows)
            return 0, 0, 0, 0

        policy = FailedRetryPolicy(
            mode=RETRY_MODE_FINITE,
            rounds=2,
            delay=0,
            error_types=("lovart_service",),
        )
        logger = _Logger()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "main._process_products_once", side_effect=process_once
        ):
            result = main._process_products(
                products, object(), _Lovart(policy), logger, Path(tmp), resume=True
            )
            final_rows = main._read_run_summary(Path(tmp))

        self.assertEqual(calls, [["SKU-1", "SKU-2"], ["SKU-2"], ["SKU-2"]])
        self.assertEqual(result, (2, 0, 0, 0))
        self.assertEqual([row["status"] for row in final_rows], ["success", "success"])
        self.assertEqual(len(logger.warnings), 2)

    def test_process_round_rebases_excel_images_after_category_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "output"
            product_id = "NW2605411N"
            stale_image = base / "3_处理中" / product_id / "image_5.png"
            current_dir = base / "4_异常" / product_id
            current_dir.mkdir(parents=True)
            current_image = current_dir / "image_5.png"
            current_image.write_bytes(b"image")
            product = SimpleNamespace(
                id=product_id,
                name_cn="Product",
                image_size="1:1",
                language="English",
                selling_points="",
                image_paths=[str(stale_image)],
                reference_images_are_product=False,
            )
            registry = SimpleNamespace(get=lambda _name: object())
            routing = SimpleNamespace(
                support_provider="lovart",
                detail_provider="lovart",
                detail_page_count=3,
            )
            observed_paths = []

            def stop_after_image_resolution(paths):
                observed_paths.append(list(paths))
                raise RuntimeError("stop after image resolution")

            with patch.dict(
                os.environ, {"LOVART_OUTPUT_DIR": str(base)}
            ), patch(
                "main.split_image_roles", side_effect=stop_after_image_resolution
            ):
                main._process_products_once(
                    [product],
                    object(),
                    object(),
                    Mock(),
                    Path(tmp) / "run",
                    resume=True,
                    image_registry=registry,
                    routing=routing,
                )

        self.assertEqual(observed_paths, [[str(current_image)]])
        self.assertEqual(product.image_paths, [str(current_image)])

    def test_explicit_policy_retries_without_constructing_lovart(self):
        product = _Product("SKU-OPENAI")
        calls = []

        def process_once(_current, _gemini, _lovart, _logger, run_dir, **_kwargs):
            calls.append(1)
            failed = len(calls) == 1
            write_run_summary(run_dir, [{
                "product_id": product.id,
                "status": "failed" if failed else "success",
                "error": (
                    "screen 4: GPT Image API is temporarily unavailable."
                    if failed else ""
                ),
            }])
            return 0, 0, 0, 0

        policy = FailedRetryPolicy(
            mode=RETRY_MODE_FINITE,
            rounds=2,
            delay=0,
            error_types=("network",),
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "main._process_products_once", side_effect=process_once
        ):
            result = main._process_products(
                [product],
                object(),
                None,
                _Logger(),
                Path(tmp),
                failed_retry_policy=policy,
            )

        self.assertEqual(calls, [1, 1])
        self.assertEqual(result, (1, 0, 0, 0))

    def test_sub2api_scheduler_rechecks_live_tasks_and_caps_active_products_at_two(self):
        """Catches live tasks being abandoned until the next full application run."""
        products = [_Product("SKU-A"), _Product("SKU-B"), _Product("SKU-C")]
        remaining = {
            "SKU-A": [
                "openai_image_task_still_running",
                "openai_image_task_still_running",
                "success",
            ],
            "SKU-B": ["openai_image_task_still_running", "success"],
            "SKU-C": ["success"],
        }
        calls = []
        resume_flags = []
        clock = {"now": 0.0}

        def process_once(current, _gemini, _lovart, _logger, run_dir, **_kwargs):
            calls.append([product.id for product in current])
            resume_flags.append(bool(_kwargs.get("resume")))
            rows = []
            for product in current:
                status = remaining[product.id].pop(0)
                rows.append({
                    "product_id": product.id,
                    "status": status,
                    "failure_code": (
                        "task_still_running"
                        if status == "openai_image_task_still_running"
                        else ""
                    ),
                    "error": "",
                })
                if product.id == "SKU-B" and status == "success":
                    clock["now"] = 25.0
            write_run_summary(run_dir, rows)
            return 0, 0, 0, 0

        provider = SimpleNamespace(api=SimpleNamespace(config=SimpleNamespace(
            profile="sub2api",
            max_parallel_tasks=2,
            task_recheck_interval=10,
        )))
        registry = SimpleNamespace(get=lambda _name: provider)
        routing = SimpleNamespace(
            support_provider="openai_image",
            detail_provider="openai_image",
            detail_page_count=4,
        )
        policy = FailedRetryPolicy(mode=RETRY_MODE_OFF, rounds=1, delay=0)

        def advance(delay):
            clock["now"] += delay
            return True

        with tempfile.TemporaryDirectory() as tmp, patch(
            "main._process_products_once", side_effect=process_once
        ), patch(
            "main.time.monotonic", side_effect=lambda: clock["now"]
        ), patch(
            "main._wait_before_failed_retry", side_effect=advance
        ):
            result = main._process_products(
                products,
                object(),
                None,
                _Logger(),
                Path(tmp),
                resume=False,
                image_registry=registry,
                routing=routing,
                failed_retry_policy=policy,
            )
            final_rows = main._read_run_summary(Path(tmp))

        self.assertEqual(
            calls,
            [["SKU-A"], ["SKU-B"], ["SKU-A"], ["SKU-B"], ["SKU-A"], ["SKU-C"]],
        )
        self.assertEqual(resume_flags, [False, False, True, True, True, False])
        self.assertEqual(result, (3, 0, 0, 0))
        self.assertEqual([row["status"] for row in final_rows], ["success"] * 3)

    def test_sub2api_scheduler_counts_persisted_live_tasks_before_new_submissions(self):
        """Catches restart recovery temporarily exceeding the configured remote cap."""
        products = [
            _Product("SKU-NEW-A"),
            _Product("SKU-NEW-B"),
            _Product("SKU-LIVE-A"),
            _Product("SKU-LIVE-B"),
        ]
        calls = []
        resume_flags = []

        def process_once(current, _gemini, _lovart, _logger, run_dir, **kwargs):
            product = current[0]
            calls.append(product.id)
            resume_flags.append(bool(kwargs.get("resume")))
            write_run_summary(run_dir, [{
                "product_id": product.id,
                "status": "success",
                "error": "",
            }])
            return 1, 0, 0, 0

        provider = SimpleNamespace(api=SimpleNamespace(config=SimpleNamespace(
            profile="sub2api",
            max_parallel_tasks=2,
            task_recheck_interval=0,
        )))
        registry = SimpleNamespace(get=lambda _name: provider)
        routing = SimpleNamespace(
            support_provider="openai_image",
            detail_provider="openai_image",
            detail_page_count=4,
        )
        policy = FailedRetryPolicy(mode=RETRY_MODE_OFF, rounds=1, delay=0)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            for product_id in ("SKU-LIVE-A", "SKU-LIVE-B"):
                product_dir = output / product_id
                product_dir.mkdir(parents=True)
                (product_dir / "status.json").write_text(
                    '{"openai_image_still_running": true, '
                    '"support_task_checkpoints": {'
                    '"white_bg": {"task_id": "live-task", "state": "running"}}}',
                    encoding="utf-8",
                )
            with patch.dict(os.environ, {"LOVART_OUTPUT_DIR": str(output)}), patch(
                "main._process_products_once", side_effect=process_once
            ):
                result = main._process_products(
                    products,
                    object(),
                    None,
                    _Logger(),
                    Path(tmp),
                    resume=False,
                    image_registry=registry,
                    routing=routing,
                    failed_retry_policy=policy,
                )

        self.assertEqual(
            calls,
            ["SKU-LIVE-A", "SKU-LIVE-B", "SKU-NEW-A", "SKU-NEW-B"],
        )
        self.assertEqual(resume_flags, [True, True, False, False])
        self.assertEqual(result, (4, 0, 0, 0))

    def test_infinite_policy_has_no_fixed_cap_and_stops_after_success(self):
        product = _Product("SKU-1")
        calls = []

        def process_once(_current, _gemini, _lovart, _logger, run_dir, **_kwargs):
            calls.append(1)
            status = "success" if len(calls) == 5 else "failed"
            error = "network connection failed" if status == "failed" else ""
            write_run_summary(
                run_dir,
                [{"product_id": product.id, "status": status, "error": error}],
            )
            return 0, 0, 0, 0

        policy = FailedRetryPolicy(
            mode=RETRY_MODE_INFINITE,
            rounds=1,
            delay=0,
            error_types=("network",),
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "main._process_products_once", side_effect=process_once
        ):
            result = main._process_products(
                [product], object(), _Lovart(policy), _Logger(), Path(tmp)
            )

        self.assertEqual(len(calls), 5)
        self.assertEqual(result, (1, 0, 0, 0))

    def test_disabled_or_unselected_category_runs_only_once(self):
        for policy in (
            FailedRetryPolicy(mode=RETRY_MODE_OFF, rounds=2, delay=0),
            FailedRetryPolicy(
                mode=RETRY_MODE_FINITE,
                rounds=3,
                delay=0,
                error_types=("gemini_upload",),
            ),
        ):
            calls = []

            def process_once(_current, _gemini, _lovart, _logger, run_dir, **_kwargs):
                calls.append(1)
                write_run_summary(run_dir, [{
                    "product_id": "SKU-1",
                    "status": "failed",
                    "error": "Unable to connect to Lovart: connection reset",
                }])
                return 0, 1, 0, 0

            with tempfile.TemporaryDirectory() as tmp, patch(
                "main._process_products_once", side_effect=process_once
            ):
                result = main._process_products(
                    [_Product("SKU-1")], object(), _Lovart(policy), _Logger(), Path(tmp)
                )

            self.assertEqual(calls, [1])
            self.assertEqual(result, (0, 1, 0, 0))

    def test_partial_detail_failure_is_retried_and_replaced_by_complete_summary(self):
        product = _Product("SKU-PARTIAL")
        calls = []

        def process_once(_current, _gemini, _lovart, _logger, run_dir, **_kwargs):
            calls.append(1)
            if len(calls) == 1:
                row = {
                    "product_id": product.id,
                    "status": "failed",
                    "error": "screen 2: temporary upstream failure",
                    "artifact_count": 2,
                    "partial_complete": True,
                }
            else:
                row = {
                    "product_id": product.id,
                    "status": "success",
                    "error": "",
                    "artifact_count": 3,
                    "partial_complete": False,
                }
            write_run_summary(run_dir, [row])
            return 0, 0, 0, 0

        policy = FailedRetryPolicy(
            mode=RETRY_MODE_FINITE,
            rounds=1,
            delay=0,
            error_types=("other",),
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "main._process_products_once", side_effect=process_once
        ):
            result = main._process_products(
                [product], object(), _Lovart(policy), _Logger(), Path(tmp)
            )
            final_rows = main._read_run_summary(Path(tmp))

        self.assertEqual(calls, [1, 1])
        self.assertEqual(result, (1, 0, 0, 0))
        self.assertEqual(final_rows[0]["status"], "success")
        self.assertEqual(final_rows[0]["artifact_count"], 3)

    def test_permanent_failure_never_falls_into_other_category(self):
        self.assertIsNone(classify_retry_failure({
            "status": "failed",
            "error": "API key authentication failed. Please check your key.",
        }))
        self.assertIsNone(classify_retry_failure({
            "status": "failed",
            "error": "Gemini 未登录，请先登录后再试。",
        }))
        self.assertIsNone(classify_retry_failure({
            "status": "failed",
            "error": (
                "GPT Image 同步请求已提交但未收到响应，结果未知；"
                "已停止自动重试。请先检查平台后台。"
            ),
        }))
        self.assertEqual(classify_retry_failure({
            "status": "failed",
            "error": "Unexpected response shape",
        }), "other")

    def test_stable_paid_create_and_live_task_codes_never_retry(self):
        for code in ("ambiguous_submission", "submission_unknown", "task_still_running"):
            self.assertIsNone(classify_retry_failure({
                "status": "failed",
                "failure_code": code,
                "error": "localized text deliberately contains network timeout words",
            }))

    def test_final_provider_task_failure_can_enter_product_retry_policy(self):
        self.assertEqual(classify_retry_failure({
            "status": "failed",
            "failure_code": "task_failed",
            "error": "localized provider rejection",
        }), "other")

    def test_infinite_policy_does_not_retry_ambiguous_paid_create(self):
        calls = []

        def process_once(_current, _gemini, _lovart, _logger, run_dir, **_kwargs):
            calls.append(1)
            if len(calls) > 1:
                raise AssertionError("ambiguous paid create was automatically retried")
            write_run_summary(run_dir, [{
                "product_id": "SKU-AMBIGUOUS",
                "status": "failed",
                "failure_code": "ambiguous_submission",
                "error": "network timeout translated differently",
            }])
            return 0, 1, 0, 0

        policy = FailedRetryPolicy(
            mode=RETRY_MODE_INFINITE,
            rounds=1,
            delay=0,
            error_types=("network", "timeout", "other"),
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "main._process_products_once", side_effect=process_once
        ):
            result = main._process_products(
                [_Product("SKU-AMBIGUOUS")],
                object(),
                None,
                _Logger(),
                Path(tmp),
                failed_retry_policy=policy,
            )

        self.assertEqual(calls, [1])
        self.assertEqual(result, (0, 1, 0, 0))

    def test_shutdown_interrupts_retry_delay(self):
        main._shutdown_requested = True
        with patch("main.time.sleep") as sleep:
            self.assertFalse(main._wait_before_failed_retry(30))
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
