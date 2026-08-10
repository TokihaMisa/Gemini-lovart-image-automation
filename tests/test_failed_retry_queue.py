import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        self.assertEqual(classify_retry_failure({
            "status": "failed",
            "error": "Unexpected response shape",
        }), "other")

    def test_shutdown_interrupts_retry_delay(self):
        main._shutdown_requested = True
        with patch("main.time.sleep") as sleep:
            self.assertFalse(main._wait_before_failed_retry(30))
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
