import socket
import ssl
import unittest
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError

from network_retry import (
    RetryKind,
    RetryPolicy,
    classify_network_error,
    retry_policy_from_config,
    run_with_retry,
    safe_retry_message,
)


class NetworkRetryTests(unittest.TestCase):
    def test_missing_config_uses_approved_balanced_policy(self):
        policy = retry_policy_from_config({})
        self.assertEqual(policy.network_attempts, 5)
        self.assertEqual(policy.page_ready_timeout, 90)
        self.assertEqual(policy.product_attempts, 2)
        self.assertEqual(policy.retry_delays, (3, 6, 12, 20))

    def test_classifies_transient_browser_and_transport_failures(self):
        cases = [
            TimeoutError(),
            socket.timeout(),
            ConnectionResetError(),
            URLError("temporary DNS failure"),
            ssl.SSLError("protocol interrupted"),
            IncompleteRead(b"partial"),
            RuntimeError("net::ERR_NETWORK_CHANGED"),
            RuntimeError("net::ERR_SSL_PROTOCOL_ERROR"),
            HTTPError("https://example.test", 429, "rate", {}, None),
            HTTPError("https://example.test", 503, "down", {}, None),
        ]
        for exc in cases:
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(classify_network_error(exc), RetryKind.TRANSIENT)

    def test_classifies_permanent_tls_and_auth_without_retry(self):
        for exc in (
            ssl.SSLCertVerificationError("certificate verify failed"),
            RuntimeError("net::ERR_CERT_AUTHORITY_INVALID"),
            RuntimeError("net::ERR_CERT_COMMON_NAME_INVALID"),
            RuntimeError("net::ERR_CERT_DATE_INVALID"),
        ):
            self.assertEqual(classify_network_error(exc), RetryKind.PERMANENT_TLS)
        self.assertEqual(
            classify_network_error(HTTPError("https://x", 403, "denied", {}, None)),
            RetryKind.AUTH,
        )

    def test_unknown_ssl_error_is_not_retried(self):
        self.assertEqual(
            classify_network_error(ssl.SSLError("unexpected ssl library failure")),
            RetryKind.OTHER,
        )
        self.assertEqual(
            classify_network_error(HTTPError("https://x", 404, "missing", {}, None)),
            RetryKind.NOT_FOUND,
        )

    def test_wrapped_network_reasons_are_classified_by_their_safe_root_cause(self):
        certificate = URLError(ssl.SSLCertVerificationError("certificate verify failed"))
        unknown_ssl = URLError(ssl.SSLError("unexpected ssl library failure"))
        permission = URLError(PermissionError("access denied"))
        connection = URLError(ConnectionResetError("connection reset"))
        dns = URLError(socket.gaierror("name resolution failed"))
        cycle = URLError("unknown")
        cycle.reason = cycle

        self.assertEqual(classify_network_error(certificate), RetryKind.PERMANENT_TLS)
        self.assertEqual(classify_network_error(unknown_ssl), RetryKind.OTHER)
        self.assertEqual(classify_network_error(permission), RetryKind.OTHER)
        self.assertEqual(classify_network_error(connection), RetryKind.TRANSIENT)
        self.assertEqual(classify_network_error(dns), RetryKind.TRANSIENT)
        self.assertEqual(classify_network_error(cycle), RetryKind.OTHER)

    def test_retry_executor_uses_exact_delays_and_returns_success(self):
        calls, delays, notices = [], [], []

        def operation():
            calls.append(len(calls) + 1)
            if len(calls) < 5:
                raise ConnectionResetError("secret must not be logged")
            return "ok"

        result = run_with_retry(
            operation,
            RetryPolicy(),
            on_retry=lambda notice: notices.append(notice),
            sleep=lambda delay: delays.append(delay),
        )
        self.assertEqual(result, "ok")
        self.assertEqual(calls, [1, 2, 3, 4, 5])
        self.assertEqual(delays, [3, 6, 12, 20])
        self.assertTrue(all("secret" not in notice for notice in notices))

    def test_permanent_tls_is_not_retried(self):
        calls = []
        with self.assertRaises(ssl.SSLCertVerificationError):
            run_with_retry(
                lambda: calls.append(1) or (_ for _ in ()).throw(
                    ssl.SSLCertVerificationError("certificate verify failed")
                ),
                RetryPolicy(),
                sleep=lambda _delay: self.fail("must not sleep"),
            )
        self.assertEqual(calls, [1])

    def test_retry_notice_contains_progress_but_not_raw_exception(self):
        message = safe_retry_message(RetryKind.TRANSIENT, 2, 5, 6)
        self.assertIn("第 2/5 次", message)
        self.assertIn("6 秒后重试", message)
