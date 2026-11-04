import unittest

import app


class _FailingOutput:
    encoding = "utf-8"

    def __init__(self, errno_value):
        self.errno_value = errno_value

    def write(self, _text):
        raise OSError(self.errno_value, "simulated output failure")

    def flush(self):
        raise OSError(self.errno_value, "simulated output failure")


class ResilientOutputTests(unittest.TestCase):
    def test_disconnected_windows_output_pipe_does_not_abort_background_task(self):
        output = app.ResilientOutputStream(_FailingOutput(22))

        self.assertEqual(output.write("status update"), len("status update"))
        output.flush()

    def test_unrelated_output_error_is_not_hidden(self):
        output = app.ResilientOutputStream(_FailingOutput(5))

        with self.assertRaises(OSError):
            output.write("status update")


if __name__ == "__main__":
    unittest.main()


class Utf8OutputTests(unittest.TestCase):
    def test_gbk_pipe_is_switched_to_utf8_so_fatal_error_marks_print(self):
        import io

        raw = io.BytesIO()
        # What a packaged EXE gets on Chinese Windows: PYTHONIOENCODING is ignored.
        stream = io.TextIOWrapper(raw, encoding="gbk", newline="\n")
        app.use_utf8_output([stream])
        stream.write("❌ 核心程序发生致命错误 🧠\n")
        stream.flush()

        self.assertEqual(raw.getvalue().decode("utf-8"), "❌ 核心程序发生致命错误 🧠\n")

    def test_streams_without_reconfigure_are_left_alone(self):
        app.use_utf8_output([object(), None])
