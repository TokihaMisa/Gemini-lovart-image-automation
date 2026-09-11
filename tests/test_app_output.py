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
