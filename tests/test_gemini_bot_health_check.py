import unittest

from gemini_bot import GeminiBot


class ProbeLocator:
    def __init__(self, *, visible=False, enabled=True, count=1):
        self.visible = visible
        self.enabled = enabled
        self._count = count
        self.clicks = 0

    @property
    def first(self):
        return self

    def count(self):
        return self._count

    def nth(self, _index):
        return self

    def is_visible(self, **_kwargs):
        return self.visible

    def is_enabled(self):
        return self.enabled

    def click(self, **_kwargs):
        self.clicks += 1


class ProbePage:
    def __init__(self, selectors):
        self.selectors = selectors
        self.requested_selectors = []

    @property
    def total_clicks(self):
        return sum(locator.clicks for locator in self.selectors.values())

    def locator(self, selector):
        self.requested_selectors.append(selector)
        return self.selectors.get(selector, ProbeLocator(count=0))


class FakeLogger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass


class GeminiBotHealthCheckTests(unittest.TestCase):
    def test_editor_probe_detects_production_editor_without_interaction(self):
        selector = (
            '[contenteditable="true"][role="textbox"], '
            'rich-textarea [contenteditable="true"], '
            '[contenteditable="true"], textarea, [role="textbox"]'
        )
        page = ProbePage({selector: ProbeLocator(visible=True)})
        bot = GeminiBot(page, {"gemini": {}}, FakeLogger())

        self.assertTrue(bot.health_check_editor())
        self.assertEqual(page.total_clicks, 0)

    def test_upload_control_probe_accepts_current_label_without_clicking(self):
        page = ProbePage(
            {'button[aria-label*="上传和工具"]': ProbeLocator(visible=True)}
        )
        bot = GeminiBot(page, {"gemini": {}}, FakeLogger())

        self.assertTrue(bot.health_check_upload_control())
        self.assertEqual(page.total_clicks, 0)

    def test_send_button_probe_requires_enabled_button_and_never_clicks(self):
        disabled = ProbePage(
            {'button[aria-label*="Send"]': ProbeLocator(visible=True, enabled=False)}
        )
        enabled = ProbePage(
            {'button[aria-label*="发送"]': ProbeLocator(visible=True, enabled=True)}
        )

        self.assertFalse(
            GeminiBot(disabled, {"gemini": {}}, FakeLogger()).health_check_send_button()
        )
        self.assertTrue(
            GeminiBot(enabled, {"gemini": {}}, FakeLogger()).health_check_send_button()
        )
        self.assertEqual(disabled.total_clicks, 0)
        self.assertEqual(enabled.total_clicks, 0)

    def test_health_check_probes_do_not_call_send_message(self):
        page = ProbePage({})
        bot = GeminiBot(page, {"gemini": {}}, FakeLogger())

        def forbidden_send(_text):
            raise AssertionError("health checks must never send a prompt")

        bot._send_message = forbidden_send

        self.assertFalse(bot.health_check_editor())
        self.assertFalse(bot.health_check_upload_control())
        self.assertFalse(bot.health_check_send_button())


if __name__ == "__main__":
    unittest.main()
