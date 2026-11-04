import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from playwright.sync_api import sync_playwright

import gemini_bot
import webui
from gemini_bot import GeminiBot
from gemini_browser_session import GeminiPageState, LoginStatus, mode_reading_paths


class QuietLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    warning = info


# Mirrors the live layout: the model button renders late, and the thinking-level
# submenu is a menu item with aria-haspopup that controls a separate panel. The
# level labels are deliberately unlike any Gemini wording: only positions count.
LIVE_LIKE_MENU = """
<div id="composer"></div>
<script>
  window.levelClicks = 0;
  setTimeout(() => {
    document.querySelector('#composer').innerHTML =
      '<button id="mode" aria-label="Open mode picker, current mode: Turbo" aria-haspopup="menu">Turbo</button>';
    document.querySelector('#mode').onclick = () => {
      document.querySelector('#menu').style.display = 'block';
      document.querySelector('#mode').setAttribute('aria-expanded', 'true');
    };
  }, __DELAY__);
  function choose(item) {
    window.levelClicks += 1;
    document.querySelectorAll('#levels gem-menu-item').forEach(node => node.classList.remove('selected'));
    item.classList.add('selected');
    // Gemini closes both menus after a level is chosen.
    document.querySelector('#menu').style.display = 'none';
    document.querySelector('#levels').style.display = 'none';
    document.querySelector('#mode').setAttribute('aria-expanded', 'false');
  }
</script>
<div id="menu" role="menu" style="display:none">
  <gem-menu-item role="menuitem" class="selected"><span class="label">Turbo</span></gem-menu-item>
  <gem-menu-item role="menuitem"><span class="label">Deep</span></gem-menu-item>
  <gem-menu-item id="trigger" role="menuitem" aria-haspopup="true" aria-controls="levels"
      onclick="document.querySelector('#levels').style.display='block'"><span class="label">Effort</span></gem-menu-item>
</div>
<div id="levels" role="menu" style="display:none">
  <gem-menu-item role="menuitem" class="selected" onclick="choose(this)"><span class="label">Calm</span></gem-menu-item>
  <gem-menu-item role="menuitem" onclick="choose(this)"><span class="label">Steady</span></gem-menu-item>
  <gem-menu-item role="menuitem" onclick="choose(this)"><span class="label">Intense</span></gem-menu-item>
</div>
"""


class ManualThinkingLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.page = self.browser.new_page()
        self.logger = QuietLogger()

    def tearDown(self):
        self.page.close()

    def bot(self, **gemini_cfg):
        return GeminiBot(self.page, {"gemini": gemini_cfg}, self.logger)

    def load(self, delay_ms=0):
        self.page.set_content(LIVE_LIKE_MENU.replace("__DELAY__", str(delay_ms)))

    def selected_level(self):
        return self.page.evaluate(
            "() => [...document.querySelectorAll('#levels gem-menu-item')].findIndex(n => n.classList.contains('selected')) + 1"
        )

    def test_reads_selected_level_by_position_without_text(self):
        self.load()
        self.page.evaluate("() => choose(document.querySelectorAll('#levels gem-menu-item')[1])")
        bot = self.bot()
        self.assertEqual(bot.read_thinking_level_choice(), {"index": 2, "count": 3})
        self.assertEqual(bot.current_mode_label(), "Turbo")

    def test_replays_saved_position_after_late_mode_button(self):
        self.load(delay_ms=2500)
        bot = self.bot(mode_selection="manual", manual_thinking_level={"index": 3, "count": 3})
        self.assertTrue(bot._select_thinking_mode())
        self.assertEqual(self.selected_level(), 3)

    def test_saved_last_position_means_highest_when_option_count_changes(self):
        self.load()
        self.page.evaluate("() => document.querySelector('#levels').insertAdjacentHTML('beforeend', "
                           "'<gem-menu-item role=\"menuitem\" onclick=\"choose(this)\"><span class=\"label\">Max</span></gem-menu-item>')")
        bot = self.bot(mode_selection="manual", manual_thinking_level={"index": 3, "count": 3})
        self.assertTrue(bot._select_thinking_mode())
        self.assertEqual(self.selected_level(), 4)
        self.assertTrue(any("options changed (3 -> 4)" in message for message in self.logger.messages))

    def test_already_selected_position_is_not_clicked_again(self):
        self.load()
        bot = self.bot(mode_selection="manual", manual_thinking_level={"index": 1, "count": 3})
        self.assertTrue(bot._select_thinking_mode())
        self.assertEqual(self.page.evaluate("() => window.levelClicks"), 0)

    def test_page_without_level_menu_keeps_the_browser_choice(self):
        self.page.set_content('<button aria-label="Open mode picker" aria-haspopup="menu">Turbo</button>')
        bot = self.bot(mode_selection="manual", manual_thinking_level={"index": 0, "count": 0})
        self.assertTrue(bot._select_thinking_mode())

    def test_auto_mode_waits_for_late_mode_button_before_giving_up(self):
        bot = self.bot()
        calls = []

        def open_menu():
            calls.append(1)
            return len(calls) >= 3

        bot._open_mode_menu = open_menu
        self.page.set_content("<p>loading</p>")
        started = time.monotonic()
        self.assertTrue(bot._wait_for_mode_menu())
        self.assertEqual(len(calls), 3)
        self.assertLess(time.monotonic() - started, 5)


class TemporaryChatWaitTests(unittest.TestCase):
    class Page:
        def __init__(self):
            self.waits = 0

        def wait_for_timeout(self, _ms):
            self.waits += 1

    def make_bot(self, outcomes):
        bot = GeminiBot(self.Page(), {"gemini": {}}, QuietLogger())
        attempts = []

        def start():
            outcome = outcomes[min(len(attempts), len(outcomes) - 1)]
            attempts.append(outcome)
            if outcome == "click":
                bot._note_temporary_chat_click()
                return False
            return outcome == "active"

        bot._start_temporary_chat = start
        return bot, attempts

    def test_waits_for_control_that_renders_late(self):
        bot, attempts = self.make_bot(["missing", "missing", "active"])
        self.assertTrue(bot._wait_and_start_temporary_chat())
        self.assertEqual(len(attempts), 3)

    def test_slow_activation_is_observed_not_clicked_twice(self):
        bot, attempts = self.make_bot(["click", "click"])
        with patch.object(bot, "_wait_for_temporary_chat_active", return_value=True) as observe:
            self.assertTrue(bot._wait_and_start_temporary_chat())
        self.assertEqual(attempts, ["click"])
        observe.assert_called_once()

    def test_gives_up_after_wait_window(self):
        bot, attempts = self.make_bot(["missing"])
        self.assertFalse(bot._wait_and_start_temporary_chat())
        self.assertEqual(len(attempts), gemini_bot.TEMPORARY_CHAT_WAIT_SECONDS)


class ManualModeSettingsTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config_path = Path(self.tmp.name) / "config.yaml"
        self.config_path.write_text(yaml.safe_dump({"gemini": {"base_url": "https://gemini.google.com"}}), encoding="utf-8")

    def saved_gemini(self):
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))["gemini"]

    def test_read_and_save_stores_only_positions(self):
        paths = webui.login_runtime_paths(self.config_path)
        _request, result = mode_reading_paths(paths)
        ready = LoginStatus.create(GeminiPageState.READY, True, "https://gemini.google.com/app", "zh-CN", "ok")

        def fake_sleep(_seconds):
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(json.dumps({"ok": True, "index": 3, "count": 3, "model": "3.8 Flash"}), encoding="utf-8")

        with patch.object(webui, "login_helper_is_active", return_value=True), \
                patch.object(webui, "read_login_status", return_value=ready), \
                patch.object(webui.time, "sleep", fake_sleep):
            message = webui.read_and_save_gemini_mode(self.config_path)

        self.assertIn("第 3 档（共 3 档）", message)
        self.assertIn("3.8 Flash", message)
        saved = self.saved_gemini()
        self.assertEqual(saved["mode_selection"], "manual")
        self.assertEqual(saved["manual_thinking_level"], {"index": 3, "count": 3})
        # Nothing tied to Gemini's wording ends up in the settings.
        self.assertNotIn("Flash", self.config_path.read_text(encoding="utf-8"))

    def test_read_requires_open_gemini_login_browser(self):
        with patch.object(webui, "login_helper_is_active", return_value=False):
            self.assertIn("打开 Gemini 登录浏览器", webui.read_and_save_gemini_mode(self.config_path))

    def test_manual_mode_without_saved_level_blocks_the_run(self):
        webui.save_gemini_browser_settings(False, config_path=self.config_path, mode_selection="manual")
        with patch.object(webui, "login_helper_is_active", return_value=False), \
                patch.object(webui, "_browser_selection_problem", return_value=None):
            problem = webui.guard_gemini_browser_task("gemini_browser", config_path=self.config_path)
        self.assertIn("读取当前档位并保存", problem)

    def test_switching_back_to_auto_keeps_running_unblocked(self):
        webui.save_gemini_browser_settings(False, config_path=self.config_path, mode_selection="auto")
        self.assertEqual(self.saved_gemini()["mode_selection"], "auto")
        with patch.object(webui, "login_helper_is_active", return_value=False), \
                patch.object(webui, "_browser_selection_problem", return_value=None):
            self.assertIsNone(webui.guard_gemini_browser_task("gemini_browser", config_path=self.config_path))


class ManualModeHealthCheckTests(unittest.TestCase):
    def test_model_step_keeps_the_users_model(self):
        from gemini_health_check import PlaywrightGeminiHealthCheckHarness

        class Bot:
            cfg = {"mode_selection": "manual"}

            def current_mode_label(self):
                return "3.8 Flash"

            def _select_prompt_model(self):
                raise AssertionError("manual mode must not switch the model")

        harness = PlaywrightGeminiHealthCheckHarness.__new__(PlaywrightGeminiHealthCheckHarness)
        harness.bot = Bot()
        self.assertTrue(harness.select_model())
        self.assertEqual(harness.selected_model, "3.8 Flash")


if __name__ == "__main__":
    unittest.main()
