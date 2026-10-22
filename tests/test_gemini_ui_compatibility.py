import json
import time
import unittest

from playwright.sync_api import sync_playwright

from gemini_bot import GeminiBot
from gemini_health_check import CheckState, PlaywrightGeminiHealthCheckHarness, run_checks


class QuietLogger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass


class GeminiUiCompatibilityTests(unittest.TestCase):
    """Exercise production selectors against menus shaped like the live Gemini UI."""

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
        self.bot = GeminiBot(self.page, {"gemini": {}}, QuietLogger())

    def tearDown(self):
        self.page.close()

    def health_harness(self, timeout=2):
        harness = PlaywrightGeminiHealthCheckHarness.from_test_bot(self.bot)
        harness.config = {"browser": {"page_ready_timeout": timeout}}
        self.addCleanup(harness.close)
        return harness

    def test_health_check_waits_for_delayed_editor(self):
        self.page.set_content('''<script>
            setTimeout(() => document.body.insertAdjacentHTML('beforeend',
                '<div contenteditable="true" role="textbox">Ready</div>'), 600);
        </script>''')
        self.assertTrue(self.health_harness().check_editor())

    def test_health_check_waits_for_upload_control_after_shell_loads(self):
        self.page.set_content('''
            <div contenteditable="true" role="textbox">Ready</div>
            <button>New chat</button><button>Library</button><button>Settings</button>
            <script>
                window.uploadClicks = 0;
                setTimeout(() => document.body.insertAdjacentHTML('beforeend',
                    '<button aria-label="Upload" onclick="uploadClicks++">+</button>'), 600);
            </script>
        ''')
        self.assertTrue(self.health_harness().check_upload_control())
        self.assertEqual(self.page.evaluate('window.uploadClicks'), 0)

    def test_health_check_waits_for_disabled_editor_and_upload_control(self):
        for html, method in (
            ('<textarea id="control" disabled></textarea>', 'check_editor'),
            ('<button id="control" aria-label="Upload" disabled>+</button>', 'check_upload_control'),
        ):
            with self.subTest(method=method):
                self.page.set_content(html + '''<script>
                    setTimeout(() => document.querySelector('#control').disabled = false, 600);
                </script>''')
                self.assertTrue(getattr(self.health_harness(), method)())
                self.assertFalse(self.page.locator('#control').is_disabled())

    def test_editor_probe_rejects_readonly_editor(self):
        self.page.set_content('<textarea readonly>Still loading</textarea>')
        self.assertFalse(self.bot.health_check_editor())

    def test_health_check_waits_for_send_button_to_become_enabled_without_sending(self):
        self.page.set_content('''
            <button id="send" aria-label="Send" disabled>Send</button>
            <script>
                window.sendClicks = 0;
                document.querySelector('#send').onclick = () => sendClicks++;
                setTimeout(() => document.querySelector('#send').disabled = false, 600);
            </script>
        ''')
        self.assertTrue(self.health_harness().check_send_button())
        self.assertEqual(self.page.evaluate('window.sendClicks'), 0)

    def test_health_check_waits_for_mode_button_to_mount(self):
        self.page.set_content('''<script>
            window.modeClicks = 0;
            setTimeout(() => document.body.insertAdjacentHTML('beforeend',
                '<button aria-label="Open mode picker" onclick="modeClicks++">Pro</button>'), 600);
        </script>''')
        self.assertTrue(self.health_harness().open_mode_menu())
        self.assertEqual(self.page.evaluate('window.modeClicks'), 1)

    def test_health_check_waits_for_disabled_mode_button_without_reporting_a_noop_click(self):
        self.page.set_content('''
            <button id="control" aria-label="Open mode picker" disabled>Pro</button>
            <script>
                window.modeClicks = 0;
                document.querySelector('#control').onclick = () => modeClicks++;
                setTimeout(() => document.querySelector('#control').disabled = false, 600);
            </script>
        ''')
        self.assertTrue(self.health_harness().open_mode_menu())
        self.assertEqual(self.page.evaluate('window.modeClicks'), 1)

    def test_health_check_rejects_permanently_disabled_mode_button_within_budget(self):
        self.page.set_content('<button aria-label="Open mode picker" disabled>Pro</button>')
        started = time.monotonic()
        self.assertFalse(self.health_harness(timeout=1).open_mode_menu())
        self.assertLess(time.monotonic() - started, 2.5)

    def test_health_check_waits_for_temporary_chat_button_to_mount(self):
        self.page.set_content('''<script>
            window.temporaryClicks = 0;
            setTimeout(() => document.body.insertAdjacentHTML('beforeend', `
                <button aria-label="Temporary chat" onclick="
                    temporaryClicks++;
                    this.innerHTML = '&lt;mat-icon data-mat-icon-name=close&gt;x&lt;/mat-icon&gt;';
                ">Temporary chat</button>`), 600);
        </script>''')
        self.assertTrue(self.health_harness().enter_temporary_chat())
        self.assertEqual(self.page.evaluate('window.temporaryClicks'), 1)

    def test_health_check_does_not_toggle_temporary_chat_twice_during_slow_activation(self):
        self.page.set_content('''
            <button id="temporary" aria-label="Temporary chat">Temporary chat</button>
            <script>
                window.temporaryClicks = 0;
                let pending = null;
                document.querySelector('#temporary').onclick = function () {
                    temporaryClicks++;
                    if (pending) { clearTimeout(pending); pending = null; return; }
                    pending = setTimeout(() => {
                        this.innerHTML = '<mat-icon data-mat-icon-name="close">x</mat-icon>';
                        pending = null;
                    }, 8200);
                };
            </script>
        ''')
        self.assertTrue(self.health_harness(timeout=12).enter_temporary_chat())
        self.assertEqual(self.page.evaluate('window.temporaryClicks'), 1)

    def test_default_temporary_chat_keeps_legacy_activation_grace_period(self):
        self.page.set_content('''
            <button aria-label="Temporary chat" onclick="setTimeout(() => {
                this.innerHTML = '&lt;mat-icon data-mat-icon-name=close&gt;x&lt;/mat-icon&gt;';
            }, 6500)">Temporary chat</button>
        ''')
        self.assertTrue(self.bot._start_temporary_chat())

    def test_health_check_reports_missing_control_only_after_bounded_wait(self):
        self.page.set_content('<div contenteditable="true">Ready</div>')
        harness = self.health_harness(timeout=1)
        started = time.monotonic()
        self.assertFalse(harness.check_upload_control())
        self.assertGreaterEqual(time.monotonic() - started, 0.9)
        self.assertLess(time.monotonic() - started, 4)

    def test_health_check_selects_nested_high_thinking_and_dismisses_menu(self):
        self.load_menu()
        self.assertTrue(self.health_harness().select_extended_thinking())
        self.assertEqual(self.page.locator('body').get_attribute('data-level'), 'high')
        self.page.locator('#editor').click(timeout=1500)
        self.assertEqual(self.page.locator(':focus').get_attribute('id'), 'editor')

    def load_menu(self, *, flash="3.8 Flash", language="en", current="Pro",
                  level="low", accept_level=True, accept_model=True,
                  close_level=False, high_label=None):
        labels = {
            "en": ("Thinking level", "Low", "Medium", "High"),
            "zh": ("思考等级", "低", "中", "高"),
            "es": ("Nivel de pensamiento", "Bajo", "Medio", "Alto"),
        }[language]
        if high_label:
            labels = (*labels[:3], high_label)
        options = "".join(
            f'<gem-menu-item role="menuitem" data-level="{key}" '
            f'class="{"selected" if key == level else "active"}">'
            f'<span class="label">{label}</span>'
            '<button title="Set default thinking level">star</button>'
            '<span class="sublabel">Response detail</span></gem-menu-item>'
            for key, label in zip(("low", "medium", "high"), labels[1:])
        )
        self.page.set_content(f"""
            <style>
                gem-menu-item {{ display: block; padding: 4px; }}
                [hidden] {{ display: none !important; }}
            </style>
            <main>
              <div contenteditable="true" role="textbox" id="editor"></div>
              <button id="mode" aria-label="Open mode picker, current mode: {current}"
                      aria-haspopup="menu" aria-expanded="true">{current}</button>
              <div role="menu" id="models">
                <gem-menu-item role="menuitem" data-mode-id="lite" aria-label="3.5 Flash-Lite">
                  <span class="label">3.5 Flash-Lite</span><span>Fastest answers</span>
                </gem-menu-item>
                <gem-menu-item role="menuitem" data-mode-id="flash" aria-label="{flash}">
                  <span class="label">{flash}</span>
                  <button title="Set default model">star</button><span>All-around help</span>
                </gem-menu-item>
                <gem-menu-item role="menuitem" data-mode-id="pro">
                  <span class="label">3.1 Pro</span><span>Advanced reasoning</span>
                </gem-menu-item>
                <gem-menu-item role="menuitem" id="thinking" aria-haspopup="menu">
                  <span class="label">{labels[0]}</span>
                </gem-menu-item>
                <div role="menu" id="levels" hidden>{options}</div>
              </div>
              <button title="Extended warranty">Unrelated conversation action</button>
            </main>
            <script>(() => {{
              const mode = document.querySelector('#mode');
              const models = document.querySelector('#models');
              const levels = document.querySelector('#levels');
              document.body.dataset.model = {json.dumps(current)};
              document.body.dataset.level = {json.dumps(level)};
              document.body.dataset.defaults = '0';
              mode.onclick = () => {{
                models.hidden = !models.hidden;
                mode.setAttribute('aria-expanded', String(!models.hidden));
              }};
              document.addEventListener('keydown', event => {{
                if (event.key !== 'Escape') return;
                if (!levels.hidden) levels.hidden = true;
                else {{
                  models.hidden = true;
                  mode.setAttribute('aria-expanded', 'false');
                }}
              }});
              document.querySelectorAll('[data-mode-id]').forEach(item => {{
                item.onclick = () => {{
                  if (!{json.dumps(accept_model)}) return;
                  document.body.dataset.model = item.dataset.modeId;
                  mode.textContent = item.dataset.modeId === 'flash' ? 'Flash' : 'Flash-Lite';
                  mode.setAttribute('aria-label', 'Open mode picker, current mode: ' + mode.textContent);
                  models.hidden = true;
                  mode.setAttribute('aria-expanded', 'false');
                }};
              }});
              document.querySelector('#thinking').onclick = () => levels.hidden = false;
              document.querySelectorAll('gem-menu-item[data-level]').forEach(item => {{
                item.onclick = () => {{
                  if (!{json.dumps(accept_level)}) return;
                  document.body.dataset.level = item.dataset.level;
                  document.querySelectorAll('gem-menu-item[data-level]').forEach(option =>
                    option.classList.toggle('selected', option === item));
                  if ({json.dumps(close_level)}) {{
                    models.hidden = true;
                    levels.hidden = true;
                    mode.setAttribute('aria-expanded', 'false');
                  }}
                }};
              }});
              document.querySelectorAll('gem-menu-item button').forEach(button => {{
                button.onclick = event => {{
                  event.stopPropagation();
                  document.body.dataset.defaults = String(Number(document.body.dataset.defaults) + 1);
                }};
              }});
            }})();</script>
        """)

    def load_auto_menu(self, *, current="Gemini Auto", level="low", flash=False,
                       accept_auto=True, accept_flash=True, plain_label=False):
        self.load_menu(current=current, level=level, accept_model=accept_flash)
        self.page.evaluate('''options => {
            const mode = document.querySelector('#mode');
            const models = document.querySelector('#models');
            if (!options.flash) {
                models.querySelectorAll('[data-mode-id]').forEach(item => item.remove());
            }
            const auto = document.createElement('gem-menu-item');
            auto.setAttribute('role', 'menuitem');
            auto.dataset.modeId = 'auto';
            auto.innerHTML = options.plainLabel
                ? 'Auto<br><span class="sublabel">Gemini 3 models</span>'
                : '<span class="label">Auto</span><span class="sublabel">Gemini 3 models</span>';
            auto.classList.toggle('selected', options.current === 'Gemini Auto');
            auto.onclick = () => {
                if (!options.acceptAuto) return;
                document.body.dataset.model = 'auto';
                mode.textContent = 'Gemini Auto';
                if (mode.hasAttribute('aria-label')) {
                    mode.setAttribute('aria-label', 'Open mode picker, current mode: Auto');
                }
                models.hidden = true;
                mode.setAttribute('aria-expanded', 'false');
            };
            models.prepend(auto);
        }''', {'current': current, 'flash': flash, 'acceptAuto': accept_auto,
               'plainLabel': plain_label})

    def load_native_auto_menu(self, *, level='low'):
        self.page.set_content('''
            <style>
                [hidden] { display: none !important; }
                .mat-mdc-menu-item-text > span { display: block; }
                .google-symbols { display: block; }
            </style>
            <div id="editor" contenteditable="true" role="textbox"></div>
            <button id="mode" aria-label="Open mode picker" aria-expanded="true">
                <span>Gemini</span><span>Auto</span><span class="google-symbols">keyboard_arrow_up</span>
            </button>
            <div id="models" role="menu">
                <button id="auto" role="menuitemradio" aria-checked="true"
                        aria-label="Auto Gemini 3 models" class="mat-mdc-menu-item">
                    <span class="google-symbols">check</span>
                    <span class="mat-mdc-menu-item-text"><span>Auto</span><span>Gemini 3 models</span></span>
                </button>
                <button id="thinking" role="menuitem" aria-haspopup="menu">
                    Thinking level<span class="google-symbols">chevron_right</span>
                </button>
                <div id="levels" role="menu" hidden>
                    <button role="menuitemradio" id="low" aria-checked="true">Low</button>
                    <button role="menuitemradio" id="high" aria-label="High Extra thorough"
                            aria-checked="false" class="mat-mdc-menu-item">
                        <span class="mat-mdc-menu-item-text"><span>High</span><span>Extra thorough</span></span>
                    </button>
                </div>
            </div>
            <script>
                window.autoClicks = 0;
                document.querySelector('#auto').onclick = () => autoClicks++;
                const mode = document.querySelector('#mode');
                const models = document.querySelector('#models');
                const levels = document.querySelector('#levels');
                mode.onclick = () => {
                    models.hidden = !models.hidden;
                    mode.setAttribute('aria-expanded', String(!models.hidden));
                };
                document.querySelector('#thinking').onclick = () => levels.hidden = false;
                document.querySelector('#high').onclick = () => {
                    document.body.dataset.level = 'high';
                    document.querySelector('#high').setAttribute('aria-checked', 'true');
                    document.querySelector('#low').setAttribute('aria-checked', 'false');
                    levels.hidden = true;
                    models.hidden = true;
                    mode.setAttribute('aria-expanded', 'false');
                };
                document.addEventListener('keydown', event => {
                    if (event.key !== 'Escape') return;
                    if (!levels.hidden) levels.hidden = true;
                    else { models.hidden = true; mode.setAttribute('aria-expanded', 'false'); }
                });
            </script>
        ''')
        self.page.evaluate('''level => {
            document.body.dataset.level = level;
            if (level === 'high') {
                document.querySelector('#high').setAttribute('aria-checked', 'true');
                document.querySelector('#low').setAttribute('aria-checked', 'false');
                document.querySelector('#high').insertAdjacentHTML('afterbegin',
                    '<span class="google-symbols">check</span>');
            }
        }''', level)

    def test_native_auto_model_with_check_icon_and_subtitle_passes_health_check(self):
        self.load_native_auto_menu()
        harness = self.health_harness()
        self.assertTrue(harness.select_model())
        self.assertEqual(harness.selected_model, 'Auto')
        self.assertEqual(self.page.evaluate('autoClicks'), 0)

    def test_native_auto_menu_reaches_high_thinking(self):
        self.load_native_auto_menu()
        self.assertTrue(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-level'), 'high')
        self.assertEqual(self.page.evaluate('autoClicks'), 0)

    def test_native_auto_option_with_description_can_switch_from_pro(self):
        self.load_native_auto_menu()
        self.page.locator('#mode').evaluate('''node => {
            node.textContent = 'Pro';
            document.querySelector('#auto').setAttribute('aria-checked', 'false');
            document.querySelector('#auto .google-symbols').remove();
            document.querySelector('#auto').onclick = () => {
                autoClicks++;
                node.innerHTML = '<span>Gemini</span><span>Auto</span>'
                    + '<span class="google-symbols">keyboard_arrow_down</span>';
            };
        }''')
        self.assertTrue(self.health_harness().select_model())
        self.assertEqual(self.page.evaluate('autoClicks'), 1)

    def test_native_checked_high_label_ignores_check_icon_and_accessible_description(self):
        self.load_native_auto_menu(level='high')
        self.page.locator('#thinking').click()
        self.assertTrue(self.bot._current_thinking_level_is_extended())

    def test_native_high_selection_accepts_font_check_but_not_other_icons(self):
        self.load_native_auto_menu(level='high')
        self.page.locator('#thinking').click()
        self.page.locator('#high').evaluate("node => node.removeAttribute('aria-checked')")
        self.assertTrue(self.bot._current_thinking_level_is_extended())
        self.page.locator('#high .google-symbols').evaluate("node => node.textContent = 'star'")
        self.assertFalse(self.bot._current_thinking_level_is_extended())

    def test_hidden_or_nested_default_check_does_not_confirm_high(self):
        for icon in (
            '<mat-icon data-mat-icon-name="check" style="display:none">check</mat-icon>',
            '<span class="google-symbols" style="display:none">check</span>',
            '<button title="Set default"><mat-icon data-mat-icon-name="check">check</mat-icon></button>',
        ):
            with self.subTest(icon=icon):
                self.load_native_auto_menu()
                self.page.locator('#thinking').click()
                self.page.locator('#high').evaluate(
                    '''(node, icon) => {
                        const option = document.createElement('div');
                        for (const attribute of node.attributes) {
                            option.setAttribute(attribute.name, attribute.value);
                        }
                        option.innerHTML = icon + node.innerHTML;
                        node.replaceWith(option);
                    }''', icon)
                self.assertFalse(self.bot._current_thinking_level_is_extended())

    def test_auto_only_interface_selects_high_thinking_and_releases_editor(self):
        self.load_auto_menu()
        self.assertTrue(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-level'), 'high')
        self.page.locator('#editor').click(timeout=1500)
        self.assertEqual(self.page.locator(':focus').get_attribute('id'), 'editor')

    def test_auto_only_interface_accepts_already_selected_high_thinking(self):
        self.load_auto_menu(level='high')
        self.assertTrue(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-level'), 'high')

    def test_auto_only_toolbar_opens_without_aria_labels(self):
        self.load_auto_menu()
        self.page.locator('#mode').click()
        self.page.locator('#mode').evaluate('''node => {
            node.removeAttribute('aria-label');
            node.removeAttribute('aria-haspopup');
            node.insertAdjacentHTML('beforeend', '<mat-icon>keyboard_arrow_down</mat-icon>');
        }''')
        self.page.locator('main').evaluate('''node => node.insertAdjacentHTML('afterbegin',
            '<button>Auto upload</button>')''')
        self.assertTrue(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-level'), 'high')

    def test_auto_option_with_model_family_description_remains_selectable(self):
        self.load_auto_menu(current='Pro', plain_label=True)
        self.assertTrue(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-model'), 'auto')

    def test_auto_availability_keeps_non_lite_flash_preferred(self):
        self.load_auto_menu(flash=True)
        self.assertTrue(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-model'), 'flash')

    def test_auto_does_not_hide_failed_available_flash_selection(self):
        self.load_auto_menu(flash=True, accept_flash=False)
        self.assertFalse(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-level'), 'low')

    def test_auto_does_not_hide_flash_switch_failure_after_menu_rerenders(self):
        for change in ('remove', 'disable'):
            with self.subTest(change=change):
                self.load_auto_menu(flash=True)
                self.page.locator('[data-mode-id="flash"]').evaluate('''(node, change) => {
                    node.onclick = () => change === 'remove' ? node.remove()
                        : node.setAttribute('aria-disabled', 'true');
                }''', change)
                self.assertFalse(self.bot._select_thinking_mode())
                self.assertEqual(self.page.locator('body').get_attribute('data-level'), 'low')

    def test_health_check_retry_does_not_downgrade_a_failed_flash_switch_to_auto(self):
        self.load_auto_menu(flash=True)
        self.page.locator('[data-mode-id="flash"]').evaluate('node => node.onclick = () => node.remove()')
        harness = self.health_harness(timeout=3)
        self.assertFalse(harness.select_model())
        self.assertIsNone(harness.selected_model)

    def test_repeated_thinking_selection_does_not_hide_a_failed_flash_switch(self):
        self.load_auto_menu(flash=True)
        self.page.locator('[data-mode-id="flash"]').evaluate('node => node.onclick = () => node.remove()')
        self.assertFalse(self.bot._select_thinking_mode())
        self.assertFalse(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-level'), 'low')

    def test_auto_fallback_never_selects_flash_lite(self):
        self.load_auto_menu(current='Pro')
        self.page.locator('#models').evaluate('''node => {
            const lite = document.createElement('gem-menu-item');
            lite.setAttribute('role', 'menuitem');
            lite.innerHTML = '<span class="label">3.5 Flash-Lite</span>';
            lite.onclick = () => document.body.dataset.model = 'lite';
            node.prepend(lite);
        }''')
        self.assertTrue(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-model'), 'auto')

    def test_auto_click_without_model_switch_is_rejected(self):
        self.load_auto_menu(current='Pro', accept_auto=False)
        self.assertFalse(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-model'), 'Pro')

    def test_unrelated_auto_button_does_not_confirm_a_failed_model_switch(self):
        self.load_auto_menu(current='Pro', accept_auto=False)
        self.page.locator('main').evaluate('''node => node.insertAdjacentHTML('afterbegin',
            '<button>Auto</button>')''')
        self.assertFalse(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-model'), 'Pro')

    def test_unrelated_auto_button_does_not_override_unlabelled_pro_toolbar(self):
        self.load_auto_menu(current='Pro', accept_auto=False)
        self.page.locator('#mode').evaluate('''node => {
            node.removeAttribute('aria-label'); node.removeAttribute('aria-haspopup');
        }''')
        self.page.locator('main').evaluate('''node => node.insertAdjacentHTML('afterbegin',
            '<button>Auto</button>')''')
        self.assertFalse(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-model'), 'Pro')

    def test_auto_switch_from_pro_requires_an_available_model_option(self):
        for disabled in (False, True):
            with self.subTest(disabled=disabled):
                self.load_auto_menu(current='Pro')
                option = self.page.locator('[data-mode-id="auto"]')
                option.evaluate("node => node.setAttribute('aria-disabled', 'true')"
                                if disabled else "node => node.remove()")
                self.assertFalse(self.bot._select_thinking_mode())
                self.assertEqual(self.page.locator('body').get_attribute('data-level'), 'low')

    def test_current_auto_does_not_require_reselecting_a_model_menu_item(self):
        self.load_auto_menu()
        self.page.locator('[data-mode-id="auto"]').evaluate('node => node.remove()')
        self.assertTrue(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-level'), 'high')

    def test_auto_without_high_thinking_is_rejected(self):
        self.load_auto_menu()
        self.page.locator('[data-level="high"]').evaluate('node => node.remove()')
        self.assertFalse(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-level'), 'low')

    def test_health_check_reports_selected_auto_model(self):
        self.load_auto_menu()
        self.page.locator('main').evaluate('''node => node.insertAdjacentHTML('afterbegin',
            '<button aria-label="Temporary chat"><mat-icon data-mat-icon-name="close">x</mat-icon></button>')''')
        harness = self.health_harness()
        checks = run_checks(harness)
        try:
            results = [next(checks) for _ in range(9)]
        finally:
            checks.close()
        self.assertTrue(all(result.state is CheckState.PASS for result in results))
        model = next(result for result in results if result.name == '模型选择')
        self.assertEqual(model.message, '已选择 Auto')
        self.assertEqual(self.page.locator('body').get_attribute('data-level'), 'high')

    def test_flash_selection_never_clicks_flash_lite_or_default_star(self):
        self.load_menu(flash="3.5 Flash")
        self.assertTrue(self.bot._click_flash_model())
        self.assertEqual(self.page.locator("body").get_attribute("data-model"), "flash")
        self.assertEqual(self.page.locator("body").get_attribute("data-defaults"), "0")

    def test_flash_selection_accepts_versionless_and_new_major_labels(self):
        for label in ("Flash", "Gemini 4 Flash"):
            with self.subTest(label=label):
                self.load_menu(flash=label)
                self.assertTrue(self.bot._click_flash_model())
                self.assertEqual(self.page.locator("body").get_attribute("data-model"), "flash")

    def test_available_flash_option_does_not_mean_flash_is_current(self):
        self.load_menu(current="Pro")
        self.assertFalse(self.bot._current_model_is_flash())

    def test_flash_click_without_a_model_switch_does_not_report_success(self):
        self.load_menu(accept_model=False)
        self.assertFalse(self.bot._click_flash_model())

    def test_button_menu_items_remain_supported(self):
        self.page.set_content("""
            <button id="mode" aria-label="Open mode picker" aria-haspopup="menu">Pro</button>
            <div role="menu"><button role="menuitem" id="flash">3 Flash</button></div>
            <script>
                document.querySelector('#flash').onclick = () =>
                    document.querySelector('#mode').textContent = 'Flash';
            </script>
        """)
        self.assertTrue(self.bot._click_flash_model())
        self.assertEqual(self.page.locator("#mode").inner_text(), "Flash")

    def test_legacy_toolbar_without_aria_labels_can_confirm_flash(self):
        self.load_menu()
        self.page.locator('#mode').evaluate("node => { node.removeAttribute('aria-label'); node.removeAttribute('aria-haspopup'); }")
        # The model-switch handler updates aria-label, so remove it after the switch too.
        self.page.locator('[data-mode-id="flash"] .label').click()
        self.page.locator('#mode').evaluate("node => node.removeAttribute('aria-label')")
        self.assertTrue(self.bot._current_model_is_flash())

    def test_missing_or_disabled_flash_does_not_substitute_flash_lite(self):
        for disabled in (False, True):
            with self.subTest(disabled=disabled):
                self.load_menu()
                option = self.page.locator('[data-mode-id="flash"]')
                if disabled:
                    option.evaluate("node => node.setAttribute('aria-disabled', 'true')")
                else:
                    option.evaluate("node => node.remove()")
                self.assertFalse(self.bot._click_flash_model())
                self.assertEqual(self.page.locator("body").get_attribute("data-model"), "Pro")

    def test_unrelated_extended_text_does_not_mean_high_thinking_is_selected(self):
        self.load_menu()
        self.assertFalse(self.bot._current_thinking_level_is_extended())

    def test_selected_high_quality_item_in_another_menu_is_not_thinking(self):
        self.load_menu()
        self.page.locator('main').evaluate('''node => node.insertAdjacentHTML('afterbegin',
            '<div role="menu"><gem-menu-item role="menuitem" class="selected">'
            + '<span class="label">High quality</span></gem-menu-item></div>')''')
        self.assertFalse(self.bot._current_thinking_level_is_extended())
        self.assertTrue(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator('body').get_attribute('data-level'), 'high')

    def test_full_selection_uses_high_thinking_in_new_multilingual_submenus(self):
        for language in ("en", "zh", "es"):
            with self.subTest(language=language):
                self.load_menu(language=language)
                self.assertTrue(self.bot._select_thinking_mode())
                self.assertEqual(self.page.locator("body").get_attribute("data-model"), "flash")
                self.assertEqual(self.page.locator("body").get_attribute("data-level"), "high")
                self.assertEqual(self.page.locator("body").get_attribute("data-defaults"), "0")

    def test_high_option_is_selected_only_when_the_menu_confirms_it(self):
        self.load_menu(accept_level=False)
        self.page.locator("#levels").evaluate("node => node.hidden = false")
        self.assertFalse(self.bot._click_extended_thinking_level())

    def test_selection_is_verified_when_gemini_closes_both_menus(self):
        self.load_menu(close_level=True)
        self.assertTrue(self.bot._select_thinking_mode())
        self.assertEqual(self.page.locator("body").get_attribute("data-level"), "high")

    def test_verified_mode_menus_are_dismissed_before_editor_interaction(self):
        self.load_menu(close_level=True)
        self.page.add_style_tag(content='''
            #editor { height: 50px; width: 200px; }
            #models:not([hidden])::before {
                content: ''; position: fixed; inset: 0; z-index: 10;
            }
        ''')
        self.assertTrue(self.bot._select_thinking_mode())
        self.assertFalse(self.page.locator('#models').is_visible())
        self.page.locator('#editor').click(timeout=1000)
        self.assertEqual(self.page.locator(':focus').get_attribute('id'), 'editor')

    def test_legacy_extended_level_labels_remain_supported(self):
        for label in ("Extended", "扩展", "Ampliado"):
            with self.subTest(label=label):
                self.load_menu(high_label=label)
                self.assertTrue(self.bot._select_thinking_mode())
                self.assertEqual(self.page.locator("body").get_attribute("data-level"), "high")

    def test_focused_legacy_top_level_option_does_not_report_selection(self):
        self.page.set_content('''
            <div role="menu"><gem-menu-item role="menuitem" class="active">
              <span class="label">Extended thinking</span>
            </gem-menu-item></div>
        ''')
        self.assertFalse(self.bot._extended_thinking_option_is_checked())

    def test_legacy_top_level_click_is_verified(self):
        for accept in (False, True):
            with self.subTest(accept=accept):
                self.load_menu()
                self.page.locator('#models').evaluate('''(node, accept) => {
                    node.querySelector('#thinking').remove();
                    node.querySelector('#levels').remove();
                    const item = document.createElement('gem-menu-item');
                    item.setAttribute('role', 'menuitem');
                    item.innerHTML = '<span class="label">Extended thinking</span>';
                    item.onclick = () => {
                        if (accept) {
                            item.classList.add('selected');
                            document.body.dataset.level = 'high';
                        }
                    };
                    node.append(item);
                }''', accept)
                self.assertEqual(self.bot._select_thinking_mode(), accept)
                self.assertEqual(self.page.locator('body').get_attribute('data-level'),
                                 'high' if accept else 'low')

    def test_focused_high_option_is_not_treated_as_selected(self):
        self.load_menu()
        self.page.locator("#levels").evaluate("node => node.hidden = false")
        self.assertFalse(self.bot._current_thinking_level_is_extended())
        self.page.locator('gem-menu-item[data-level="high"] .label').click()
        self.assertTrue(self.bot._current_thinking_level_is_extended())


if __name__ == "__main__":
    unittest.main()
