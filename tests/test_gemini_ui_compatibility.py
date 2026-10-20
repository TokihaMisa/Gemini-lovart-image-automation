import json
import unittest

from playwright.sync_api import sync_playwright

from gemini_bot import GeminiBot


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
