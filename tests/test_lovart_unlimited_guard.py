import tempfile
import unittest
from pathlib import Path

from lovart_api import AgentSkillError
from lovart_bot import LovartBot, resolve_lovart_tool_config, unlimited_model_catalog


def unlimited_state(*tools, unlimited=True, enabled=True):
    return {
        "unlimited": unlimited,
        "unlimited_enable": enabled,
        "unlimited_list": [
            {"status": 1, "alias_list": [f"provider/{tool}", tool]}
            for tool in tools
        ],
    }


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def warning(self, message):
        self.messages.append(message)


class _ModeSkill:
    def __init__(self, states):
        self.states = list(states)
        self.set_calls = []
        self.confirm_calls = 0

    def set_mode(self, unlimited):
        self.set_calls.append(unlimited)
        return {}

    def query_mode(self):
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]

    def confirm(self, _thread_id):
        self.confirm_calls += 1
        raise AssertionError("unlimited mode must never confirm a paid operation")


class LovartUnlimitedGuardTests(unittest.TestCase):
    def make_bot(self, skill, fast=False):
        bot = LovartBot.__new__(LovartBot)
        bot.cfg = {"max_confirmation_rounds": 2, "max_auto_confirm_credits": 10}
        bot.logger = _Logger()
        bot.skill = skill
        bot._fast_mode = fast
        bot._unlimited_tool_names = ()
        bot._configured_unlimited_models = ()
        bot.tool_config = resolve_lovart_tool_config({
            "image_model": "auto",
            "model_selection": "prefer",
            "reasoning_mode": "fast",
        })
        return bot

    def test_set_unlimited_mode_is_verified_and_caches_server_tools(self):
        state = unlimited_state(
            "generate_image_nano_banana_pro",
            "generate_image_seedream_v4_5",
        )
        skill = _ModeSkill([state])
        bot = self.make_bot(skill)

        bot.set_fast_mode(False)

        self.assertEqual(skill.set_calls, [True])
        self.assertEqual(
            bot._unlimited_tool_names,
            ("generate_image_nano_banana_pro", "generate_image_seedream_v4_5"),
        )
        self.assertTrue(any("verified by server: unlimited" in msg for msg in bot.logger.messages))

    def test_auto_is_hard_limited_to_current_unlimited_tools(self):
        allowed = (
            "generate_image_gpt_image_2",
            "generate_image_nano_banana_pro",
        )
        skill = _ModeSkill([unlimited_state(*allowed)])
        bot = self.make_bot(skill)

        options = bot._prepare_send_options(bot.tool_config)

        self.assertEqual(skill.set_calls, [True])
        self.assertEqual(options["include_tools"], list(allowed))
        self.assertIsNone(options["mode"])

    def test_selected_paid_model_is_blocked_before_submission(self):
        skill = _ModeSkill([unlimited_state("generate_image_nano_banana_pro")])
        bot = self.make_bot(skill)
        selected = resolve_lovart_tool_config({
            "image_model": "gpt_image_2",
            "model_selection": "force",
            "reasoning_mode": "fast",
        })

        with self.assertRaisesRegex(AgentSkillError, "不在当前账号的无限生成列表"):
            bot._prepare_send_options(selected)

    def test_mode_mismatch_is_reapplied_then_stops_if_still_fast(self):
        fast_state = unlimited_state(unlimited=False)
        skill = _ModeSkill([fast_state, fast_state])
        bot = self.make_bot(skill)

        with self.assertRaisesRegex(AgentSkillError, "避免错误计费"):
            bot._verify_generation_mode(expected_unlimited=True)

        self.assertEqual(skill.set_calls, [True])

    def test_disabled_unlimited_entitlement_stops_before_generation(self):
        skill = _ModeSkill([unlimited_state(enabled=False)])
        bot = self.make_bot(skill)

        with self.assertRaisesRegex(AgentSkillError, "未启用无限生成权限"):
            bot._verify_generation_mode(expected_unlimited=True)

    def test_unlimited_mode_never_confirms_even_without_estimated_cost(self):
        skill = _ModeSkill([unlimited_state("generate_image_nano_banana_pro")])
        bot = self.make_bot(skill)

        class Advisor:
            def advise_lovart_confirmation(self, **_kwargs):
                raise AssertionError("advisor must not be called in unlimited mode")

        with tempfile.TemporaryDirectory() as tmp:
            result = bot._resolve_pending_confirmations(
                result={
                    "final_status": "pending_confirmation",
                    "pending_confirmation": {"message": "approval required"},
                },
                product_dir=Path(tmp),
                product_id="SKU-1",
                product_name_cn="Product",
                language="English",
                selling_points="",
                project_id="project-1",
                thread_id="thread-1",
                confirmation_advisor=Advisor(),
            )

        self.assertIn("credit-required", result["warning"])
        self.assertEqual(skill.confirm_calls, 0)

    def test_catalog_preserves_server_order_and_plan_restrictions(self):
        state = {
            "unlimited": True,
            "unlimited_enable": True,
            "unlimited_list": [
                {
                    "name": "Nano Banana Pro",
                    "status": 1,
                    "extraItem": "1K",
                    "alias_list": ["generate_image_nano_banana_pro"],
                },
                {
                    "name": "Paid Model",
                    "status": 0,
                    "alias_list": ["generate_image_gpt_image_2"],
                },
            ],
        }

        self.assertEqual(
            unlimited_model_catalog(state),
            [{
                "model": "nano_banana_pro",
                "tool_name": "generate_image_nano_banana_pro",
                "label": "Nano Banana Pro",
                "restriction": "1K",
            }],
        )

    def test_saved_annual_models_are_filtered_for_monthly_account(self):
        skill = _ModeSkill([unlimited_state("generate_image_nano_banana")])
        bot = self.make_bot(skill)
        bot._configured_unlimited_models = (
            "nano_banana_pro",
            "nano_banana",
            "gpt_image_2",
        )

        models = bot._unlimited_attempt_models(bot.tool_config)

        self.assertEqual(models, ["nano_banana"])
        self.assertTrue(any("will be skipped" in msg for msg in bot.logger.messages))

    def test_unlimited_models_are_forced_one_at_a_time_in_saved_order(self):
        skill = _ModeSkill([unlimited_state(
            "generate_image_nano_banana",
            "generate_image_nano_banana_2",
        )])
        bot = self.make_bot(skill)
        bot._configured_unlimited_models = ("nano_banana_2", "nano_banana")
        calls = []

        def execute(**kwargs):
            calls.append((bot.tool_config.copy(), kwargs.copy()))
            if len(calls) == 1:
                return {
                    "final_status": "pending_confirmation",
                    "generation_succeeded": False,
                }, "project-1", "thread-paid"
            return {
                "final_status": "done",
                "generation_succeeded": True,
            }, "project-1", "thread-free"

        result, project_id, thread_id = bot._execute_with_fallback(
            execute, project_id="project-1"
        )

        self.assertEqual([call[0]["image_model"] for call in calls], [
            "nano_banana_2", "nano_banana"
        ])
        self.assertTrue(all(call[0]["model_selection"] == "force" for call in calls))
        self.assertTrue(calls[1][1]["force_new_thread"])
        self.assertEqual(result["used_model"], "nano_banana_2 ➔ nano_banana")
        self.assertEqual((project_id, thread_id), ("project-1", "thread-free"))
        self.assertEqual(bot.tool_config["image_model"], "auto")

    def test_stale_selection_with_no_live_unlimited_model_stops(self):
        skill = _ModeSkill([unlimited_state("generate_image_nano_banana")])
        bot = self.make_bot(skill)
        bot._configured_unlimited_models = ("nano_banana_pro",)

        with self.assertRaisesRegex(AgentSkillError, "重新检测并保存模型"):
            bot._unlimited_attempt_models(bot.tool_config)


if __name__ == "__main__":
    unittest.main()
