from __future__ import annotations

import unittest

from novel_agent.prompts import build_chapter_review_messages
from novel_agent.style import analyze_prose
from novel_agent.engine import _merge_style_report


class ProseStyleLintTest(unittest.TestCase):
    def test_detects_repeated_template_signals_without_claiming_ai_probability(self):
        text = "".join(
            "他猛地抬头，仿佛一道看不见的闪电劈过房间。"
            "紧接着，他咬紧牙关，心脏狂跳。"
            for _ in range(30)
        )
        report = analyze_prose(text)
        self.assertLess(report["score"], 100)
        codes = {item["code"] for item in report["issues"]}
        self.assertIn("style_simile", codes)
        self.assertIn("style_intensifier", codes)
        self.assertNotIn("probability", report)

    def test_review_prompt_receives_report_only_when_enabled(self):
        report = {
            "score": 70,
            "issues": [
                {
                    "code": "style_simile",
                    "severity": "major",
                    "evidence": "仿佛一道看不见的闪电",
                    "problem": "比喻过密",
                    "fix": "改用具体动作",
                }
            ],
        }
        enabled = "\n".join(
            item["content"]
            for item in build_chapter_review_messages({}, {}, "正文", {}, report, True)
        )
        disabled = "\n".join(
            item["content"]
            for item in build_chapter_review_messages({}, {}, "正文", {}, report, False)
        )
        self.assertIn("style_simile", enabled)
        self.assertIn("仿佛一道看不见的闪电", enabled)
        self.assertNotIn("style_simile", disabled)

    def test_local_signal_never_forces_a_revision_verdict(self):
        review = {"decision": "pass", "issues": [], "revision_instructions": []}
        report = {
            "score": 50,
            "issues": [
                {
                    "code": "style_simile",
                    "severity": "major",
                    "evidence": "仿佛一道看不见的闪电",
                    "problem": "比喻过密",
                    "fix": "改用具体动作",
                }
            ],
        }
        merged = _merge_style_report(review, report)
        self.assertEqual(merged["decision"], "pass")
        self.assertIn("naturalness_report", merged)


if __name__ == "__main__":
    unittest.main()
