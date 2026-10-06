"""Lightweight local prose linting for template-like AI writing.

This module intentionally does not decide whether a passage is good
literature.  It reports conservative, explainable signals that a language
model can use during revision: repeated stock transitions, stacked similes,
direct emotion labels, and exposition-heavy phrasing.  The original prose is
never sent anywhere by this module.
"""

from __future__ import annotations

import re
from math import ceil
from typing import Any


_PATTERN_GROUPS: dict[str, tuple[str, ...]] = {
    "simile": ("如同", "仿佛", "像是", "宛如", "犹如"),
    "intensifier": (
        "猛地",
        "瞬间",
        "骤然",
        "陡然",
        "紧接着",
        "毫无预兆",
        "刹那间",
        "顷刻间",
    ),
    "emotion_cliche": (
        "心脏狂跳",
        "心头一紧",
        "咬紧牙关",
        "呼吸急促",
        "倒吸一口凉气",
        "脸色大变",
        "浑身一震",
        "眼神冰冷",
        "不禁",
    ),
    "explanation": (
        "这意味着",
        "他意识到",
        "她意识到",
        "他知道",
        "她知道",
        "显然",
        "毫无疑问",
    ),
}

_GROUP_RULES: dict[str, tuple[str, float, str, str]] = {
    # group, minimum absolute hits, sentence ratio, problem, fix
    "simile": (
        "比喻",
        5,
        0.10,
        "比喻和类比过于密集，画面感开始变成统一模板。",
        "删去重复比喻；能用具体动作、物件或声音表达时，优先使用事实细节。",
    ),
    "intensifier": (
        "强刺激转折",
        6,
        0.12,
        "强刺激副词和突发转折重复，导致每个动作都像高潮。",
        "减少强度副词，改变句式和节奏，让部分动作自然发生。",
    ),
    "emotion_cliche": (
        "情绪套语",
        4,
        0.08,
        "情绪和身体反应反复使用固定表达。",
        "用停顿、误动作、回避、说错话或具体选择表现情绪，不直接命名情绪。",
    ),
    "explanation": (
        "解释腔",
        3,
        0.05,
        "叙述多次替读者解释人物已经能够从行动中理解的结论。",
        "删去总结句，保留证据，让读者从行动和对话自行推断。",
    ),
}

_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in _SENTENCE_RE.findall(text) if item.strip()]


def _occurrences(text: str, phrases: tuple[str, ...]) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    for phrase in phrases:
        start = 0
        while True:
            index = text.find(phrase, start)
            if index < 0:
                break
            hits.append((phrase, index))
            start = index + max(1, len(phrase))
    return sorted(hits, key=lambda item: item[1])


def _snippet(text: str, index: int, phrase: str, width: int = 30) -> str:
    start = max(0, index - width // 2)
    end = min(len(text), index + len(phrase) + width // 2)
    return text[start:end].replace("\n", " ").strip()


def analyze_prose(text: str, *, min_chars: int = 500) -> dict[str, Any]:
    """Return conservative, JSON-serialisable prose quality signals.

    Short drafts are deliberately left alone because frequency-based signals
    are unreliable on a few sentences.  A caller can still show the metrics
    for longer chapters and decide whether an LLM revision is warranted.
    """

    value = str(text or "").strip()
    sentences = _sentences(value)
    sentence_count = max(1, len(sentences))
    report: dict[str, Any] = {
        "score": 100,
        "sentence_count": len(sentences),
        "character_count": len(value),
        "counts": {},
        "issues": [],
        "insufficient_text": len(value) < min_chars,
    }
    if len(value) < min_chars:
        return report

    issues: list[dict[str, Any]] = []
    for key, phrases in _PATTERN_GROUPS.items():
        hits = _occurrences(value, phrases)
        count = len(hits)
        label, minimum, ratio, problem, fix = _GROUP_RULES[key]
        threshold = max(minimum, ceil(sentence_count * ratio))
        report["counts"][key] = count
        if count < threshold:
            continue
        severity = "major" if count >= threshold * 1.8 else "minor"
        examples: list[str] = []
        for phrase, index in hits:
            example = _snippet(value, index, phrase)
            if example and example not in examples:
                examples.append(example)
            if len(examples) >= 3:
                break
        issues.append(
            {
                "code": f"style_{key}",
                "severity": severity,
                "count": count,
                "threshold": threshold,
                "evidence": "；".join(examples),
                "problem": f"{label}：{problem}",
                "fix": fix,
            }
        )

    report["issues"] = issues
    report["score"] = max(
        0,
        100
        - sum(22 if issue["severity"] == "major" else 9 for issue in issues),
    )
    return report


__all__ = ["analyze_prose"]
