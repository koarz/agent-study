"""Abstract prose-style profiles extracted from local reference text.

The extractor is deliberately deterministic and local.  It records aggregate
statistics, categorical tendencies and generic writing guidance only; source
sentences and excerpts are never persisted in the profile or exposed through
``to_prompt_dict``.  This keeps reference material separate from story RAG and
from the prompts used to generate prose.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .storage import atomic_write_json


STYLE_PROFILE_SCHEMA_VERSION = 1
DEFAULT_ANALYSIS_CHARS = 2_000_000

STYLE_DIMENSIONS: tuple[str, ...] = (
    "sentence_structure",
    "paragraphing",
    "dialogue",
    "narrative_perspective",
    "description_balance",
    "pace",
    "information_reveal",
    "language_temperature",
    "rhetoric",
    "explanation",
    "chapter_endings",
    "conflict_suspense",
)

DIMENSION_ALIASES: dict[str, str] = {
    "sentence_structure": "sentence_structure",
    "sentence": "sentence_structure",
    "sentences": "sentence_structure",
    "句式": "sentence_structure",
    "句长": "sentence_structure",
    "平均句长": "sentence_structure",
    "paragraphing": "paragraphing",
    "paragraph": "paragraphing",
    "段落": "paragraphing",
    "段落长度": "paragraphing",
    "平均段落长度": "paragraphing",
    "dialogue": "dialogue",
    "对白": "dialogue",
    "对白风格": "dialogue",
    "narrative_perspective": "narrative_perspective",
    "perspective": "narrative_perspective",
    "pov": "narrative_perspective",
    "叙事视角": "narrative_perspective",
    "视角": "narrative_perspective",
    "description_balance": "description_balance",
    "description": "description_balance",
    "描写": "description_balance",
    "描写风格": "description_balance",
    "描写比例": "description_balance",
    "pace": "pace",
    "pacing": "pace",
    "节奏": "pace",
    "叙事节奏": "pace",
    "information_reveal": "information_reveal",
    "reveal": "information_reveal",
    "信息揭示": "information_reveal",
    "信息揭示方式": "information_reveal",
    "language_temperature": "language_temperature",
    "temperature": "language_temperature",
    "语言冷暖": "language_temperature",
    "冷暖": "language_temperature",
    "rhetoric": "rhetoric",
    "修辞": "rhetoric",
    "比喻修辞": "rhetoric",
    "explanation": "explanation",
    "解释": "explanation",
    "解释程度": "explanation",
    "chapter_endings": "chapter_endings",
    "ending": "chapter_endings",
    "endings": "chapter_endings",
    "章节结尾": "chapter_endings",
    "章末": "chapter_endings",
    "conflict_suspense": "conflict_suspense",
    "conflict": "conflict_suspense",
    "suspense": "conflict_suspense",
    "冲突悬念": "conflict_suspense",
    "冲突和悬念": "conflict_suspense",
}

_METRIC_DIMENSIONS: dict[str, str] = {
    "average_sentence_length": "sentence_structure",
    "short_sentence_ratio": "sentence_structure",
    "long_sentence_ratio": "sentence_structure",
    "average_paragraph_length": "paragraphing",
    "dialogue_ratio": "dialogue",
    "average_dialogue_length": "dialogue",
    "dialogue_question_ratio": "dialogue",
    "action_ratio": "description_balance",
    "psychology_ratio": "description_balance",
    "environment_ratio": "description_balance",
    "pace_score": "pace",
    "direct_reveal_ratio": "information_reveal",
    "progressive_reveal_ratio": "information_reveal",
    "withholding_ratio": "information_reveal",
    "language_temperature_score": "language_temperature",
    "rhetoric_density_per_1000_chars": "rhetoric",
    "explanation_ratio": "explanation",
    "conflict_ratio": "conflict_suspense",
    "suspense_ratio": "conflict_suspense",
}

_CHAPTER_HEADING_RE = re.compile(
    r"^(?:第\s*[零〇一二三四五六七八九十百千万两0-9]+\s*[章节卷回部篇]|"
    r"(?:chapter|part|volume)\s+[0-9ivxlcdm]+\b)",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
_DIALOGUE_RE = re.compile(
    r"“([^”\n]{1,3000})”|「([^」\n]{1,3000})」|『([^』\n]{1,3000})』|"
    r'(?<![A-Za-z0-9])"([^"\n]{1,3000})"'
)

_ACTION_WORDS = (
    "走", "跑", "冲", "退", "转身", "抬手", "伸手", "抓", "握", "推", "拉",
    "拔", "挥", "踢", "砍", "刺", "跳", "落下", "站起", "坐下", "打开", "关上",
    "迈步", "追", "逃", "扑", "闪", "躲", "撞", "按住", "扔", "接住",
)
_PSYCHOLOGY_WORDS = (
    "心中", "心里", "内心", "念头", "想到", "想起", "觉得", "意识到", "明白",
    "知道", "记得", "怀疑", "犹豫", "害怕", "恐惧", "担忧", "惊讶", "愤怒",
    "后悔", "决定", "猜测", "困惑", "希望", "绝望", "情绪", "直觉",
)
_ENVIRONMENT_WORDS = (
    "天空", "夜色", "阳光", "月光", "灯光", "阴影", "风声", "雨声", "雾气",
    "空气", "街道", "房间", "屋内", "走廊", "山谷", "树林", "河面", "海面",
    "地面", "墙壁", "门窗", "温度", "寒意", "热浪", "尘土", "气味", "四周",
)
_DIRECT_REVEAL_WORDS = (
    "原来", "其实", "真相", "答案是", "这意味着", "也就是说", "终于明白",
    "终于知道", "揭开", "证实", "确认", "正是", "竟然是",
)
_PROGRESSIVE_REVEAL_WORDS = (
    "线索", "蛛丝马迹", "似乎", "也许", "或许", "怀疑", "察觉", "隐约", "逐渐",
    "一点点", "端倪", "迹象", "推测", "猜测", "暗示", "发现", "注意到",
)
_WITHHOLDING_WORDS = (
    "没有回答", "沉默", "不置可否", "欲言又止", "没有解释", "避而不谈", "秘密",
    "谜团", "不肯说", "尚未知道", "无人知晓", "藏着", "隐瞒", "未曾提起",
)
_COLD_WORDS = (
    "冷", "冰", "寒", "阴沉", "灰暗", "黑暗", "沉默", "克制", "锋利", "残酷",
    "漠然", "死寂", "血腥", "压抑", "疏离", "凛冽", "僵硬", "戒备",
)
_WARM_WORDS = (
    "温暖", "暖意", "笑意", "温柔", "明亮", "希望", "安慰", "拥抱", "喜悦",
    "柔和", "亲切", "安心", "轻快", "热情", "善意", "幸福", "体贴",
)
_RHETORIC_WORDS = (
    "如同", "仿佛", "像是", "宛如", "犹如", "好似", "一般", "般", "似的",
    "恰似", "好像", "如一", "化作", "吞没", "撕裂", "燃烧着",
)
_EXPLANATION_WORDS = (
    "因为", "所以", "因此", "意味着", "也就是说", "换句话说", "显然", "事实上",
    "原因是", "这说明", "换言之", "可见", "正因为", "由此", "从而", "必然",
)
_CONFLICT_WORDS = (
    "战斗", "交锋", "冲突", "对峙", "敌人", "威胁", "危险", "危机", "追杀",
    "逃亡", "攻击", "阻止", "背叛", "阴谋", "陷阱", "争夺", "反抗", "杀意",
    "决斗", "围攻", "报复", "逼迫", "失控", "代价", "生死",
)
_SUSPENSE_WORDS = (
    "忽然", "突然", "却", "竟", "不料", "是谁", "为什么", "秘密", "谜", "未知",
    "异常", "诡异", "不对劲", "真相", "隐瞒", "线索", "踪迹", "预感", "等待",
)


class StyleProfileError(ValueError):
    """Raised when a style profile or mix specification is invalid."""


def _plain_text(value: Any, label: str, *, maximum: int = 240) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StyleProfileError(f"{label} 必须是非空字符串")
    cleaned = " ".join(value.split())
    if len(cleaned) > maximum:
        raise StyleProfileError(f"{label} 过长，最多 {maximum} 个字符")
    return cleaned


def _guidance_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise StyleProfileError(f"{label} 必须是非空字符串数组")
    if len(value) > 8:
        raise StyleProfileError(f"{label} 最多包含 8 条规则")
    result: list[str] = []
    for index, item in enumerate(value, start=1):
        cleaned = _plain_text(item, f"{label}[{index}]", maximum=220)
        if cleaned not in result:
            result.append(cleaned)
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StyleProfileError(f"{label} 必须是数字")
    number = float(value)
    if not math.isfinite(number):
        raise StyleProfileError(f"{label} 必须是有限数字")
    return number


def _clean_signal(value: Any, label: str) -> str | float | int | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _finite_number(value, label)
    if isinstance(value, str):
        return _plain_text(value, label, maximum=120)
    raise StyleProfileError(f"{label} 只能是字符串、数字或布尔值")


def _normalise_dimension(key: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StyleProfileError(f"dimensions.{key} 必须是对象")
    label = _plain_text(value.get("label"), f"dimensions.{key}.label", maximum=120)
    guidance = _guidance_list(value.get("guidance"), f"dimensions.{key}.guidance")
    raw_signals = value.get("signals", {})
    if not isinstance(raw_signals, Mapping):
        raise StyleProfileError(f"dimensions.{key}.signals 必须是对象")
    signals = {
        str(signal_key): _clean_signal(
            signal_value,
            f"dimensions.{key}.signals.{signal_key}",
        )
        for signal_key, signal_value in raw_signals.items()
    }
    result: dict[str, Any] = {
        "label": label,
        "signals": signals,
        "guidance": guidance,
    }
    raw_components = value.get("components")
    if raw_components is not None:
        if not isinstance(raw_components, list) or not raw_components:
            raise StyleProfileError(f"dimensions.{key}.components 必须是非空数组")
        components: list[dict[str, Any]] = []
        for index, component in enumerate(raw_components, start=1):
            if not isinstance(component, Mapping):
                raise StyleProfileError(
                    f"dimensions.{key}.components[{index}] 必须是对象"
                )
            weight = _finite_number(
                component.get("weight"),
                f"dimensions.{key}.components[{index}].weight",
            )
            if weight <= 0 or weight > 100:
                raise StyleProfileError("风格分量权重必须大于 0 且不超过 100")
            component_guidance = component.get("guidance", [])
            if component_guidance:
                component_guidance = _guidance_list(
                    component_guidance,
                    f"dimensions.{key}.components[{index}].guidance",
                )[:3]
            else:
                component_guidance = []
            components.append(
                {
                    "profile": _plain_text(
                        component.get("profile"),
                        f"dimensions.{key}.components[{index}].profile",
                        maximum=120,
                    ),
                    "weight": round(weight, 4),
                    "label": _plain_text(
                        component.get("label"),
                        f"dimensions.{key}.components[{index}].label",
                        maximum=120,
                    ),
                    "guidance": component_guidance,
                }
            )
        result["components"] = components
        base_weight = _finite_number(
            value.get("base_weight", 0),
            f"dimensions.{key}.base_weight",
        )
        if base_weight < 0 or base_weight > 100:
            raise StyleProfileError("base_weight 必须在 0 到 100 之间")
        result["base_weight"] = round(base_weight, 4)
    return result


@dataclass(slots=True)
class StyleProfile:
    """Validated aggregate style characteristics safe for prompt injection."""

    name: str
    language: str
    metrics: dict[str, float]
    dimensions: dict[str, dict[str, Any]]
    source: dict[str, Any]
    schema_version: int = STYLE_PROFILE_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != STYLE_PROFILE_SCHEMA_VERSION:
            raise StyleProfileError(
                f"不支持的文风画像版本：{self.schema_version}；"
                f"当前版本为 {STYLE_PROFILE_SCHEMA_VERSION}"
            )
        self.name = _plain_text(self.name, "StyleProfile.name", maximum=120)
        self.language = _plain_text(
            self.language,
            "StyleProfile.language",
            maximum=40,
        )
        if not isinstance(self.metrics, Mapping):
            raise StyleProfileError("StyleProfile.metrics 必须是对象")
        self.metrics = {
            str(key): round(_finite_number(value, f"metrics.{key}"), 6)
            for key, value in self.metrics.items()
        }
        if not isinstance(self.dimensions, Mapping):
            raise StyleProfileError("StyleProfile.dimensions 必须是对象")
        unknown = set(self.dimensions) - set(STYLE_DIMENSIONS)
        missing = set(STYLE_DIMENSIONS) - set(self.dimensions)
        if unknown:
            raise StyleProfileError(
                "文风画像包含未知维度：" + "、".join(sorted(unknown))
            )
        if missing:
            raise StyleProfileError(
                "文风画像缺少维度：" + "、".join(sorted(missing))
            )
        self.dimensions = {
            key: _normalise_dimension(key, self.dimensions[key])
            for key in STYLE_DIMENSIONS
        }
        if not isinstance(self.source, Mapping):
            raise StyleProfileError("StyleProfile.source 必须是对象")
        clean_source: dict[str, Any] = {}
        for key, value in self.source.items():
            label = f"source.{key}"
            if isinstance(value, list):
                if len(value) > 50:
                    raise StyleProfileError(f"{label} 条目过多")
                clean_source[str(key)] = [
                    _clean_signal(item, f"{label}[{index}]")
                    for index, item in enumerate(value)
                ]
            else:
                clean_source[str(key)] = _clean_signal(value, label)
        self.source = clean_source

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "language": self.language,
            "metrics": dict(self.metrics),
            "dimensions": {
                key: dict(value) for key, value in self.dimensions.items()
            },
            "source": dict(self.source),
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return only abstract, allow-listed fields for model prompts."""

        self.validate()
        prompt_dimensions: dict[str, dict[str, Any]] = {}
        for key in STYLE_DIMENSIONS:
            value = self.dimensions[key]
            item: dict[str, Any] = {
                "label": value["label"],
                "signals": dict(value.get("signals", {})),
                "guidance": list(value["guidance"]),
            }
            if value.get("components"):
                item["components"] = [
                    {
                        "profile": component["profile"],
                        "weight": component["weight"],
                        "label": component["label"],
                        "guidance": list(component.get("guidance", [])),
                    }
                    for component in value["components"]
                ]
                item["base_weight"] = value.get("base_weight", 0)
            prompt_dimensions[key] = item
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "language": self.language,
            "metrics": dict(self.metrics),
            "dimensions": prompt_dimensions,
            "reference_policy": (
                "只应用抽象统计特征和通用写作规则；不得复现、续接或引用参考文本。"
            ),
        }

    def save(self, path: str | Path) -> Path:
        return atomic_write_json(Path(path).expanduser(), self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StyleProfile:
        if not isinstance(data, Mapping):
            raise StyleProfileError("文风画像根节点必须是 JSON 对象")
        allowed = {
            "schema_version",
            "name",
            "language",
            "metrics",
            "dimensions",
            "source",
        }
        unknown = set(data) - allowed
        if unknown:
            raise StyleProfileError(
                "文风画像包含未知顶层字段：" + "、".join(sorted(unknown))
            )
        profile = cls(
            schema_version=data.get("schema_version", STYLE_PROFILE_SCHEMA_VERSION),
            name=data.get("name", ""),
            language=data.get("language", "zh-CN"),
            metrics=dict(data.get("metrics", {}))
            if isinstance(data.get("metrics", {}), Mapping)
            else data.get("metrics", {}),
            dimensions=dict(data.get("dimensions", {}))
            if isinstance(data.get("dimensions", {}), Mapping)
            else data.get("dimensions", {}),
            source=dict(data.get("source", {}))
            if isinstance(data.get("source", {}), Mapping)
            else data.get("source", {}),
        )
        profile.validate()
        return profile

    @classmethod
    def load(cls, path: str | Path) -> StyleProfile:
        target = Path(path).expanduser()
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"文风画像不存在：{target}") from None
        except json.JSONDecodeError as exc:
            raise StyleProfileError(
                f"文风画像不是有效 JSON：{target}（{exc}）"
            ) from exc
        return cls.from_dict(value)


def normalise_dimension_name(value: str) -> str:
    key = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    dimension = DIMENSION_ALIASES.get(key)
    if dimension is None:
        readable = "、".join(STYLE_DIMENSIONS)
        raise StyleProfileError(f"未知文风维度 {value!r}；可用维度：{readable}")
    return dimension


def _compile_phrases(values: Sequence[str]) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(item) for item in sorted(values, key=len, reverse=True)))


_ACTION_RE = _compile_phrases(_ACTION_WORDS)
_PSYCHOLOGY_RE = _compile_phrases(_PSYCHOLOGY_WORDS)
_ENVIRONMENT_RE = _compile_phrases(_ENVIRONMENT_WORDS)
_DIRECT_REVEAL_RE = _compile_phrases(_DIRECT_REVEAL_WORDS)
_PROGRESSIVE_REVEAL_RE = _compile_phrases(_PROGRESSIVE_REVEAL_WORDS)
_WITHHOLDING_RE = _compile_phrases(_WITHHOLDING_WORDS)
_COLD_RE = _compile_phrases(_COLD_WORDS)
_WARM_RE = _compile_phrases(_WARM_WORDS)
_RHETORIC_RE = _compile_phrases(_RHETORIC_WORDS)
_EXPLANATION_RE = _compile_phrases(_EXPLANATION_WORDS)
_CONFLICT_RE = _compile_phrases(_CONFLICT_WORDS)
_SUSPENSE_RE = _compile_phrases(_SUSPENSE_WORDS)


def _decode_source(raw: bytes, path: Path) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise StyleProfileError(f"无法识别参考文本编码：{path}")


def _clean_paragraph(value: str) -> str:
    return " ".join(value.strip().split())


def _paragraphs_and_endings(text: str) -> tuple[list[str], list[str], int]:
    paragraphs: list[str] = []
    endings: list[str] = []
    chapter_count = 0
    last_paragraph = ""
    in_chapter = False
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = _clean_paragraph(raw_line)
        if not line:
            continue
        if _CHAPTER_HEADING_RE.match(line):
            if in_chapter and last_paragraph:
                endings.append(last_paragraph)
            chapter_count += 1
            in_chapter = True
            last_paragraph = ""
            continue
        if len(line) <= 6 and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", line):
            continue
        paragraphs.append(line)
        last_paragraph = line
    if in_chapter and last_paragraph:
        endings.append(last_paragraph)
    return paragraphs, endings, chapter_count


def _stratified_sample(paragraphs: Sequence[str], max_chars: int) -> list[str]:
    if not paragraphs:
        return []
    total_chars = sum(len(item) for item in paragraphs)
    if max_chars <= 0 or total_chars <= max_chars:
        return list(paragraphs)
    estimated = max(1, int(len(paragraphs) * max_chars / total_chars))
    if estimated == 1:
        return [paragraphs[len(paragraphs) // 2][:max_chars]]
    indices = [
        round(index * (len(paragraphs) - 1) / (estimated - 1))
        for index in range(estimated)
    ]
    sampled: list[str] = []
    used = 0
    for index in dict.fromkeys(indices):
        remaining = max_chars - used
        if remaining <= 0:
            break
        paragraph = paragraphs[index]
        if len(paragraph) > remaining:
            paragraph = paragraph[:remaining]
        if paragraph:
            sampled.append(paragraph)
            used += len(paragraph)
    return sampled


def _sentences(paragraphs: Sequence[str]) -> list[str]:
    return [
        sentence.strip()
        for paragraph in paragraphs
        for sentence in _SENTENCE_RE.findall(paragraph)
        if sentence.strip()
    ]


def _visible_length(value: str) -> int:
    return len(re.sub(r"[\s。！？!?；;，,、：“”‘’「」『』()（）《》\[\]]+", "", value))


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def _rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _dialogue_metrics(paragraphs: Sequence[str], total_chars: int) -> tuple[float, float, float]:
    utterances: list[str] = []
    for paragraph in paragraphs:
        for match in _DIALOGUE_RE.finditer(paragraph):
            utterance = next((item for item in match.groups() if item is not None), "")
            if utterance.strip():
                utterances.append(utterance.strip())
    dialogue_chars = sum(_visible_length(item) for item in utterances)
    average_length = _ratio(dialogue_chars, len(utterances))
    question_ratio = _ratio(
        sum(1 for item in utterances if "？" in item or "?" in item),
        len(utterances),
    )
    return (
        _ratio(dialogue_chars, total_chars),
        average_length,
        question_ratio,
    )


def _narrative_perspective(paragraphs: Sequence[str]) -> tuple[str, dict[str, float]]:
    narrative = "\n".join(_DIALOGUE_RE.sub("", item) for item in paragraphs)
    first = len(re.findall(r"我们|咱们|本人|我", narrative))
    second = len(re.findall(r"你们|您|你", narrative))
    third = len(re.findall(r"他们|她们|它们|他|她|它", narrative))
    total = first + second + third
    first_ratio = _ratio(first, total)
    second_ratio = _ratio(second, total)
    third_ratio = _ratio(third, total)
    if first_ratio >= 0.55:
        label = "第一人称叙事倾向"
    elif third_ratio >= 0.62:
        label = "第三人称叙事倾向"
    elif second_ratio >= 0.35:
        label = "第二人称或强对话指向倾向"
    else:
        label = "混合或多视角叙事倾向"
    return label, {
        "first_person_ratio": _rounded(first_ratio),
        "second_person_ratio": _rounded(second_ratio),
        "third_person_ratio": _rounded(third_ratio),
    }


def _description_ratios(sentences: Sequence[str]) -> tuple[float, float, float]:
    action = sum(1 for item in sentences if _ACTION_RE.search(item))
    psychology = sum(1 for item in sentences if _PSYCHOLOGY_RE.search(item))
    environment = sum(1 for item in sentences if _ENVIRONMENT_RE.search(item))
    total = action + psychology + environment
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return action / total, psychology / total, environment / total


def _chapter_ending_metrics(endings: Sequence[str]) -> tuple[str, dict[str, float]]:
    counts = {
        "question": 0,
        "reveal": 0,
        "suspense": 0,
        "action": 0,
        "reflection": 0,
        "calm": 0,
    }
    for ending in endings:
        tail = ending[-240:]
        if "？" in tail or "?" in tail:
            category = "question"
        elif _DIRECT_REVEAL_RE.search(tail):
            category = "reveal"
        elif _SUSPENSE_RE.search(tail):
            category = "suspense"
        elif _ACTION_RE.search(tail):
            category = "action"
        elif _PSYCHOLOGY_RE.search(tail):
            category = "reflection"
        else:
            category = "calm"
        counts[category] += 1
    total = sum(counts.values())
    ratios = {key: _rounded(_ratio(value, total)) for key, value in counts.items()}
    labels = {
        "question": "疑问式章末钩子",
        "reveal": "信息揭示式章末钩子",
        "suspense": "悬念或反转式章末钩子",
        "action": "动作未完成式章末推进",
        "reflection": "人物思考式收束",
        "calm": "平静收束或场景落点",
    }
    dominant = max(counts, key=counts.get) if total else "calm"
    return labels[dominant], ratios


def _description_label(action: float, psychology: float, environment: float) -> str:
    values = {
        "动作驱动": action,
        "心理驱动": psychology,
        "环境氛围驱动": environment,
    }
    dominant, value = max(values.items(), key=lambda item: item[1])
    if value < 0.43:
        return "动作、心理与环境较均衡"
    return dominant


def _profile_dimensions(metrics: Mapping[str, float], perspective: tuple[str, dict[str, float]], ending: tuple[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    sentence_length = metrics["average_sentence_length"]
    paragraph_length = metrics["average_paragraph_length"]
    dialogue_ratio = metrics["dialogue_ratio"]
    action = metrics["action_ratio"]
    psychology = metrics["psychology_ratio"]
    environment = metrics["environment_ratio"]
    pace_score = metrics["pace_score"]
    rhetoric_density = metrics["rhetoric_density_per_1000_chars"]
    explanation_ratio = metrics["explanation_ratio"]

    sentence_label = (
        "短句为主" if sentence_length < 15 else
        "中短句为主" if sentence_length < 23 else
        "中长句为主" if sentence_length < 34 else
        "长句和复合句较多"
    )
    paragraph_label = (
        "短段落密集" if paragraph_length < 45 else
        "中等段落长度" if paragraph_length < 100 else
        "长段落展开"
    )
    dialogue_label = (
        "对白克制、叙述占主导" if dialogue_ratio < 0.16 else
        "对白与叙述较均衡" if dialogue_ratio < 0.36 else
        "对白驱动明显"
    )
    pace_label = "快速推进" if pace_score >= 0.62 else "舒缓蓄势" if pace_score <= 0.4 else "张弛适中"
    direct = metrics["direct_reveal_ratio"]
    progressive = metrics["progressive_reveal_ratio"]
    withholding = metrics["withholding_ratio"]
    reveal_values = {
        "直接解释和确认": direct,
        "依靠线索渐进揭示": progressive,
        "延迟回答并保留空白": withholding,
    }
    reveal_label = max(reveal_values.items(), key=lambda item: item[1])[0]
    temperature = metrics["language_temperature_score"]
    temperature_label = "偏冷峻克制" if temperature <= -0.2 else "偏温暖亲近" if temperature >= 0.2 else "冷暖中性"
    rhetoric_label = "修辞稀疏" if rhetoric_density < 2.5 else "修辞适中" if rhetoric_density < 6.5 else "比喻和修辞较密"
    explanation_label = "低解释度" if explanation_ratio < 0.07 else "适度解释" if explanation_ratio < 0.16 else "解释性较强"
    conflict = metrics["conflict_ratio"]
    suspense = metrics["suspense_ratio"]
    conflict_label = (
        "高频冲突与悬念推进" if conflict + suspense >= 0.28 else
        "阶段性升级冲突" if conflict + suspense >= 0.13 else
        "低冲突、重铺垫蓄势"
    )
    perspective_label, perspective_signals = perspective
    ending_label, ending_signals = ending
    return {
        "sentence_structure": {
            "label": sentence_label,
            "signals": {
                "average_sentence_length": sentence_length,
                "short_sentence_ratio": metrics["short_sentence_ratio"],
                "long_sentence_ratio": metrics["long_sentence_ratio"],
            },
            "guidance": [
                f"句子平均长度控制在约 {sentence_length:.1f} 字附近，并保留自然波动。",
                "关键动作和判断可用短句落点，信息复杂处再使用较长句展开。",
            ],
        },
        "paragraphing": {
            "label": paragraph_label,
            "signals": {"average_paragraph_length": paragraph_length},
            "guidance": [
                f"段落平均约 {paragraph_length:.1f} 字，按动作、观察或话题变化自然分段。",
                "避免所有段落长度整齐一致，高潮处可缩短段落形成呼吸变化。",
            ],
        },
        "dialogue": {
            "label": dialogue_label,
            "signals": {
                "dialogue_ratio": dialogue_ratio,
                "average_dialogue_length": metrics["average_dialogue_length"],
                "dialogue_question_ratio": metrics["dialogue_question_ratio"],
            },
            "guidance": [
                f"对白正文占比约 {dialogue_ratio:.0%}，让对白承担行动、试探或信息交换。",
                "保持人物口吻差异，避免对白直接复述旁白已经说明的信息。",
            ],
        },
        "narrative_perspective": {
            "label": perspective_label,
            "signals": perspective_signals,
            "guidance": [
                f"采用{perspective_label}，每个场景维持稳定的感知与知识边界。",
                "视角切换必须有清楚的场景边界，不让人物获得未亲历或未获知的信息。",
            ],
        },
        "description_balance": {
            "label": _description_label(action, psychology, environment),
            "signals": {
                "action_ratio": action,
                "psychology_ratio": psychology,
                "environment_ratio": environment,
            },
            "guidance": [
                f"描写资源约按动作 {action:.0%}、心理 {psychology:.0%}、环境 {environment:.0%} 分配。",
                "同一信息优先选择最有效的一种呈现方式，避免动作、心理和旁白重复解释。",
            ],
        },
        "pace": {
            "label": pace_label,
            "signals": {"pace_score": pace_score},
            "guidance": [
                f"整体保持{pace_label}，用段落长度、对白比例和动作密度共同调速。",
                "重要转折前允许短暂蓄势，转折后用具体后果推动下一步行动。",
            ],
        },
        "information_reveal": {
            "label": reveal_label,
            "signals": {
                "direct_reveal_ratio": direct,
                "progressive_reveal_ratio": progressive,
                "withholding_ratio": withholding,
            },
            "guidance": [
                f"信息揭示以“{reveal_label}”为主要方式。",
                "先呈现可观察证据，再决定是否确认结论；不要用旁白提前说破后续谜底。",
            ],
        },
        "language_temperature": {
            "label": temperature_label,
            "signals": {"temperature_score": temperature},
            "guidance": [
                f"词语情感温度保持{temperature_label}，通过物件、动作和场景质感体现。",
                "不要连续使用抽象情绪形容词，让人物选择承担主要情感表达。",
            ],
        },
        "rhetoric": {
            "label": rhetoric_label,
            "signals": {"rhetoric_density_per_1000_chars": rhetoric_density},
            "guidance": [
                f"每千字比喻和显性修辞约 {rhetoric_density:.1f} 处，保持{rhetoric_label}。",
                "修辞必须服务观察或情绪转折，避免连续比喻同一对象。",
            ],
        },
        "explanation": {
            "label": explanation_label,
            "signals": {"explanation_ratio": explanation_ratio},
            "guidance": [
                f"保持{explanation_label}，因果连接和总结句约覆盖 {explanation_ratio:.0%} 的句子。",
                "行动和对白已经足以说明结论时，删去重复解释；复杂规则首次出现时可简短澄清。",
            ],
        },
        "chapter_endings": {
            "label": ending_label,
            "signals": ending_signals,
            "guidance": [
                f"章末主要采用“{ending_label}”。",
                "钩子必须来自本章因果链，让新问题、未完成动作或理解变化推动下一章。",
            ],
        },
        "conflict_suspense": {
            "label": conflict_label,
            "signals": {
                "conflict_ratio": conflict,
                "suspense_ratio": suspense,
            },
            "guidance": [
                f"冲突与悬念采用“{conflict_label}”的推进强度。",
                "每次升级都改变目标、代价或可选方案，避免只靠突然袭击重复制造刺激。",
            ],
        },
    }


def extract_style_profile(
    input_path: str | Path,
    *,
    name: str | None = None,
    language: str = "zh-CN",
    max_analysis_chars: int = DEFAULT_ANALYSIS_CHARS,
) -> StyleProfile:
    """Extract an aggregate style profile from a local plain-text work."""

    target = Path(input_path).expanduser()
    try:
        raw = target.read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(f"参考文本不存在：{target}") from None
    if not raw:
        raise StyleProfileError(f"参考文本为空：{target}")
    text, encoding = _decode_source(raw, target)
    paragraphs, endings, chapter_count = _paragraphs_and_endings(text)
    if not paragraphs:
        raise StyleProfileError("参考文本没有可分析的正文段落")
    sampled = _stratified_sample(paragraphs, max_analysis_chars)
    sampled_chars = sum(_visible_length(item) for item in sampled)
    if sampled_chars < 100:
        raise StyleProfileError("参考文本过短，至少需要约 100 个有效字符")
    sentences = _sentences(sampled)
    if not sentences:
        raise StyleProfileError("参考文本没有可识别的完整句子")
    sentence_lengths = [_visible_length(item) for item in sentences if _visible_length(item) > 0]
    paragraph_lengths = [_visible_length(item) for item in sampled if _visible_length(item) > 0]
    average_sentence = _ratio(sum(sentence_lengths), len(sentence_lengths))
    average_paragraph = _ratio(sum(paragraph_lengths), len(paragraph_lengths))
    short_sentence_ratio = _ratio(sum(1 for item in sentence_lengths if item <= 12), len(sentence_lengths))
    long_sentence_ratio = _ratio(sum(1 for item in sentence_lengths if item >= 36), len(sentence_lengths))
    dialogue_ratio, average_dialogue, dialogue_question_ratio = _dialogue_metrics(
        sampled,
        sampled_chars,
    )
    action_ratio, psychology_ratio, environment_ratio = _description_ratios(sentences)
    direct_reveal = _ratio(sum(1 for item in sentences if _DIRECT_REVEAL_RE.search(item)), len(sentences))
    progressive_reveal = _ratio(sum(1 for item in sentences if _PROGRESSIVE_REVEAL_RE.search(item)), len(sentences))
    withholding = _ratio(sum(1 for item in sentences if _WITHHOLDING_RE.search(item)), len(sentences))
    joined = "\n".join(sampled)
    cold_hits = len(_COLD_RE.findall(joined))
    warm_hits = len(_WARM_RE.findall(joined))
    temperature_score = _ratio(warm_hits - cold_hits, warm_hits + cold_hits)
    rhetoric_density = _ratio(len(_RHETORIC_RE.findall(joined)) * 1000, sampled_chars)
    explanation_ratio = _ratio(sum(1 for item in sentences if _EXPLANATION_RE.search(item)), len(sentences))
    conflict_ratio = _ratio(sum(1 for item in sentences if _CONFLICT_RE.search(item)), len(sentences))
    suspense_ratio = _ratio(sum(1 for item in sentences if _SUSPENSE_RE.search(item)), len(sentences))
    pace_score = max(
        0.0,
        min(
            1.0,
            0.36
            + short_sentence_ratio * 0.35
            + dialogue_ratio * 0.2
            + action_ratio * 0.2
            - long_sentence_ratio * 0.25
            - min(0.18, average_paragraph / 1000),
        ),
    )
    metrics = {
        "average_sentence_length": _rounded(average_sentence, 3),
        "average_paragraph_length": _rounded(average_paragraph, 3),
        "short_sentence_ratio": _rounded(short_sentence_ratio),
        "long_sentence_ratio": _rounded(long_sentence_ratio),
        "dialogue_ratio": _rounded(dialogue_ratio),
        "average_dialogue_length": _rounded(average_dialogue, 3),
        "dialogue_question_ratio": _rounded(dialogue_question_ratio),
        "action_ratio": _rounded(action_ratio),
        "psychology_ratio": _rounded(psychology_ratio),
        "environment_ratio": _rounded(environment_ratio),
        "pace_score": _rounded(pace_score),
        "direct_reveal_ratio": _rounded(direct_reveal),
        "progressive_reveal_ratio": _rounded(progressive_reveal),
        "withholding_ratio": _rounded(withholding),
        "language_temperature_score": _rounded(temperature_score),
        "rhetoric_density_per_1000_chars": _rounded(rhetoric_density, 3),
        "explanation_ratio": _rounded(explanation_ratio),
        "conflict_ratio": _rounded(conflict_ratio),
        "suspense_ratio": _rounded(suspense_ratio),
    }
    perspective = _narrative_perspective(sampled)
    ending_sample = _stratified_sample(endings, 120_000) if endings else []
    ending = _chapter_ending_metrics(ending_sample)
    profile = StyleProfile(
        name=name or target.stem,
        language=language,
        metrics=metrics,
        dimensions=_profile_dimensions(metrics, perspective, ending),
        source={
            "kind": "local_aggregate_analysis",
            "extractor_version": "heuristic-v1",
            "filename": target.name,
            "encoding": encoding,
            "content_sha256": hashlib.sha256(raw).hexdigest(),
            "source_characters": len(text),
            "analyzed_characters": sampled_chars,
            "source_paragraphs": len(paragraphs),
            "analyzed_paragraphs": len(sampled),
            "analyzed_sentences": len(sentences),
            "detected_chapters": chapter_count,
            "analyzed_chapter_endings": len(ending_sample),
        },
    )
    profile.validate()
    return profile


def _profile_aliases(
    profiles: Sequence[StyleProfile],
    paths: Sequence[Path] | None = None,
) -> dict[str, str]:
    aliases: dict[str, str] = {}

    def add(alias: str, profile_name: str) -> None:
        key = alias.strip().casefold()
        if not key:
            return
        existing = aliases.get(key)
        if existing is not None and existing != profile_name:
            raise StyleProfileError(f"文风画像别名冲突：{alias}")
        aliases[key] = profile_name

    names: set[str] = set()
    for index, profile in enumerate(profiles, start=1):
        if profile.name in names:
            raise StyleProfileError(f"文风画像名称重复：{profile.name}")
        names.add(profile.name)
        add(profile.name, profile.name)
        add(str(index), profile.name)
        if paths is not None:
            add(paths[index - 1].stem, profile.name)
    return aliases


def parse_style_mix_specs(
    specs: Sequence[str],
    profiles: Sequence[StyleProfile],
    *,
    paths: Sequence[Path] | None = None,
) -> dict[str, dict[str, float]]:
    """Parse ``dimension=profile:weight,...`` mix specifications."""

    aliases = _profile_aliases(profiles, paths)
    result: dict[str, dict[str, float]] = {}
    for raw_spec in specs:
        spec = str(raw_spec).strip()
        if "=" not in spec:
            raise StyleProfileError(
                f"文风混合参数格式错误：{raw_spec!r}；应为 维度=画像名:权重"
            )
        raw_dimension, raw_components = spec.split("=", 1)
        dimension = normalise_dimension_name(raw_dimension)
        components = [
            item.strip()
            for item in re.split(r"[,，]", raw_components)
            if item.strip()
        ]
        if not components:
            raise StyleProfileError(f"文风维度 {raw_dimension!r} 没有指定画像")
        weights = result.setdefault(dimension, {})
        for component in components:
            if ":" not in component:
                raise StyleProfileError(
                    f"文风分量格式错误：{component!r}；应为 画像名:权重"
                )
            raw_name, raw_weight = component.rsplit(":", 1)
            canonical_name = aliases.get(raw_name.strip().casefold())
            if canonical_name is None:
                available = "、".join(profile.name for profile in profiles)
                raise StyleProfileError(
                    f"未知文风画像 {raw_name!r}；已加载：{available}"
                )
            try:
                weight = float(raw_weight)
            except ValueError as exc:
                raise StyleProfileError(f"文风权重不是数字：{raw_weight!r}") from exc
            if not math.isfinite(weight) or weight < 0 or weight > 100:
                raise StyleProfileError("文风权重必须在 0 到 100 之间")
            weights[canonical_name] = weight
    return result


def _numeric_signal_average(
    profiles: Sequence[StyleProfile],
    weights: Mapping[str, float],
    dimension: str,
) -> dict[str, float]:
    keys = {
        key
        for profile in profiles
        for key, value in profile.dimensions[dimension].get("signals", {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    result: dict[str, float] = {}
    for key in sorted(keys):
        values = [
            (float(profile.dimensions[dimension]["signals"][key]), weights[profile.name])
            for profile in profiles
            if isinstance(profile.dimensions[dimension].get("signals", {}).get(key), (int, float))
            and weights.get(profile.name, 0) > 0
        ]
        total = sum(weight for _, weight in values)
        if total > 0:
            result[key] = _rounded(sum(value * weight for value, weight in values) / total, 6)
    return result


def mix_style_profiles(
    profiles: Sequence[StyleProfile],
    dimension_weights: Mapping[str, Mapping[str, float]] | None = None,
    *,
    name: str | None = None,
) -> StyleProfile:
    """Blend profiles independently per style dimension.

    Explicit weights are percentages of influence.  If their sum is below
    100, the remainder is retained as project/base style.  Sums above 100 are
    proportionally normalised.  Dimensions without an explicit mix use equal
    weights across all loaded profiles.
    """

    if not profiles:
        raise StyleProfileError("至少需要一个文风画像")
    validated = [StyleProfile.from_dict(profile.to_dict()) for profile in profiles]
    _profile_aliases(validated)
    raw_weights = dimension_weights or {}
    normalised_specs: dict[str, dict[str, float]] = {}
    for raw_dimension, values in raw_weights.items():
        dimension = normalise_dimension_name(raw_dimension)
        if not isinstance(values, Mapping):
            raise StyleProfileError(f"维度 {raw_dimension} 的混合权重必须是对象")
        unknown_profiles = set(values) - {profile.name for profile in validated}
        if unknown_profiles:
            raise StyleProfileError(
                "混合权重引用了未知画像：" + "、".join(sorted(unknown_profiles))
            )
        clean = {
            str(profile_name): _finite_number(weight, f"{dimension}.{profile_name}")
            for profile_name, weight in values.items()
        }
        if any(weight < 0 or weight > 100 for weight in clean.values()):
            raise StyleProfileError("文风权重必须在 0 到 100 之间")
        if sum(clean.values()) <= 0:
            raise StyleProfileError(f"维度 {dimension} 的权重总和必须大于 0")
        normalised_specs[dimension] = clean

    dimensions: dict[str, dict[str, Any]] = {}
    effective_weights: dict[str, dict[str, float]] = {}
    for dimension in STYLE_DIMENSIONS:
        if dimension in normalised_specs:
            weights = {
                profile.name: float(normalised_specs[dimension].get(profile.name, 0))
                for profile in validated
            }
            total = sum(weights.values())
            if total > 100:
                weights = {
                    profile_name: weight * 100 / total
                    for profile_name, weight in weights.items()
                }
                total = 100.0
            base_weight = max(0.0, 100.0 - total)
        else:
            equal = 100.0 / len(validated)
            weights = {profile.name: equal for profile in validated}
            base_weight = 0.0
        weights = {
            profile_name: round(weight, 4)
            for profile_name, weight in weights.items()
            if weight > 0
        }
        effective_weights[dimension] = weights
        components: list[dict[str, Any]] = []
        guidance: list[str] = []
        labels: list[str] = []
        for profile in validated:
            weight = weights.get(profile.name, 0)
            if weight <= 0:
                continue
            source_dimension = profile.dimensions[dimension]
            component_guidance = list(source_dimension["guidance"][:2])
            components.append(
                {
                    "profile": profile.name,
                    "weight": weight,
                    "label": source_dimension["label"],
                    "guidance": component_guidance,
                }
            )
            labels.append(f"{profile.name} {weight:g}%")
            guidance.extend(
                f"{profile.name}（{weight:g}%）：{item}"
                for item in component_guidance
            )
        if base_weight > 0:
            labels.append(f"项目基础风格 {base_weight:g}%")
            guidance.append(
                f"保留项目原有风格约束 {base_weight:g}%，画像不得覆盖人物声音、剧情事实和用户明确要求。"
            )
        dimensions[dimension] = {
            "label": " + ".join(labels),
            "signals": _numeric_signal_average(validated, weights, dimension),
            "guidance": guidance[:8],
            "components": components,
            "base_weight": round(base_weight, 4),
        }

    metric_keys = {key for profile in validated for key in profile.metrics}
    metrics: dict[str, float] = {}
    for key in sorted(metric_keys):
        dimension = _METRIC_DIMENSIONS.get(key)
        weights = (
            effective_weights[dimension]
            if dimension is not None
            else {profile.name: 1.0 for profile in validated}
        )
        values = [
            (profile.metrics[key], weights.get(profile.name, 0))
            for profile in validated
            if key in profile.metrics and weights.get(profile.name, 0) > 0
        ]
        total = sum(weight for _, weight in values)
        if total > 0:
            metrics[key] = _rounded(
                sum(value * weight for value, weight in values) / total,
                6,
            )

    profile = StyleProfile(
        name=name or " + ".join(item.name for item in validated),
        language=validated[0].language,
        metrics=metrics,
        dimensions=dimensions,
        source={
            "kind": "mixed_abstract_profiles",
            "profile_names": [item.name for item in validated],
        },
    )
    profile.validate()
    return profile


def load_style_profile_selection(
    paths: Sequence[str | Path],
    mix_specs: Sequence[str] = (),
    *,
    name: str | None = None,
) -> StyleProfile | None:
    """Load one profile or compose several profiles for a CLI invocation."""

    if not paths:
        if mix_specs:
            raise StyleProfileError("使用 --style-mix 前至少要提供一个 --style-profile")
        return None
    resolved = [Path(path).expanduser() for path in paths]
    profiles = [StyleProfile.load(path) for path in resolved]
    if len(profiles) == 1 and not mix_specs and name is None:
        return profiles[0]
    weights = parse_style_mix_specs(mix_specs, profiles, paths=resolved)
    return mix_style_profiles(profiles, weights, name=name)


__all__ = [
    "STYLE_PROFILE_SCHEMA_VERSION",
    "DEFAULT_ANALYSIS_CHARS",
    "STYLE_DIMENSIONS",
    "DIMENSION_ALIASES",
    "StyleProfile",
    "StyleProfileError",
    "extract_style_profile",
    "normalise_dimension_name",
    "parse_style_mix_specs",
    "mix_style_profiles",
    "load_style_profile_selection",
]
