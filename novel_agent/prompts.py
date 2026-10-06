"""Prompt builders for the autonomous novel-writing workflow.

The builders deliberately accept plain mappings instead of importing the data
model layer.  This keeps the LLM boundary easy to test and lets callers pass a
dataclass converted with ``asdict`` or already-persisted JSON unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


STORY_BIBLE_SCHEMA = """{
  "synopsis": "适合平台作品详情页的长版作品简介，约 120-260 字，不泄露最终结局",
  "short_synopsis": "适合列表页的一句话简介，约 30-80 字",
  "world_setting": "时代、地点、社会结构和关键环境的完整描述",
  "central_conflict": "核心冲突、失败代价与明确结局方向",
  "themes": ["主题"],
  "tone": "整体基调",
  "style_guide": ["可执行的叙事与语言规则"],
  "rules": ["全书不得违反的世界规则或连续性约束"],
  "characters": [{
    "id": "char_唯一ID", "name": "姓名", "role": "主角/配角/反派",
    "description": "角色定位与辨识点", "age": "年龄或年龄段", "appearance": "外貌辨识点",
    "personality": ["稳定性格特征"], "background": "经历、秘密和成因",
    "goals": ["外在目标与内在需求"], "conflicts": ["恐惧、缺陷和阻力"],
    "arc": "从起点经关键转变到终点的完整人物弧",
    "relationships": {"char_另一角色ID": "初始关系及计划变化"},
    "state": {"location": "开篇地点", "knows": [], "inventory": []}
  }],
  "outline": [{
    "number": 1, "title": "章节名", "summary": "本章因果与局势变化摘要",
    "purpose": "本章不可替代的叙事职责", "pov": "视角角色姓名",
    "characters": ["char_ID"], "beats": ["按先后顺序写明行动、阻力、转折和章末钩子"],
    "foreshadow_ids": ["clue_ID"], "target_words": 2500, "status": "planned"
  }],
  "foreshadows": [{
    "id": "clue_唯一ID", "description": "伏笔内容、埋设方式及预期回收意义",
    "status": "planned", "plant_chapter": 1, "payoff_chapter": 3,
    "related_characters": ["char_ID"], "notes": ["发展章节及隐蔽程度"]
  }]
}"""


CHAPTER_PLAN_SCHEMA = """{
  "number": 1,
  "title": "章节名",
  "summary": "本章因果与局势变化摘要",
  "purpose": "本章完成后故事发生的不可逆变化",
  "pov": "视角角色姓名",
  "characters": ["char_ID"],
  "beats": ["按顺序写明地点、参与者、具体行动、阻力、情绪变化和揭示信息"],
  "foreshadow_ids": ["本章必须处理的 clue_ID"],
  "target_words": 2500,
  "status": "planned"
}"""


REVIEW_SCHEMA = """{
  "score": 0,
  "decision": "pass或revise",
  "strengths": ["值得保留之处"],
  "issues": [{
    "severity": "critical/major/minor", "category": "continuity/character/plot/foreshadowing/pacing/style/prose",
    "evidence": "正文中的具体证据", "problem": "问题", "fix": "可执行修改方法"
  }],
  "continuity_checks": [{"constraint": "约束或事实", "result": "pass/fail", "note": "说明"}],
  "foreshadowing_checks": [{"clue_id": "clue_ID", "result": "pass/fail/missing", "note": "说明"}],
  "style_checks": [{"category": "boilerplate/repetition/explanation/dialogue/rhythm", "result": "pass/fail", "evidence": "原文短引", "note": "说明"}],
  "revision_instructions": ["按优先级排序、无需改变全书设定的具体改写指令"]
}"""


MEMORY_UPDATE_SCHEMA = """{
  "summary": "只保留因果链与关键变化的章节摘要",
  "timeline_events": [{"order": 1, "time": "故事内时间", "location": "地点", "event": "事件"}],
  "character_updates": {
    "char_ID": {"location": "章末地点", "physical_state": "身体状态", "emotional_state": "情绪与态度", "knows": ["已知事实"], "goals": ["当前目标"], "inventory": ["现有物品"]}
  },
  "relationship_updates": [{"from": "char_ID", "to": "char_ID", "change": "关系变化", "cause": "原因"}],
  "new_facts": [{"id": "fact_唯一ID", "fact": "后续不可矛盾的事实", "source": "正文证据"}],
  "foreshadow_updates": {"clue_ID": "planned/planted/developing/resolved/abandoned"},
  "unresolved_threads": ["仍待解决的问题"],
  "continuity_warnings": ["正文与既有设定可能存在的冲突"],
  "next_chapter_context": "供下一章直接使用的简短衔接状态"
}"""


_JSON_SYSTEM = (
    "你是严谨的长篇小说架构师。你只能依据提示中给出的资料工作，不得擅自覆盖明确约束。"
    "输出必须是一个可被标准 JSON 解析器读取的 JSON 对象；不要输出 Markdown 代码围栏、解释或思考过程。"
    "所有 ID 必须稳定且可复用，所有章节号必须是整数。"
)

_PROSE_SYSTEM = (
    "你是擅长人物弧线、场景调度和伏笔控制的专业小说家。"
    "必须使用创作需求指定的语言完成正文；未指定时使用简体中文。"
    "严格遵守故事圣经、连续性记忆和本章计划；不要泄露创作提纲，不要解释写作过程。"
)


def _json(value: Any) -> str:
    """Serialize prompt data deterministically and without ASCII escaping."""

    if value is None:
        return "null"
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    """Support domain dataclasses without coupling prompts to ``models``."""

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    raise TypeError(f"无法序列化到提示词 JSON：{type(value).__name__}")


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_story_bible_messages(config: Mapping[str, Any] | Any) -> list[dict[str, str]]:
    """Build the planning prompt that creates the complete story bible."""

    user = f"""根据以下创作需求，设计一部能够按章连续创作的完整小说。

创作需求：
{_json(config)}

要求：
1. 人物目标、缺陷、秘密和成长弧必须相互推动，不要只做标签堆砌。
2. 主线必须有升级、转折和明确结局；每章都要改变局势，不能是可删除的填充章。
3. 为重要伏笔分配稳定 clue_ID，明确埋设、发展和回收章节；回收必须改变理解或行动。
4. outline 的数量必须等于创作需求中的目标章节数，编号从 1 连续递增，target_words 采用需求值。
5. 世界规则和连续性约束必须具体、可检查；角色使用稳定 char_ID。
6. 不要模仿在世作家的独特文风；将风格要求转化为通用叙事特征。
7. 同时生成 synopsis 和 short_synopsis：简介要突出主角、目标、核心冲突、代价或悬念，
   面向读者而非解释创作过程；不得透露最终反转和结局，不得虚构故事圣经之外的设定。

严格按以下结构输出 JSON，并用完整内容替换示例文字：
{STORY_BIBLE_SCHEMA}"""
    return _messages(_JSON_SYSTEM, user)


SYNOPSIS_SCHEMA = """{
  "synopsis": "适合平台作品详情页的长版作品简介，约 120-260 字",
  "short_synopsis": "适合列表页的一句话简介，约 30-80 字"
}"""


def build_synopsis_messages(
    brief: Mapping[str, Any] | Any,
    story_bible: Mapping[str, Any] | Any,
) -> list[dict[str, str]]:
    """Build a prompt for generating or refreshing platform-facing copy."""

    user = f"""根据创作需求和故事圣经，生成一份可以直接用于小说平台的作品简介。

创作需求：
{_json(brief)}

故事圣经：
{_json(story_bible)}

写作要求：
1. 长版简介应包含主角、核心目标、主要阻力、失败代价和一个能吸引读者的悬念。
2. 不要透露最终结局、最终反转或关键谜底；不要把后续章节的秘密全部解释出来。
3. 只使用资料中明确出现的设定、人物和冲突，不要擅自添加新世界规则。
4. 不要写“本书讲述”“作者将带你”等元话语，不要输出章节大纲、评论或创作说明。
5. short_synopsis 必须是一句话，适合列表页、搜索结果或封面副标题。
6. 使用创作需求指定的语言；未指定时使用简体中文。

严格只输出以下 JSON 对象：
{SYNOPSIS_SCHEMA}"""
    return _messages(_JSON_SYSTEM, user)


def build_chapter_plan_messages(
    story_bible: Mapping[str, Any] | Any,
    chapter_outline: Mapping[str, Any] | Any,
    memory: Mapping[str, Any] | Sequence[Any] | Any,
    previous_summary: str = "",
) -> list[dict[str, str]]:
    """Build a scene-level plan for one chapter."""

    user = f"""把目标章节扩展为可直接写作的场景节拍表。

故事圣经：
{_json(story_bible)}

目标章节大纲：
{_json(chapter_outline)}

截至上一章的连续性记忆：
{_json(memory)}

上一章摘要：
{previous_summary or '这是第一章，没有上一章。'}

规划规则：
1. 开场必须与上一章章末状态无缝衔接；人物只能知道记忆中已获知的信息。
2. 每个 beat 都要包含具体行动、阻力和变化，多个 beat 之间形成清楚的因果链。
3. 只处理本章大纲指定的主线、人物变化和伏笔；不得提前回收后续伏笔。
4. clue_ID、char_ID 应复用故事圣经中的 ID，不要创造同义重复项。
5. 最后一个 beat 必须落实本章大纲中的章末钩子，但不能靠无依据巧合。

严格按以下结构输出 JSON：
{CHAPTER_PLAN_SCHEMA}"""
    return _messages(_JSON_SYSTEM, user)


def build_chapter_draft_messages(
    story_bible: Mapping[str, Any] | Any,
    chapter_plan: Mapping[str, Any] | Any,
    memory: Mapping[str, Any] | Sequence[Any] | Any,
    previous_excerpt: str = "",
    style_constraints: Mapping[str, Any] | Sequence[Any] | str | None = None,
    natural_prose: bool = True,
) -> list[dict[str, str]]:
    """Build the prose-generation prompt for a planned chapter."""

    natural_rules = (
        "7. 避免连续使用相同句式、空泛比喻和强刺激转折词；不要让每个动作都像高潮。\n"
        "8. 少直接命名情绪，优先用停顿、误动作、选择、对白和具体物件表现人物状态。\n"
        "9. 不要用解释句重复已经由行动呈现的信息，保留自然的节奏缓冲和不完整对白。"
        if natural_prose
        else ""
    )
    user = f"""根据资料写出本章完整正文。

故事圣经（只使用与本章有关的设定）：
{_json(story_bible)}

本章场景计划：
{_json(chapter_plan)}

当前连续性记忆：
{_json(memory)}

上一章末尾片段（用于语气和动作衔接）：
{previous_excerpt or '这是第一章。'}

附加风格约束：
{_json(style_constraints) if style_constraints is not None else '无'}

写作规则：
1. 只输出小说正文，不输出 JSON、章节分析、提纲、字数说明或 Markdown 代码围栏。
2. 用可见行动、选择、对白和感官细节呈现冲突；避免用总结代替关键场景。
3. 保持人物声音、视角和知识边界一致，不得为推动情节让人物突然降智或全知。
4. 伏笔要自然进入环境、行动或对白，不得标注“这是伏笔”；未计划的谜团不要随意新增。
5. 覆盖计划中的全部必要 beat，但允许为行文自然调整过渡；章末落实指定钩子。
6. 不复述故事圣经，不模仿在世作家的标志性表达，不引用受版权保护的原文。
{natural_rules}"""
    return _messages(_PROSE_SYSTEM, user)


def build_chapter_review_messages(
    story_bible: Mapping[str, Any] | Any,
    chapter_plan: Mapping[str, Any] | Any,
    draft: str,
    memory: Mapping[str, Any] | Sequence[Any] | Any,
    style_report: Mapping[str, Any] | None = None,
    natural_prose: bool = True,
) -> list[dict[str, str]]:
    """Build a structured continuity and quality review prompt."""

    natural_review = (
        "本地去模板化预检（仅作为可核对的线索，不可替代你的判断）：\n"
        + (_json(style_report) if style_report is not None else "未启用")
        + "\n"
        if natural_prose
        else ""
    )
    natural_rules = (
        "7. 同时检查语言是否出现机械套语、相同比喻密集重复、解释腔、角色口吻同质化和过度均匀的节奏。\n"
        "8. 去模板化问题只能提出局部、可安全执行的措辞或节奏修改；不得借此改变事件、人物动机、伏笔或世界规则。"
        if natural_prose
        else ""
    )
    style_mode_rule = (
        ""
        if natural_prose
        else "7. 本次未启用去模板化审稿，style_checks 必须返回空数组，不进行文风判断。"
    )
    user = f"""审查本章草稿，重点找出会破坏长篇连贯性的具体问题。

故事圣经：
{_json(story_bible)}

本章计划：
{_json(chapter_plan)}

写作前连续性记忆：
{_json(memory)}

{natural_review}

待审草稿：
---正文开始---
{draft}
---正文结束---

审稿规则：
1. 分别检查人物动机与知识边界、时间地点、世界规则、物品状态、因果链和视角一致性。
2. 对本章计划中的伏笔逐项检查是否埋设/回收；不得要求提前解释计划留待后文的谜团。
3. issues 必须引用草稿中的具体证据并给出局部可执行修复；不要泛泛评价。
4. critical 表示设定或因果崩溃，major 表示明显影响阅读或漏掉必要 beat，minor 表示润色项。
5. 存在任一 critical/major 问题时 decision 必须为 revise；否则可以 pass。
6. score 为 0 到 100 的整数。只评价，不改写正文。
{natural_rules}
{style_mode_rule}

严格按以下结构输出 JSON：
{REVIEW_SCHEMA}"""
    return _messages(_JSON_SYSTEM, user)


def build_chapter_revision_messages(
    story_bible: Mapping[str, Any] | Any,
    chapter_plan: Mapping[str, Any] | Any,
    draft: str,
    review: Mapping[str, Any] | Any,
    memory: Mapping[str, Any] | Sequence[Any] | Any,
    natural_prose: bool = True,
) -> list[dict[str, str]]:
    """Build a prose-only revision prompt from a structured review."""

    natural_rules = (
        "4. 去模板化时优先做局部句式调整：减少重复比喻、强刺激转折和情绪套语，避免把抽象情绪改成另一种套话。\n"
        "5. 保留所有事件顺序、人物知识、数字、地点、物品、伏笔和章末钩子；无法安全修改的内容保持原样。\n"
        "6. 输出完整的改写后正文，不要只输出差异、摘要、JSON、Markdown 代码围栏或说明。"
        if natural_prose
        else "4. 输出完整的改写后正文，不要只输出差异、摘要、JSON、Markdown 代码围栏或说明。"
    )
    user = f"""按照审稿意见改写本章，并保持原稿中被认可的部分。

故事圣经：
{_json(story_bible)}

本章计划：
{_json(chapter_plan)}

写作前连续性记忆：
{_json(memory)}

审稿意见：
{_json(review)}

原稿：
---正文开始---
{draft}
---正文结束---

改写规则：
1. 优先解决 critical，再解决 major；minor 修改不得引入新的连续性问题。
2. 不改变故事圣经中的既定结局、人物核心设定或后续章节职责。
3. 用正文内的行动、对白和细节完成修复，不得写解释性批注。
{natural_rules}"""
    return _messages(_PROSE_SYSTEM, user)


def build_memory_update_messages(
    story_bible: Mapping[str, Any] | Any,
    chapter_plan: Mapping[str, Any] | Any,
    finalized_text: str,
    current_memory: Mapping[str, Any] | Sequence[Any] | Any,
) -> list[dict[str, str]]:
    """Build the fact-extraction prompt used after finalizing a chapter."""

    user = f"""从定稿正文提取后续创作所需的增量记忆。事实以正文实际发生内容为准；计划中未写出的内容不能算已发生。

故事圣经：
{_json(story_bible)}

本章计划：
{_json(chapter_plan)}

写作前记忆：
{_json(current_memory)}

本章定稿正文：
---正文开始---
{finalized_text}
---正文结束---

提取规则：
1. 摘要保留“因为—所以”的因果链、人物选择和不可逆变化，省略修辞与普通对白。
2. 只记录后续可能影响情节的状态；人物知识必须写清是谁知道什么。
3. 伏笔状态只能依据正文证据更新；使用既有 clue_ID，不确定时保留原状态并加入 warning。
4. new_facts 必须原子化、可检查，不得把推测当事实；新 ID 要稳定且避免与旧事实重复。
5. 如正文违反故事圣经或既有记忆，只报告 continuity_warnings，不得暗中篡改旧事实来消除冲突。

严格按以下结构输出 JSON：
{MEMORY_UPDATE_SCHEMA}"""
    return _messages(_JSON_SYSTEM, user)


# Short aliases keep orchestration code readable and preserve compatibility
# with early prototypes that omitted the ``build_`` prefix.
story_bible_messages = build_story_bible_messages
chapter_plan_messages = build_chapter_plan_messages
chapter_draft_messages = build_chapter_draft_messages
chapter_review_messages = build_chapter_review_messages
chapter_revision_messages = build_chapter_revision_messages
memory_update_messages = build_memory_update_messages
synopsis_messages = build_synopsis_messages
build_review_messages = build_chapter_review_messages
build_revision_messages = build_chapter_revision_messages


__all__ = [
    "STORY_BIBLE_SCHEMA",
    "CHAPTER_PLAN_SCHEMA",
    "REVIEW_SCHEMA",
    "MEMORY_UPDATE_SCHEMA",
    "SYNOPSIS_SCHEMA",
    "build_story_bible_messages",
    "build_chapter_plan_messages",
    "build_chapter_draft_messages",
    "build_chapter_review_messages",
    "build_chapter_revision_messages",
    "build_memory_update_messages",
    "build_synopsis_messages",
    "story_bible_messages",
    "chapter_plan_messages",
    "chapter_draft_messages",
    "chapter_review_messages",
    "chapter_revision_messages",
    "memory_update_messages",
    "synopsis_messages",
    "build_review_messages",
    "build_revision_messages",
]
