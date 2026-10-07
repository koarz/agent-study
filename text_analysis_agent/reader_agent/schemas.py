"""模型输出的明确结构；只验证格式，事实与引用仍由原文校验和复核负责。"""


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def array(items, maximum):
    return {"type": "array", "items": items, "maxItems": maximum}


TEXT = {"type": "string"}
STATUS = {"type": "string", "enum": ["answered", "unclear", "not_found"]}
CITATION = obj({"chunk_id": TEXT, "quote": TEXT})
CLAIM = obj({"id": TEXT, "text": TEXT, "citations": array(CITATION, 8)})
ANSWER = obj({"status": STATUS, "claims": array(CLAIM, 12)})
ANSWER_WITH_HANDLES = obj({"status": STATUS, "claims": array(obj({"id": TEXT, "text": TEXT,
    "citations": array(obj({"chunk_id": TEXT, "quote_id": TEXT}), 8)}), 12)})
PLAN = obj({"queries": array(TEXT, 3)})
KEYWORD = {"type": "string", "maxLength": 12, "pattern": r"^[^\s，。！？；：,?!;:]+$"}
KEYWORDS = {**array(KEYWORD, 4), "minItems": 2}
QUERY_ENTITY = obj({'name': TEXT, 'kind': {'type': 'string', 'enum': ['person', 'organization', 'object', 'other']}})
QUERY_PLAN = obj({"literal": KEYWORDS, "paraphrase": KEYWORDS, "broader": KEYWORDS, "entities": array(QUERY_ENTITY, 6)})
QUERY_REVIEW = obj({'accepted_indices': array({'type': 'integer'}, 3), 'action_terms': array(KEYWORD, 3)})
VERIFY = obj({"verdicts": array(obj({"claim_id": TEXT, "supported": {"type": "boolean"}}), 12),
              "question_fully_answered": {"type": "boolean"}})
GRAPH = obj({"status": STATUS, "facts": array(obj({
    "id": TEXT, "subject": TEXT, "predicate": TEXT, "object": TEXT, "time_text": TEXT,
    "attribution": {"type": "string", "enum": ["narration", "dialogue", "rumor", "dream", "hypothesis", "unclear"]},
    "certainty": {"type": "string", "enum": ["explicit", "unclear"]}, "citations": array(CITATION, 8),
}), 12)})


def matches(value, schema):
    """核对本模块使用的结构约束，不尝试改名、猜测字段或修补内容。"""
    kind = schema["type"]
    if kind == "object":
        return (isinstance(value, dict) and set(value) == set(schema["properties"]) and
                all(matches(value[key], child) for key, child in schema["properties"].items()))
    if kind == "array":
        return isinstance(value, list) and schema.get('minItems', 0) <= len(value) <= schema["maxItems"] and all(matches(item, schema["items"]) for item in value)
    if kind == "integer":
        return type(value) is int
    if kind == "boolean":
        return type(value) is bool
    if kind == "string":
        import re
        return (isinstance(value, str) and ("enum" not in schema or value in schema["enum"])
                and len(value) <= schema.get('maxLength', len(value))
                and ('pattern' not in schema or re.fullmatch(schema['pattern'], value) is not None))
    return False

# 精读只允许选择已提供的段落编号，不接收规划器生成的事实。
SELECT_EVIDENCE = obj({'chunk_ids': array(TEXT, 8)})
PROOF_CHECK = obj({
    'status': {'type': 'string', 'enum': ['supported', 'missing', 'conflicting', 'not_applicable']},
    'reason': TEXT,
})
PROOF_VERIFY = obj({'verdicts': array(obj({
    'claim_id': TEXT, 'supported': {'type': 'boolean'},
    'checks': obj({field: PROOF_CHECK for field in ('subject', 'relation', 'attributes', 'timeline')}),
}), 12), 'question_fully_answered': {'type': 'boolean'}})

# 专名清单参与程序检查；输出中的地点全称不能仅存在于未引用的背景。
ANSWER_WITH_MENTIONS = obj({'status': STATUS, 'claims': array(obj({
    'id': TEXT, 'text': TEXT,
    'mentions': array(TEXT, 20),
    'citations': array(obj({'chunk_id': TEXT, 'quote_id': TEXT}), 8),
}), 12)})

SOURCE_ROLES = obj({'locations': array(obj({
    'quote_index': {'type': 'integer'}, 'quote': TEXT, 'place': TEXT,
    'role': {'type': 'string', 'enum': ['physical', 'affiliation', 'origin', 'unclear']},
}), 40)})

SINGLE_SOURCE_ROLES = obj({'locations': array(obj({
    'place': TEXT, 'role': {'type': 'string', 'enum': ['physical', 'affiliation', 'origin', 'unclear']},
}), 40)})
CLAIM_MENTIONS = obj({'claims': array(obj({'claim_id': TEXT, 'names': array(TEXT, 20)}), 12)})
