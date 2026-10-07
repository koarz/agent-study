"""限制检索与上下文规模，并执行精确引用校验和语义复核。"""

from __future__ import annotations

from . import prompts, schemas
from .corpus import Chunk, Corpus, terms
from .llm import Backend, ModelOutputError, request_json
from .logging_utils import emit
import logging
import re


def quote_catalog(chosen):
    """原文段落按连续区间编号，保留空格、换行、标点和真实字符偏移。"""
    evidence, references = [], {}
    for chunk in chosen.values():
        passages, left = [], 0
        # 优先按完整段落分段；超长段落再按句子边界切开，任何字符都不改写。
        ends = [match.end() for match in re.finditer(r'\r?\n', chunk.text)] + [len(chunk.text)]
        for end in ends:
            while left < end:
                right = min(end, left + 500)
                if right < end:
                    boundary = max(chunk.text.rfind(mark, left + 250, right) for mark in ('。', '！', '？', '；'))
                    if boundary >= 0:
                        right = boundary + 1
                quote = chunk.text[left:right]
                if quote.strip():
                    quote_id = 'p' + str(left)
                    references[(chunk.id, quote_id)] = (quote, left)
                    passages.append({'quote_id': quote_id, 'quote': quote})
                left = right
        item = chunk.to_dict()
        item.pop('text')
        evidence.append({**item, 'passages': passages})
    return evidence, references


def evidence_overview(candidates, question, budget):
    """用精确摘录展示所有候选；这一步只选待读段落，不产出事实或引用。"""
    catalog, _ = quote_catalog({c.id: c for c in candidates})
    words = set(terms(question))
    per_chunk = max(1, budget // max(1, len(catalog)))
    result = []
    for item in catalog:
        passages = item['passages']
        # 同时保留相关正文、开头和结尾，避免只看前部而漏掉事件结果与场景。
        ranked = sorted(range(len(passages)), key=lambda i: (
            -len(words.intersection(terms(passages[i]['quote']))), i))
        indices = list(dict.fromkeys([*ranked[:1], len(passages) - 1, 0, *ranked[1:]]))
        selected, used = {}, 0
        for index in indices:
            if index < 0:
                continue
            passage = passages[index]
            available = per_chunk - used
            if available <= 0:
                break
            quote = passage['quote']
            # 摘录可在概览中截短，正式回答始终重新读取完整的不可变段落。
            if len(quote) > available:
                if selected:
                    continue
                quote = quote[:available]
            selected[index] = {**passage, 'quote': quote}
            used += len(quote)
        result.append({'id': item['id'], 'chapter': item['chapter'],
                       'passages': [selected[i] for i in sorted(selected)]})
    return result


def resolve_references(draft, references):
    """只解析本次程序提供的引文编号，不修改模型结论或修补引文。"""
    claims = []
    for claim in draft['claims']:
        citations = []
        for citation in claim['citations']:
            reference = references.get((citation['chunk_id'], citation['quote_id']))
            if reference is None:
                raise ModelOutputError('模型引用了本次原文中不存在的引文编号')
            quote, position = reference
            citations.append({'chunk_id': citation['chunk_id'], 'quote': quote, '_relative_start': position})
        claims.append({**claim, 'citations': citations})
    return {**draft, 'claims': claims}


def expand_reference_context(draft, chosen, entities, max_chars=800):
    """只扩大同段中的连续原文区间，补齐姓名指代；不改写字符或认定事件成立。"""
    for claim in draft['claims']:
        names = list(dict.fromkeys([*[n for n in entities if n in claim['text']], *claim.get('mentions', [])]))
        for citation in claim['citations']:
            chunk = chosen[citation['chunk_id']]
            left = citation['_relative_start']
            right = left + len(citation['quote'])
            for name in names:
                if name in chunk.text[left:right] or not name or name not in chunk.text:
                    continue
                positions = [m.start() for m in re.finditer(re.escape(name), chunk.text)]
                nearest = min(positions, key=lambda p: min(abs(p - left), abs(p - right)))
                start, end = min(left, nearest), max(right, nearest + len(name))
                if end - start > max_chars:
                    continue
                line_start = chunk.text.rfind('\n', 0, start) + 1
                line_end = chunk.text.find('\n', end)
                line_end = len(chunk.text) if line_end < 0 else line_end + 1
                if line_end - line_start <= max_chars:
                    start, end = line_start, line_end
                left, right = start, end
            citation['quote'] = chunk.text[left:right]
            citation['_relative_start'] = left
    return draft


class ReadingAgent:
    def __init__(self, corpus: Corpus, backend: Backend | None = None, *, max_context_chars: int = 16000,
                 top_k: int = 6, search_rounds: int = 2, retriever=None, verifier: Backend | None = None):
        if max_context_chars < 1000 or top_k < 1 or not 1 <= search_rounds <= 3:
            raise ValueError("上下文预算至少 1000，top_k 至少 1，检索轮数须为 1 到 3")
        self.corpus = corpus
        self.backend = backend
        self.max_context_chars = max_context_chars
        self.top_k = top_k
        self.search_rounds = search_rounds
        self.retriever = retriever
        self.verifier = verifier or backend

    async def ask(self, question: str, *, sources: list[str] | None = None, extractive: bool = False, history=None,
                  evidence_ids=None, required_entities=None) -> dict:
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("问题须为 1 到 4000 字的非空文本")
        if not extractive and self.backend is None:
            raise ValueError("问答需要模型后端；可使用 --extractive 只检索原文")
        self.corpus.begin_read()
        # history 参数仅为旧调用兼容保留；任何历史问题和回答均不进入本次处理。
        context = {}
        source_ids = self.corpus.select_sources(sources)
        stats = self.corpus.stats(source_ids)
        if self.retriever is not None and not extractive and evidence_ids is None:
            await self.retriever.prepare(source_ids)
        chosen: dict[str, Chunk] = {}
        queries: list[str] = []
        entities = [name for name in (required_entities or []) if isinstance(name, str) and name in question]
        used = 0
        active_budget = self.max_context_chars
        context_trace = []

        def add(chunk: Chunk):
            nonlocal used
            if chunk.id not in chosen and used + len(chunk.text) <= active_budget:
                chosen[chunk.id] = chunk
                used += len(chunk.text)

        full_context = stats["context_chars"] <= self.max_context_chars and not extractive and evidence_ids is None
        if evidence_ids is not None:
            for chunk_id in dict.fromkeys(evidence_ids):
                chunk = self.corpus.get_chunk(chunk_id)
                if chunk.source_id not in source_ids:
                    raise ValueError('复用的原文不属于本次书籍')
                add(chunk)
            context_trace.append({'round': 1, 'selected_chunk_ids': list(chosen), 'context_chars': used})
        elif full_context:
            for chunk in self.corpus.all_chunks(source_ids):
                add(chunk)
        else:
            # 即使模型检索规划失败，也始终保留原问题的直接检索。
            pending = [question]
            for round_index in range(1 if extractive else self.search_rounds):
                # 为后续问题和反证保留上下文，避免第一轮耗尽全部预算。
                active_budget = self.max_context_chars if extractive else max(2000,
                    self.max_context_chars * (round_index + 1) // self.search_rounds)
                active_budget = min(active_budget, self.max_context_chars)
                if not extractive:
                    try:
                        strategies = getattr(self.backend, 'query_strategies', False)
                        plan = await request_json(self.backend, (prompts.QUERY_PLAN if strategies else prompts.PLAN), {
                            **context,
                            "question": question, "previous_queries": queries,
                            "evidence": [c.to_dict() for c in chosen.values()],
                        }, schema=schemas.QUERY_PLAN if strategies else schemas.PLAN)
                        if strategies:
                            # 只接收原问题逐字出现的名称，防止规划器引入未知人物。
                            for entity in plan['entities']:
                                name = entity['name']
                                if (entity['kind'] != 'other' and 1 < len(name) <= 32
                                        and name in question and name not in entities):
                                    entities.append(name)
                            # 改写动作时保留原问题的专名，不能用“主角”等泛称替换检索主体。
                            anchors = [e['name'] for e in plan['entities'] if e['kind'] in {'person', 'organization'}
                                       and e['name'] in entities][:2] or entities[:2]
                            # 原词及放宽查询只能取问题现有词语；新增同义动作另经语言核对。
                            literal_plan = {**plan, 'literal': [w for w in plan['literal'] if w in question],
                                            'broader': [w for w in plan['broader'] if w in question]}
                            extra = [' '.join(list(dict.fromkeys([*anchors, *literal_plan[key]])))
                                     for key in ('literal', 'paraphrase', 'broader')]
                        else:
                            extra = plan.get('queries')
                        if isinstance(extra, list):
                            if getattr(self.backend, 'query_review', False):
                                # 核对只读问题和关键词，不展示原文，防止已读噪声诱导猜测答案。
                                review = await request_json(self.backend, prompts.QUERY_REVIEW,
                                    {'question': question, 'queries': extra[:3]}, schema=schemas.QUERY_REVIEW)
                                indices = review['accepted_indices']
                                if len(indices) != len(set(indices)) or any(not 0 <= i < len(extra[:3]) for i in indices):
                                    raise ModelOutputError('查询核对包含未知或重复编号')
                                extra = [extra[i] for i in indices]
                                if strategies:
                                    actions = [' '.join([*anchors, action]) for action in review['action_terms'][:2]]
                                    extra = [*extra[:1], *actions, *extra[1:]]
                            pending.extend(q for q in extra[:3]
                                           if isinstance(q, str) and 0 < len(q.strip()) <= 100)
                    except ModelOutputError:
                        emit("检索规划格式无效，保留原问题检索", level=logging.WARNING, round=round_index + 1)
                hit_lists = []
                current_queries = [query for query in dict.fromkeys(pending) if query not in queries]
                queries.extend(current_queries)
                if self.retriever is not None and not extractive and hasattr(self.retriever, 'search_many'):
                    if hasattr(self.retriever, 'begin_round'):
                        self.retriever.begin_round(round_index + 1, self.search_rounds)
                    # 事件选择器多看候选概览，全文精读仍受同一预算约束；不追加重排预算。
                    shortlist = max(self.top_k, 16) if (round_index + 1 == self.search_rounds
                        and getattr(self.backend, 'evidence_selection', False)) else self.top_k
                    hit_lists = await self.retriever.search_many(question, current_queries, source_ids, limit=shortlist)
                    # 所有轮次的召回一起重新选取，前一轮无关片段不能永久占住上下文。
                    chosen.clear()
                    used = 0
                else:
                    for query in current_queries:
                        hits = (await self.retriever.search(query, source_ids, limit=self.top_k)
                                if self.retriever is not None and not extractive else
                                self.corpus.search(query, source_ids, limit=self.top_k))
                        hit_lists.append(hits)
                # 最后一轮先展示各查询的全部入围候选，再按事件依据选择精读原文。
                # 小重排模型只判断相关性，不能决定哪些片段足以证明答案。
                primary: list[Chunk] = []
                if (not extractive and round_index + 1 == self.search_rounds
                        and getattr(self.backend, 'evidence_selection', False)):
                    available = {c.id: c for hits in hit_lists for c in hits}
                    # 事件前后常分在相邻块；把邻块加入可选概览，避免噪声先耗尽精读预算。
                    for chunk in list(available.values()):
                        for neighbor in self.corpus.neighbors(chunk):
                            available.setdefault(neighbor.id, neighbor)
                    candidates = list(available.values())
                    if candidates:
                        try:
                            from .api_calls import notify
                            notify('正在从候选原文定位事件依据')
                            selection = await request_json(self.backend, prompts.SELECT_EVIDENCE, {
                                'question': question,
                                'evidence': evidence_overview(candidates, question, active_budget),
                            }, schema=schemas.SELECT_EVIDENCE)
                            available = {c.id: c for c in candidates}
                            ids = selection['chunk_ids']
                            if len(ids) != len(set(ids)) or any(cid not in available for cid in ids):
                                raise ModelOutputError('证据选择包含未知或重复的原文段落')
                            for cid in ids:
                                add(available[cid])
                                primary.append(available[cid])
                        except ModelOutputError:
                            emit('原文证据选择无效，保留检索排序', level=logging.WARNING)
                # 各查询轮流提供候选段落，确保子问题也有获得证据的机会。
                for rank in range(self.top_k):
                    for hits in hit_lists:
                        if rank < len(hits):
                            add(hits[rank])
                            primary.append(hits[rank])
                # 保留相邻段落，帮助核对台词归属、前后条件和矛盾。
                for chunk in primary:
                    for neighbor in self.corpus.neighbors(chunk):
                        add(neighbor)
                context_trace.append({'round': round_index + 1, 'selected_chunk_ids': list(chosen), 'context_chars': used})
                pending = []

        coverage = {
            "mode": "full_context" if full_context else "retrieved",
            "sources": [s for s in self.corpus.sources() if s["id"] in source_ids],
            "total_chunks": stats["chunks"], "selected_chunks": len(chosen),
            "context_chars": used, "queries": queries, "required_entities": entities,
            "reused_evidence": evidence_ids is not None,
            "retrieval": "hybrid" if self.retriever is not None and not extractive else "lexical",
            "context_trace": context_trace,
        }
        if self.retriever is not None and hasattr(self.retriever, 'rerank_stats'):
            coverage['rerank'] = dict(self.retriever.rerank_stats)
        if self.retriever is not None and hasattr(self.retriever, 'retrieval_trace'):
            coverage['retrieval_trace'] = self.retriever.retrieval_trace
        result = {"question": question, "status": "not_found", "claims": [], "coverage": coverage,
                  "notice": "在本次提供给模型的原文范围内，没有找到足以回答问题的依据。"}
        emit("原文上下文已选定", mode=coverage["mode"], retrieval=coverage["retrieval"],
             selected_chunks=len(chosen), total_chunks=stats["chunks"], context_chars=used)
        if extractive:
            result.update(status="extractive", evidence=[c.to_dict() for c in chosen.values()],
                          notice="以下仅为关键词匹配的原文段落，未生成或验证问题的答案。")
            return result
        if not chosen:
            return result
        # 评分缓存、候选编号和调试轨迹只用于管理页面；不能作为模型的原文证据。
        model_scope = {key: coverage[key] for key in ('mode', 'total_chunks', 'selected_chunks')}
        payload = {**context, "question": question, "coverage": model_scope,
                   "evidence": [c.to_dict() for c in chosen.values()]}
        try:
            handle_mode = getattr(self.backend, 'citation_handles', False)
            generation = dict(payload)
            generation['required_entities'] = entities
            references = {}
            if handle_mode:
                generation['evidence'], references = quote_catalog(chosen)
            citation_attempts = min(2, max(1, getattr(self.backend, 'citation_attempts', 1)))
            mention_mode = handle_mode and getattr(self.backend, 'entity_mentions', False)
            answer_prompt = prompts.ANSWER_WITH_MENTIONS if mention_mode else prompts.ANSWER_WITH_HANDLES if handle_mode else prompts.ANSWER
            answer_schema = schemas.ANSWER_WITH_MENTIONS if mention_mode else schemas.ANSWER_WITH_HANDLES if handle_mode else schemas.ANSWER
            role_cache = {}
            for attempt in range(citation_attempts):
                draft = await request_json(self.backend, answer_prompt, generation, schema=answer_schema)
                try:
                    if draft['claims'] and getattr(self.backend, 'name_extraction', False):
                        # 独立识别结论中的专名，不能依赖回答模型自报清单而漏检地点。
                        extracted = await request_json(self.verifier, prompts.CLAIM_MENTIONS,
                            {'claims': [{'id': c['id'], 'text': c['text']} for c in draft['claims']]},
                            schema=schemas.CLAIM_MENTIONS)
                        names = {item['claim_id']: item['names'] for item in extracted['claims']}
                        if len(names) != len(extracted['claims']) or set(names) != {c['id'] for c in draft['claims']}:
                            raise ModelOutputError('专名识别没有完整对应本次结论')
                        for claim in draft['claims']:
                            literal = [n for n in names[claim['id']] if n.strip() and n in claim['text']]
                            claim['mentions'] = list(dict.fromkeys([*claim.get('mentions', []), *literal]))
                    if handle_mode:
                        draft = resolve_references(draft, references)
                        draft = expand_reference_context(draft, chosen, entities)
                    claims = self._validate_draft(draft, chosen)
                    self._validate_entity_quotes(claims, entities)
                except ModelOutputError as exc:
                    if attempt + 1 >= citation_attempts:
                        raise
                    emit('引文选择无效，使用同一原文重新生成', level=logging.WARNING, attempt=attempt + 2)
                    generation['validation_feedback'] = {
                        'reason': str(exc), **getattr(exc, 'diagnostics', {}),
                        'candidate_claims': [{'id': c['id'], 'text': c['text'],
                            'citation_chunk_ids': [ref['chunk_id'] for ref in c['citations']]} for c in draft['claims']],
                        'instruction': '只用本次原文重新选择引用。缺少的专名可补引直接支持该事实的原文，或删除没有依据的额外修饰，保留能回答问题的核心事实。不得为覆盖姓名而引用无关事件。复合表达拆成独立专名，时期或类别不作为专名；不添加地点层级。核心事件确实缺少证据才说明不明确。',
                    }
                    continue
                if not claims:
                    if draft['status'] == 'unclear':
                        result.update(status='unclear', notice='现有原文依据不足或信息不明确，无法作出可靠结论。')
                    return result
                generated_count = len(claims)
                feedback = None
                if getattr(self.backend, 'citation_review', False):
                    claims, feedback = await self._review_citations(claims, question, role_cache, coverage)
                if feedback and attempt + 1 < citation_attempts:
                    # 只重选当前原文中的引文；不执行检索，也不把其他问答作为事实。
                    generation['validation_feedback'] = {**feedback,
                        'candidate_claims': [{'id': c['id'], 'text': c['text'],
                            'citation_chunk_ids': [ref['chunk_id'] for ref in c['citations']]} for c in draft['claims']],
                        'instruction': '按核对结果重选本次原文引用，只回答问题所必需的一条或数条最小事实。每个事实的主体、动作、地点和时期须由它自己的引文支持。可删去无依据细节，禁止只换说法掩盖证据缺失。'}
                    emit('事件引文不完整，使用同一原文重新生成', level=logging.WARNING, attempt=attempt + 2)
                    continue
                if not claims:
                    result.update(status='unclear', validation_error=feedback['reason'], validation_detail=feedback,
                        notice='候选结论未通过引文核对，不能据此回答。下方保留部分检索原文供查看。',
                        evidence=[c.to_dict() for c in list(chosen.values())[:3]])
                    return result
                break
            verification = await request_json(self.verifier, prompts.VERIFY, {**payload, "claims": claims}, schema=schemas.VERIFY)
            accepted, fully_answered = self._validate_verification(verification, claims)
        except ModelOutputError as exc:
            emit("回答校验未通过", level=logging.WARNING, error_type=type(exc).__name__)
            # 拒绝的模型结论和未复核草稿不能作为正式回答输出。
            result.update(status="unclear", notice="模型回答或引文未通过原文校验，这不代表原文没有答案。下方提供本次检索到的原文供核对。",
                          validation_error=str(exc), validation_detail=getattr(exc, 'diagnostics', {}),
                          evidence=[c.to_dict() for c in list(chosen.values())[:3]])
            return result
        if not accepted:
            emit("候选结论全部被复核拒绝", level=logging.WARNING, rejected_claims=len(claims))
            result.update(status="unclear", notice="候选结论未通过原文核对，现有证据不足，无法可靠回答。")
            return result
        complete = draft["status"] == "answered" and fully_answered and len(accepted) == generated_count
        emit("回答复核完成", accepted_claims=len(accepted), rejected_claims=generated_count - len(accepted), fully_answered=complete)
        result.update(status="answered" if complete else "unclear", claims=accepted,
                      notice="以下结论附有原文依据。" if complete else
                      "仅列出通过核对的内容；问题仍有未明确或缺少依据的部分。")
        return result

    async def _review_citations(self, claims, question, role_cache, coverage):
        """复核仅看当前精确引文；返回缺失项供同次回答补选，失败草稿不对外展示。"""
        original_count = len(claims)
        quote_evidence = [{'claim_id': claim['id'], 'quotes': [c['quote'] for c in claim['citations']]}
                          for claim in claims]
        role_rejected = False
        if getattr(self.backend, 'source_roles', False) and any(
                re.search(r'在|地点|位置|地方|\b(?:in|at|where)\b', c['text'], re.I) for c in claims):
            source_quotes = list(dict.fromkeys(q for item in quote_evidence for q in item['quotes']))
            roles = {'locations': []}
            for index, quote in enumerate(source_quotes):
                if quote not in role_cache:
                    role_cache[quote] = await request_json(self.verifier, prompts.SINGLE_SOURCE_ROLES,
                        {'quote': quote}, schema=schemas.SINGLE_SOURCE_ROLES)
                literal_roles = [item for item in role_cache[quote]['locations']
                                 if item['place'].strip() and item['place'] in quote]
                if len(literal_roles) != len(role_cache[quote]['locations']):
                    # 辅助语言识别不能引入训练记忆中的全称；丢弃非原文项后仍执行完整事实复核。
                    emit('语义角色识别的非原文名称已忽略', level=logging.WARNING,
                         discarded_items=len(role_cache[quote]['locations']) - len(literal_roles))
                roles['locations'].extend({**item, 'quote_index': index, 'quote': item['place']}
                                          for item in literal_roles)
            claims = self._guard_location_roles(claims, roles, source_quotes)
            role_rejected = len(claims) != original_count
            if not claims:
                return [], {'reason': '引用表达的是地域身份或来源，不能证明所问事件的发生地点。'}
            quote_evidence = [item for item in quote_evidence if item['claim_id'] in {c['id'] for c in claims}]
        proof_mode = getattr(self.backend, 'proof_review', False)
        review = await request_json(self.verifier, prompts.PROOF_VERIFY if proof_mode else prompts.CITATION_VERIFY, {
            'question': question, 'claims': [{'id': c['id'], 'text': c['text']} for c in claims],
            'evidence': quote_evidence,
            # 只给范围元数据，不能让复核从其他未引用段落补出事实。
            'coverage': {key: coverage[key] for key in ('mode', 'total_chunks', 'selected_chunks')},
        }, schema=schemas.PROOF_VERIFY if proof_mode else schemas.VERIFY)
        checked = self._validate_proof(review, claims) if proof_mode else review
        accepted, _ = self._validate_verification(checked, claims)
        if len(accepted) == original_count:
            return accepted, None
        rejected_ids = {c['id'] for c in claims} - {c['id'] for c in accepted}
        details = [{'claim_id': v['claim_id'], 'checks': v.get('checks', {})}
                   for v in review['verdicts'] if v['claim_id'] in rejected_ids]
        return accepted, {'reason': '所选引文没有完整支持候选结论。',
                          'failed_checks': details, 'location_role_conflict': role_rejected}

    def _validate_draft(self, draft: dict, chosen: dict[str, Chunk]) -> list[dict]:
        if draft.get("status") not in {"answered", "unclear", "not_found"}:
            raise ModelOutputError("回答状态不合法")
        raw_claims = draft.get("claims")
        if not isinstance(raw_claims, list) or len(raw_claims) > 12:
            raise ModelOutputError("结论必须为最多 12 条的数组")
        if (draft["status"] == "not_found" and raw_claims) or (draft["status"] == "answered" and not raw_claims):
            raise ModelOutputError("回答状态与结论不一致")
        result, ids = [], set()
        for raw in raw_claims:
            if not isinstance(raw, dict):
                raise ModelOutputError("结论格式无效")
            claim_id, text, citations = raw.get("id"), raw.get("text"), raw.get("citations")
            if not isinstance(claim_id, str) or not claim_id.strip() or len(claim_id) > 50 or claim_id in ids:
                raise ModelOutputError("结论 ID 必须非空且不重复")
            if not isinstance(text, str) or not 0 < len(text.strip()) <= 1500:
                raise ModelOutputError("结论必须为 1 到 1500 字的非空文本")
            if not isinstance(citations, list) or not 1 <= len(citations) <= 8:
                raise ModelOutputError("每条结论必须附有 1 到 8 条原文引用")
            validated = []
            for citation in citations:
                if not isinstance(citation, dict) or not isinstance(citation.get("chunk_id"), str):
                    raise ModelOutputError("引用格式无效")
                chunk = chosen.get(citation["chunk_id"])
                if chunk is None:
                    raise ModelOutputError("引用的段落不在本次提供的原文范围内")
                quote = citation.get('quote')
                if not isinstance(quote, str) or not quote.strip() or quote not in chunk.text:
                    raise ModelOutputError('模型引文不是对应段落中逐字一致的连续原文')
                validated.append(self.corpus.citation(chunk, quote, relative_start=citation.get('_relative_start')))
            # 专名完整覆盖是必要条件，不能从标题、未引用背景或记忆补上地点全称。
            mentions = raw.get('mentions', [])
            quotes = ''.join(c['quote'] for c in validated)
            missing = [name for name in mentions if not name.strip() or name not in text or name not in quotes]
            if missing:
                raise ModelOutputError('结论中的专名没有被所选引文完整覆盖，需要补充原文引用。',
                                       diagnostics={'missing_mentions': missing, 'claim_id': claim_id})
            ids.add(claim_id)
            result.append({"id": claim_id, "text": text, "citations": validated})
        return result

    @staticmethod
    def _validate_entity_quotes(claims, entities):
        """名称对齐是必要检查；通过仍须独立核对事件和语义，不能推导事实。"""
        for claim in claims:
            required = [name for name in entities if name in claim['text']]
            if not required and len(entities) == 1:
                required = entities
            quotes = ''.join(c['quote'] for c in claim['citations'])
            missing = [name for name in required if name not in quotes]
            if missing:
                raise ModelOutputError('引文没有明确覆盖结论的主体，需要补充原文指代依据。',
                                       diagnostics={'missing_mentions': missing})

    @staticmethod
    def _guard_location_roles(claims, roles, source_quotes):
        """原文仅有地域身份或来源时，拒绝把它改成实际发生位置。"""
        for item in roles['locations']:
            index = item['quote_index']
            if (not 0 <= index < len(source_quotes) or not item['quote'].strip()
                    or item['quote'] not in source_quotes[index] or not item['place'].strip()
                    or item['place'] not in item['quote']):
                raise ModelOutputError('语义角色依据不是对应引文中逐字一致的原文')
        accepted = []
        for claim in claims:
            quotes = {c['quote'] for c in claim['citations']}
            places = {}
            for item in roles['locations']:
                if source_quotes[item['quote_index']] in quotes:
                    places.setdefault(item['place'], set()).add(item['role'])
            invalid = False
            for place, kinds in places.items():
                if kinds <= {'affiliation', 'origin'}:
                    # 这里只拦截已识别的身份／来源偷换，不从关键词自行推导事实。
                    locative = re.search(r'(?:在|(?<!属)于)\s*' + re.escape(place), claim['text'])
                    named_location = place in claim['text'] and re.search(r'地点|位置|地方|所在', claim['text'])
                    english_location = re.search(r'\b(?:in|at)\s+' + re.escape(place), claim['text'], re.I)
                    if locative or named_location or english_location:
                        invalid = True
                        break
            if not invalid:
                accepted.append(claim)
        return accepted

    @staticmethod
    def _validate_proof(review, claims):
        """分项证明必须全部成立；缺失或冲突不能被总判断覆盖。"""
        by_id = {claim['id']: claim for claim in claims}
        verdicts = []
        for verdict in review['verdicts']:
            claim = by_id.get(verdict['claim_id'])
            if claim is None:
                raise ModelOutputError('证明核对包含未知结论')
            supported = verdict['supported']
            for field, check in verdict['checks'].items():
                if check['status'] in {'missing', 'conflicting'}:
                    supported = False
                if check['status'] == 'supported' and not check['reason'].strip():
                    supported = False
                # 主体和所问关系不能被标成“不适用”而绕过事件核对。
                if check['status'] == 'not_applicable' and field in {'subject', 'relation'}:
                    supported = False
            verdicts.append({'claim_id': claim['id'], 'supported': supported})
        return {'verdicts': verdicts, 'question_fully_answered': review['question_fully_answered']}

    @staticmethod
    def _validate_verification(verification: dict, claims: list[dict]):
        verdicts = verification.get("verdicts")
        fully_answered = verification.get("question_fully_answered")
        if not isinstance(verdicts, list) or type(fully_answered) is not bool:
            raise ModelOutputError("核对结果格式无效")
        expected = {claim["id"] for claim in claims}
        approved: dict[str, bool] = {}
        for verdict in verdicts:
            if not isinstance(verdict, dict):
                raise ModelOutputError("核对结果格式无效")
            claim_id = verdict.get("claim_id")
            if not isinstance(claim_id, str) or claim_id not in expected or claim_id in approved:
                raise ModelOutputError("核对结果包含未知或重复的结论")
            if type(verdict.get("supported")) is not bool:
                raise ModelOutputError("核对结果必须使用布尔值")
            approved[claim_id] = verdict["supported"]
        if set(approved) != expected:
            raise ModelOutputError("部分结论未经核对")
        return [claim for claim in claims if approved[claim["id"]]], fully_answered
