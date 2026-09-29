"""
问句扩展（当前 M2：规则；M4 计划 LLM Multi-Query）。

从用户问题中提取：
  - 反引号内的标识符 `APIRouter`
  - CamelCase / snake_case 术语
  - 英文关键词串

扩展问句会分别走 dense/sparse，再经 RRF 合并，提高召回覆盖面。
"""

from __future__ import annotations

import re
from typing import List

from docpulse.config import AppConfig, load_config

BACKTICK_RE = re.compile(r"`([^`]+)`")
CAMEL_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]+\b")
SNAKE_RE = re.compile(r"\b[a-z]+_[a-z0-9_]+\b")


def rule_expand_queries(question: str, cfg: AppConfig | None = None) -> List[str]:
    """
    规则扩展问句（M4 再叠加 LLM）。

    返回列表：原问 + 最多 rule_expansions 条扩展（去重、保序）。
    """
    cfg = cfg or load_config()
    n = max(0, cfg.query_rewrite.rule_expansions)
    if n == 0:
        return [question]

    extras: List[str] = []
    seen = {question.strip().lower()}

    def add(q: str) -> None:
        q = q.strip()
        if not q or q.lower() in seen:
            return
        seen.add(q.lower())
        extras.append(q)

    for m in BACKTICK_RE.findall(question):
        add(m)

    for m in CAMEL_RE.findall(question):
        add(m)
        # CamelCase 拆词：FastAPI → "Fast API"，BM25 按词匹配时更容易命中文档中的写法
        add(re.sub(r"([a-z])([A-Z])", r"\1 \2", m))

    for m in SNAKE_RE.findall(question):
        add(m.replace("_", " "))

    # 英文短问：抽取关键词子串作为「宽松查询」，提高召回（Multi-Query 的轻量版）
    words = [w for w in re.findall(r"[A-Za-z]{3,}", question)]
    if len(words) >= 2:
        add(" ".join(words[:6]))

    queries = [question]
    queries.extend(extras[:n])
    return queries
