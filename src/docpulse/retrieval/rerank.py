"""
Cross-Encoder 重排（bge-reranker-base）。

Bi-Encoder vs Cross-Encoder（检索两阶段常见组合）：
  - Bi-Encoder（本项目的 embedding）：query 与 doc 分别编码再算相似度，速度快，适合百万级召回
  - Cross-Encoder：把 [query, passage] 拼在一起过 Transformer，交互更充分，精度高但慢

因此流水线为：Bi-Encoder/BM25 粗召回大量候选 → Cross-Encoder 对 top-N 精排。

在 RRF 融合后的候选 child 上，用 (query, passage) 对打分，
取 top_k 再映射回 parent 上下文（见 engine.retrieve_contexts）。
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Tuple

from docpulse.config import load_config
from docpulse.embeddings import resolve_device


@lru_cache(maxsize=1)
def get_reranker(model_name: Optional[str] = None, device: Optional[str] = None):
    """懒加载 sentence_transformers.CrossEncoder。"""
    cfg = load_config()
    name = model_name or cfg.rerank.model
    dev = resolve_device(device or cfg.rerank.device)
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, device=dev)


def rerank_pairs(
    query: str,
    pairs: List[Tuple[str, str]],
    *,
    top_k: int = 8,
) -> List[Tuple[str, float]]:
    """
    对候选 passage 重排。

    pairs: (chunk_id, passage_text)，返回 (chunk_id, score) 按分数降序 top_k。
    """
    if not pairs:
        return []
    model = get_reranker()
    # CrossEncoder 输入为句子对列表，模型内部做 query-doc 交叉注意力
    inputs = [[query, text] for _, text in pairs]
    scores = model.predict(inputs)  # 每个 passage 一个相关性标量
    ranked = sorted(
        zip([cid for cid, _ in pairs], scores),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    return ranked[:top_k]
