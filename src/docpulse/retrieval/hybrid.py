"""
Reciprocal Rank Fusion (RRF) — 多路检索结果融合。

将 dense、sparse、多 query 等多条排名列表合并为一个分数：
  score(d) += 1 / (k + rank(d))
k 越大，排名靠后的文档得分衰减越慢（更平滑）。
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def rrf_fuse(
    ranked_lists: List[List[str]],
    *,
    rrf_k: int = 60,
    top_n: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion（倒数排名融合）。

    输入：多个 chunk_id 排名列表（顺序即 rank 0,1,2...）
    输出：按 RRF 分降序的 (chunk_id, score) 列表，截断 top_n。

    公式：对每个文档 d，score(d) = Σ 1/(k + rank_i(d))
      - rank_i 是 d 在第 i 路检索中的名次（从 0 起）
      - k 为平滑常数（默认 60），越大则头部/尾部名次差距越小

    示例：某 chunk 在 dense 排第 0、在 BM25 排第 2，k=60
      score = 1/(60+1) + 1/(60+3) ≈ 0.0164 + 0.0159

    优点：不需对齐各路分数的量纲（向量相似度 vs BM25 分），只依赖排名。
    """
    scores: Dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking):
            # rank+1 避免除零；同一 doc 在多路出现则分数累加（多路共识加分）
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    return ordered[:top_n]
