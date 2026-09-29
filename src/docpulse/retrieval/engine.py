"""
RAG 检索编排核心（RAGEngine）。

在线问答流水线：
  1. rule_expand_queries — 多路问句
  2. dense (Chroma) + sparse (BM25) — 每问各召回 K 条
  3. rrf_fuse — 合并为候选 child_id
  4. rerank_pairs — Cross-Encoder 精排
  5. 按 parent_id 去重，取 Parent 全文作 LLM 上下文
  6. generate_answer / generate_answer_stream — DeepSeek
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from docpulse.config import AppConfig, load_config, project_root
from docpulse.embeddings import embed_texts
from docpulse.generation.answer import generate_answer, generate_answer_stream
from docpulse.ingest.indexer import TOKEN_RE
from docpulse.ingest.state import bm25_path, chunks_path, load_chunks
from docpulse.models import ChunkRecord
from docpulse.retrieval.hybrid import rrf_fuse
from docpulse.retrieval.query_rewrite import rule_expand_queries
from docpulse.retrieval.rerank import rerank_pairs
from docpulse.vectorstore import get_vector_store


@dataclass
class SourceRef:
    """展示给用户的引用元数据（一条 child 命中对应一个 parent 来源）。"""

    file_path: str
    heading_path: str
    section_title: str
    chunk_id: str
    parent_id: str
    score: float


@dataclass
class ContextSnippet:
    """送入 LLM 的检索片段（Parent 级上下文）。"""

    file_path: str
    heading_path: str
    section_title: str
    text: str
    score: float


@dataclass
class RetrievalResult:
    """非流式 ask() 的完整返回。"""

    answer: str
    sources: List[SourceRef]
    queries_used: List[str]
    snippets: List[ContextSnippet]


class RAGEngine:
    """封装配置、状态路径与检索/生成方法。"""

    def __init__(self, cfg: Optional[AppConfig] = None):
        self.cfg = cfg or load_config()
        self.root = project_root()
        self.state_root = self.cfg.state_path(self.root)
        # 按 project 缓存 parent_id/chunk_id → ChunkRecord
        self._parents_cache: Dict[str, Dict[str, ChunkRecord]] = {}

    def _parents_for(self, project: str) -> Dict[str, ChunkRecord]:
        """从 chunks.jsonl 加载所有 parent，键含 parent_id 与 chunk_id。"""
        if project not in self._parents_cache:
            all_chunks = load_chunks(chunks_path(self.state_root, project))
            parents: Dict[str, ChunkRecord] = {}
            for c in all_chunks:
                if c.level == "parent":
                    parents[c.parent_id] = c
                    parents[c.chunk_id] = c
            self._parents_cache[project] = parents
        return self._parents_cache[project]

    def _load_bm25(self, project: str):
        path = bm25_path(self.state_root, project)
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 索引不存在: {path}\n请先运行: python -m docpulse index"
            )
        with path.open("rb") as f:
            data = pickle.load(f)
        return data["chunk_ids"], data["bm25"]

    def _dense_search(
        self, query: str, project: str, limit: int
    ) -> List[Tuple[str, float]]:
        """
        稠密向量检索（语义相似度）。

        流程：问句 → SentenceTransformer 编码（L2 归一化）
             → Chroma HNSW 近似最近邻（cosine）
        """
        vec = embed_texts([query])[0]
        return get_vector_store().query_dense(vec, project, limit)

    def _sparse_search(
        self, query: str, project: str, limit: int
    ) -> List[Tuple[str, float]]:
        """
        稀疏检索（BM25 关键词匹配）。

        get_scores 对语料中每个文档返回一个分数；按分降序取 top limit。
        过滤 score > 0，避免无匹配词时返回噪声。
        """
        chunk_ids, bm25 = self._load_bm25(project)
        tokens = [t.lower() for t in TOKEN_RE.findall(query)]
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        ranked = sorted(
            zip(chunk_ids, scores), key=lambda x: float(x[1]), reverse=True
        )[:limit]
        return [(cid, float(s)) for cid, s in ranked if s > 0]

    def _payload_map(self, chunk_ids: List[str]) -> Dict[str, dict]:
        """从 Chroma 批量取 metadata + document 文本。"""
        return get_vector_store().get_payloads(chunk_ids)

    def retrieve_contexts(
        self,
        question: str,
        *,
        project: str = "fastapi",
        use_hybrid: bool = True,
        use_rerank: bool = True,
        use_rewrite: bool = True,
    ) -> Tuple[List[str], List[SourceRef], List[str], List[ContextSnippet]]:
        """
        仅检索，不调用 LLM。

        返回：
          context_blocks — 拼接给 LLM 的字符串列表
          sources / snippets — 元数据
          queries — 实际用于检索的问句列表
        """
        rc = self.cfg.retrieval

        # ── 步骤 1：问句扩展（Multi-Query）────────────────────────────
        queries = (
            rule_expand_queries(question, self.cfg)
            if use_rewrite
            else [question]
        )

        dense_lists: List[List[str]] = []
        sparse_lists: List[List[str]] = []

        # ── 步骤 2：每路问句分别做 dense + sparse 召回 ───────────────
        for q in queries:
            if use_hybrid:
                dense_lists.append(
                    [cid for cid, _ in self._dense_search(q, project, rc.dense_k)]
                )
                sparse_lists.append(
                    [cid for cid, _ in self._sparse_search(q, project, rc.sparse_k)]
                )
            else:
                # 仅 dense 时也走多 query（若开启 rewrite）
                dense_lists.append(
                    [cid for cid, _ in self._dense_search(q, project, rc.dense_k)]
                )

        # ── 步骤 3：RRF 融合多路排名 → 候选 child_id 列表 ───────────
        fused = rrf_fuse(
            dense_lists + (sparse_lists if use_hybrid else []),
            rrf_k=rc.rrf_k,
            top_n=rc.rrf_k,
        )
        candidate_ids = [cid for cid, _ in fused]

        # 从 Chroma 批量取 passage 文本与 metadata（parent_id 等）
        payloads = self._payload_map(candidate_ids)
        pairs: List[Tuple[str, str]] = []
        for cid in candidate_ids:
            p = payloads.get(cid)
            if p:
                pairs.append((cid, p.get("text", "")))

        # ── 步骤 4：Cross-Encoder 精排（用原始用户问句，不用扩展问句）──
        if use_rerank and pairs:
            reranked = rerank_pairs(question, pairs, top_k=rc.rerank_k)
            top_child_ids = [cid for cid, _ in reranked]
            score_map = {cid: sc for cid, sc in reranked}
        else:
            top_child_ids = candidate_ids[: rc.rerank_k]
            score_map = {cid: fused[i][1] for i, cid in enumerate(top_child_ids)}

        parents = self._parents_for(project)
        seen_parents: set = set()
        context_blocks: List[str] = []
        sources: List[SourceRef] = []
        snippets: List[ContextSnippet] = []

        # ── 步骤 5：Child → Parent 扩展，按 parent 去重 ─────────────
        # 多个 child 可能属于同一章节；只取 parent 全文一次，避免重复占用 LLM 上下文
        for cid in top_child_ids:
            p = payloads.get(cid, {})
            pid = p.get("parent_id", "")
            if pid in seen_parents:
                continue
            seen_parents.add(pid)
            parent = parents.get(pid)
            block = parent.text if parent else p.get("text", "")
            fp = p.get("file_path", "")
            hp = p.get("heading_path", "")
            sc = float(score_map.get(cid, 0.0))
            context_blocks.append(f"[{fp}] {hp}\n{block}")
            sources.append(
                SourceRef(
                    file_path=fp,
                    heading_path=hp,
                    section_title=p.get("section_title", ""),
                    chunk_id=cid,
                    parent_id=pid,
                    score=sc,
                )
            )
            snippets.append(
                ContextSnippet(
                    file_path=fp,
                    heading_path=hp,
                    section_title=p.get("section_title", ""),
                    text=block,
                    score=sc,
                )
            )
            if len(context_blocks) >= rc.final_parent_n:
                break

        return context_blocks, sources, queries, snippets

    def ask(
        self,
        question: str,
        *,
        project: str = "fastapi",
        use_hybrid: bool = True,
        use_rerank: bool = True,
        use_rewrite: bool = True,
    ) -> RetrievalResult:
        """检索 + 一次性生成完整答案。"""
        contexts, sources, queries, snippets = self.retrieve_contexts(
            question,
            project=project,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            use_rewrite=use_rewrite,
        )
        answer = generate_answer(question, contexts, self.cfg)
        return RetrievalResult(
            answer=answer,
            sources=sources,
            queries_used=queries,
            snippets=snippets,
        )

    def stream_answer(
        self,
        question: str,
        *,
        project: str = "fastapi",
        use_hybrid: bool = True,
        use_rerank: bool = True,
        use_rewrite: bool = True,
    ):
        """
        生成器协议（供 Web SSE 使用）：
          第一次 yield: (snippets, sources, queries)
          之后 yield: 每个 token 字符串
        """
        contexts, sources, queries, snippets = self.retrieve_contexts(
            question,
            project=project,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank,
            use_rewrite=use_rewrite,
        )
        yield snippets, sources, queries
        for token in generate_answer_stream(question, contexts, self.cfg):
            yield token


@lru_cache(maxsize=1)
def get_engine() -> RAGEngine:
    """进程内单例引擎（模型与 Chroma 客户端也多为单例）。"""
    return RAGEngine()
