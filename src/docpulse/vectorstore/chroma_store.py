"""
ChromaDB 本地持久化向量存储。

设计要点：
  - PersistentClient 数据目录 config.vector_store.persist_dir
  - 仅索引 level=child 的块；检索时 where 过滤 project + is_active
  - chunk_id 作为 Chroma 主键 id
  - cosine 空间：query 返回 distance，转为 1-distance 作相似度
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.api.models.Collection import Collection

from docpulse.config import AppConfig, load_config, project_root
from docpulse.models import ChunkRecord


def _metadata(c: ChunkRecord) -> dict:
    """Chroma 元数据仅支持标量类型（bool/int/float/str）。"""
    return {
        "doc_id": c.doc_id,
        "chunk_id": c.chunk_id,
        "parent_id": c.parent_id,
        "project": c.project,
        "file_path": c.file_path,
        "section_title": c.section_title,
        "heading_path": c.heading_path,
        "level": c.level,
        "is_active": bool(c.is_active),
    }


def _payload_from_get(meta: dict, document: Optional[str]) -> dict:
    """合并 metadata 与正文，供 engine 重排与组上下文。"""
    p = dict(meta)
    p["text"] = document or ""
    return p


class ChromaStore:
    """本地持久化向量库（ChromaDB，无需 Docker）。"""

    def __init__(self, cfg: Optional[AppConfig] = None):
        self.cfg = cfg or load_config()
        self.root = project_root()
        self.persist_dir = self.root / self.cfg.vector_store.persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection: Optional[Collection] = None

    @property
    def collection_name(self) -> str:
        return self.cfg.vector_store.collection

    @property
    def collection(self) -> Collection:
        """懒加载 collection，HNSW 使用 cosine 距离。"""
        if self._collection is None:
            # HNSW（Hierarchical Navigable Small World）：近似最近邻图索引
            # 比暴力全量比对快得多，适合本地十万级 chunk 规模
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def reset_collection(self) -> Collection:
        """--recreate 时删除整个 collection 再新建。"""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def delete_project(self, project_name: str) -> None:
        """按 metadata.project 删除该项目的所有向量（全量重建索引前）。"""
        try:
            self.collection.delete(where={"project": project_name})
        except Exception:
            pass

    def upsert_children(
        self,
        children: List[ChunkRecord],
        embeddings: List[List[float]],
        *,
        batch_size: int = 100,
    ) -> None:
        """批量 upsert：id=chunk_id，向量+正文+metadata。"""
        col = self.collection
        for i in range(0, len(children), batch_size):
            batch_c = children[i : i + batch_size]
            batch_e = embeddings[i : i + batch_size]
            col.upsert(
                ids=[c.chunk_id for c in batch_c],
                embeddings=batch_e,
                documents=[c.text for c in batch_c],
                metadatas=[_metadata(c) for c in batch_c],
            )

    def query_dense(
        self,
        query_vector: List[float],
        project: str,
        limit: int,
    ) -> List[Tuple[str, float]]:
        """
        向量近邻检索，仅 child 且 is_active。

        返回 (chunk_id, similarity_score)，score 越大越相似。
        """
        # where 过滤：只检索指定项目的 active child（parent 不进向量库）
        res = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where={
                "$and": [
                    {"project": {"$eq": project}},
                    {"level": {"$eq": "child"}},
                    {"is_active": {"$eq": True}},
                ]
            },
            include=["metadatas", "distances"],
        )
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        # Chroma 返回 cosine distance；相似度 ≈ 1 - distance（越大越相关）
        return [(cid, 1.0 - float(d)) for cid, d in zip(ids, dists)]

    def get_payloads(self, chunk_ids: List[str]) -> Dict[str, dict]:
        """按 id 批量 get，用于 RRF 后取 passage 文本与 parent_id。"""
        if not chunk_ids:
            return {}
        res = self.collection.get(
            ids=chunk_ids,
            include=["metadatas", "documents"],
        )
        out: Dict[str, dict] = {}
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        docs = res.get("documents") or []
        for cid, meta, doc in zip(ids, metas, docs):
            if meta:
                out[cid] = _payload_from_get(meta, doc)
        return out


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaStore:
    return ChromaStore()
