"""
索引构建（index 命令，M2）。

输入：data/state/<project>/chunks.jsonl
输出：
  - Chroma collection 中仅 child 的向量 + metadata
  - bm25.pkl（chunk_id 列表 + BM25Okapi 对象）
"""

from __future__ import annotations

import pickle
import re
from typing import List, Optional

from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.progress import track

from docpulse.config import AppConfig, load_config, project_root
from docpulse.embeddings import embed_texts, get_embedding_model
from docpulse.ingest.state import bm25_path, chunks_path, load_chunks
from docpulse.models import ChunkRecord
from docpulse.vectorstore import get_vector_store

console = Console()
# 英文/数字/下划线分词，与 engine 稀疏检索一致
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> List[str]:
    """英文友好分词：字母数字下划线连续段。与 engine._sparse_search 保持一致。"""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def build_bm25(children: List[ChunkRecord], out_path) -> BM25Okapi:
    """
    用全部 child 文本建 BM25Okapi 稀疏索引。

    BM25（Best Matching 25）要点：
      - 基于词频 TF 与逆文档频率 IDF，擅长精确关键词匹配（如 API 名、参数名）
      - 与向量检索互补：向量擅长语义相近，BM25 擅长字面匹配
      - rank_bm25.BM25Okapi 在内存中对语料建倒排，get_scores(query_tokens) 返回每篇得分

    持久化时同时保存 chunk_ids 列表，保证分数下标与 ChunkRecord 一一对应。
    """
    corpus_tokens = [_tokenize(c.text) for c in children]
    bm25 = BM25Okapi(corpus_tokens)
    payload = {
        "chunk_ids": [c.chunk_id for c in children],
        "bm25": bm25,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    return bm25


def index_project(
    project_name: str,
    *,
    cfg: Optional[AppConfig] = None,
    recreate: bool = False,
) -> dict:
    """
    单项目索引。

    recreate=True：删除整个 Chroma collection 后重建（多项目共用 collection 时仅首个项目触发）。
    否则：只 delete 该 project 的向量再 upsert（M3 将改为按 doc 增量）。
    """
    cfg = cfg or load_config()
    root = project_root()
    state_root = cfg.state_path(root)
    all_chunks = load_chunks(chunks_path(state_root, project_name))
    if not all_chunks:
        raise FileNotFoundError(
            f"无 chunks 数据，请先 ingest: data/state/{project_name}/chunks.jsonl"
        )

    children = [c for c in all_chunks if c.level == "child" and c.is_active]
    parents = {c.chunk_id: c for c in all_chunks if c.level == "parent"}
    if not children:
        raise ValueError("没有可索引的 child chunks")

    model = get_embedding_model()
    vector_size = model.get_sentence_embedding_dimension()
    store = get_vector_store()

    if recreate:
        console.print("[yellow]重建本地向量库 collection[/yellow]")
        store.reset_collection()
    else:
        store.delete_project(project_name)

    batch_size = 64
    texts = [c.text for c in children]
    console.print(f"[bold]向量化[/bold] {len(children)} 条 child …")
    vectors = embed_texts(texts, batch_size=batch_size)

    console.print("[bold]写入 Chroma[/bold]（本地持久化，无需 Docker）…")
    for i in track(range(0, len(children), batch_size), description="Chroma upsert"):
        batch_c = children[i : i + batch_size]
        batch_v = vectors[i : i + batch_size]
        store.upsert_children(batch_c, batch_v, batch_size=batch_size)

    console.print("[bold]构建 BM25[/bold] …")
    build_bm25(children, bm25_path(state_root, project_name))

    return {
        "children": len(children),
        "parents": len(parents),
        "vector_size": vector_size,
        "vector_store": "chroma",
        "persist_dir": str(store.persist_dir),
        "collection": store.collection_name,
    }


def run_index(
    *,
    project_name: Optional[str] = None,
    recreate: bool = False,
    config_path: Optional[str] = None,
) -> None:
    """CLI index 入口。"""
    cfg = load_config(config_path)
    projects = [p.name for p in cfg.projects]
    if project_name:
        if project_name not in projects:
            raise SystemExit(f"未知项目: {project_name}")
        projects = [project_name]

    for name in projects:
        console.print(f"\n[bold cyan]索引 {name}[/bold cyan]")
        stats = index_project(name, cfg=cfg, recreate=recreate and name == projects[0])
        console.print(stats)
    console.print("\n[green]索引完成[/green] 向量数据目录:", cfg.vector_store.persist_dir)
