"""
稳定 ID 与文件指纹。

设计原则：
  - doc_id / chunk_id / parent_id 均由确定性哈希生成，同一内容重复 ingest 得到相同 ID
  - content_hash 用于判断文件是否变更，驱动增量 ingest（M3 将用于向量增量）
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_text(text: str) -> str:
    """对 UTF-8 文本做 SHA256。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """读取整个文件内容后哈希（与 ingest 时 content_hash 一致）。"""
    return sha256_text(path.read_text(encoding="utf-8", errors="replace"))


def make_doc_id(project: str, rel_path: str) -> str:
    """
    一个 Markdown 文件对应一个 doc_id（16 位十六进制前缀）。

    用确定性哈希而非自增 ID：重复 ingest 同一文件路径时 ID 不变，
    便于增量更新时按 doc_id 替换旧 chunks（见 pipeline.process_project）。
    """
    raw = f"{project}:{rel_path}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def make_chunk_id(doc_id: str, heading_path: str, span: str, level: str) -> str:
    """
    块级 ID。span 区分 parent 或 child:N。
    level 参与哈希，避免 parent/child 碰撞。
    """
    raw = f"{doc_id}|{heading_path}|{span}|{level}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def make_parent_id(doc_id: str, heading_path: str) -> str:
    """章节 Parent 的稳定 ID，child 的 parent_id 指向它。"""
    return make_chunk_id(doc_id, heading_path, "parent", "parent")
