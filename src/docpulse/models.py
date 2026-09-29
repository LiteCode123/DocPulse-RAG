"""
核心数据模型（入库与检索共用）。

ChunkRecord：一条可检索单元，level 为 parent 或 child。
  - 检索索引只写入 child；生成答案时回溯 parent 文本作为上下文。
FileFingerprint：单文件的 content_hash，用于 ingest 增量跳过未变文件。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """ISO 格式 UTC 时间戳，写入 updated_at 字段。"""
    return datetime.now(timezone.utc).isoformat()


class ChunkRecord(BaseModel):
    """
    单条可检索块（Child 或 Parent 元数据）。

    标识关系：
      doc_id     — 一个 Markdown 文件
      parent_id  — 一个章节（##/###）
      chunk_id   — 全局唯一，child/parent 各有一条
    """

    doc_id: str
    chunk_id: str
    parent_id: str
    project: str
    file_path: str  # 相对仓库根，如 docs/en/docs/tutorial/first-steps.md
    section_title: str
    heading_path: str  # 如 "Tutorial > First Steps"
    git_commit: str  # 入库时的 HEAD 或 "local"
    content_hash: str  # 源文件 SHA256，M3 增量索引用
    is_active: bool = True  # 软删除标记（M3）
    updated_at: str = Field(default_factory=utc_now_iso)
    level: Literal["parent", "child"]  # 索引与检索只用 child；parent 供上下文扩展
    text: str
    child_index: int = 0  # 同一 parent 下第几个 child，从 0 递增


class FileFingerprint(BaseModel):
    """文件级指纹，存于 data/state/<project>/file_fingerprints.json。"""

    file_path: str
    content_hash: str
    doc_id: str
    updated_at: str = Field(default_factory=utc_now_iso)
