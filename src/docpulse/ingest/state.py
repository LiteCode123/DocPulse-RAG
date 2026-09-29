"""
入库状态持久化（JSON / JSONL / pickle）。

目录布局 data/state/<project_name>/：
  file_fingerprints.json — 每文件 content_hash
  chunks.jsonl           — 全部 ChunkRecord（parent + child）
  last_commit.txt        — 最近一次 sync 的 git commit
  bm25.pkl               — M2 稀疏检索索引（index 命令生成）
"""

from __future__ import annotations

import json
from pathlib import Path

from docpulse.models import ChunkRecord, FileFingerprint


def project_state_dir(state_root: Path, project_name: str) -> Path:
    """确保项目状态目录存在并返回路径。"""
    d = state_root / project_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def fingerprints_path(state_root: Path, project_name: str) -> Path:
    return project_state_dir(state_root, project_name) / "file_fingerprints.json"


def chunks_path(state_root: Path, project_name: str) -> Path:
    return project_state_dir(state_root, project_name) / "chunks.jsonl"


def commit_path(state_root: Path, project_name: str) -> Path:
    return project_state_dir(state_root, project_name) / "last_commit.txt"


def bm25_path(state_root: Path, project_name: str) -> Path:
    return project_state_dir(state_root, project_name) / "bm25.pkl"


def load_chunks(path: Path) -> list[ChunkRecord]:
    """从 JSONL 逐行加载；文件不存在返回空列表。"""
    if not path.exists():
        return []
    records: list[ChunkRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(ChunkRecord.model_validate(json.loads(line)))
    return records


def load_fingerprints(path: Path) -> dict[str, FileFingerprint]:
    """键为相对文件路径 rel_path。"""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: FileFingerprint.model_validate(v) for k, v in data.items()}


def save_fingerprints(path: Path, fps: dict[str, FileFingerprint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v.model_dump() for k, v in fps.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_chunks(path: Path, chunks: list[ChunkRecord]) -> None:
    """全量覆写 chunks.jsonl（增量逻辑在 pipeline 合并后一次写入）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def save_commit(path: Path, commit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(commit, encoding="utf-8")
