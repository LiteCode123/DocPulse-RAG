"""
入库主编排（ingest 命令）。

流程：sync_repo → 遍历 docs_glob → 指纹比对 → 解析分块 → 合并 chunks.jsonl
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from docpulse.config import AppConfig, ProjectConfig, load_config, project_root
from docpulse.ingest import chunker, parser
from docpulse.ingest.fingerprint import (
    make_chunk_id,
    make_doc_id,
    make_parent_id,
    sha256_file,
)
from docpulse.ingest.git_sync import repo_dir, sync_repo
from docpulse.ingest.state import (
    chunks_path,
    commit_path,
    fingerprints_path,
    load_chunks,
    load_fingerprints,
    save_chunks,
    save_commit,
    save_fingerprints,
)
from docpulse.models import ChunkRecord, FileFingerprint, utc_now_iso

console = Console()


def _iter_doc_files(repo_path: Path, docs_glob: str) -> list[Path]:
    """按 glob 收集所有 Markdown 文件并排序。"""
    return sorted(repo_path.glob(docs_glob))


def _rel_path(repo_path: Path, file_path: Path) -> str:
    """转为 POSIX 相对路径，作为 file_path 字段。"""
    return file_path.relative_to(repo_path).as_posix()


def _file_to_chunks(
    project: ProjectConfig,
    rel: str,
    file_path: Path,
    commit: str,
    cfg: AppConfig,
) -> tuple[list[ChunkRecord], FileFingerprint]:
    """
    单文件：解析 → Parent-Child → ChunkRecord 列表 + 文件指纹。

    无 ## 标题的纯文档：退化为单个 (document) parent + 一个 child。
    """
    content_hash = sha256_file(file_path)
    doc_id = make_doc_id(project.name, rel)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    parsed = parser.parse_markdown(rel, text)
    pairs = chunker.chunk_document(parsed, cfg.chunk)

    chunks: list[ChunkRecord] = []
    for parent, children in pairs:
        parent_id = make_parent_id(doc_id, parent.heading_path)
        # Parent 记录写入 chunks.jsonl 但不进向量库；检索命中 child 后按 parent_id 取全文
        chunks.append(
            ChunkRecord(
                doc_id=doc_id,
                chunk_id=make_chunk_id(doc_id, parent.heading_path, "0", "parent"),
                parent_id=parent_id,
                project=project.name,
                file_path=rel,
                section_title=parent.section_title,
                heading_path=parent.heading_path,
                git_commit=commit,
                content_hash=content_hash,
                level="parent",
                text=parent.text,
            )
        )
        for child in children:
            chunks.append(
                ChunkRecord(
                    doc_id=doc_id,
                    chunk_id=make_chunk_id(
                        doc_id, child.parent_heading_path, f"child:{child.child_index}", "child"
                    ),
                    parent_id=parent_id,
                    project=project.name,
                    file_path=rel,
                    section_title=parent.section_title,
                    heading_path=parent.heading_path,
                    git_commit=commit,
                    content_hash=content_hash,
                    level="child",
                    text=child.text,
                    child_index=child.child_index,
                )
            )

    # 无章节标题时的兜底：整文件当作一块
    if not chunks and text.strip():
        heading_path = "(document)"
        parent_id = make_parent_id(doc_id, heading_path)
        parent_text = text[: cfg.chunk.parent_max_chars]
        chunks.append(
            ChunkRecord(
                doc_id=doc_id,
                chunk_id=make_chunk_id(doc_id, heading_path, "0", "parent"),
                parent_id=parent_id,
                project=project.name,
                file_path=rel,
                section_title="(root)",
                heading_path=heading_path,
                git_commit=commit,
                content_hash=content_hash,
                level="parent",
                text=parent_text,
            )
        )
        chunks.append(
            ChunkRecord(
                doc_id=doc_id,
                chunk_id=make_chunk_id(doc_id, heading_path, "child:0", "child"),
                parent_id=parent_id,
                project=project.name,
                file_path=rel,
                section_title="(root)",
                heading_path=heading_path,
                git_commit=commit,
                content_hash=content_hash,
                level="child",
                text=parent_text,
            )
        )

    fp = FileFingerprint(
        file_path=rel,
        content_hash=content_hash,
        doc_id=doc_id,
        updated_at=utc_now_iso(),
    )
    return chunks, fp


def process_project(
    project: ProjectConfig,
    cfg: AppConfig,
    *,
    pull: bool = False,
    skip_unchanged: bool = True,
    no_sync: bool = False,
    force_clone: bool = False,
) -> dict[str, int]:
    """
    处理一个项目的完整 ingest。

    skip_unchanged=True 时：
      - 文件 hash 未变 → 保留旧 chunks，只更新指纹表
      - 变更文件 → 重算 chunks，并从全量列表中替换同 doc_id 的旧块
    """
    root = project_root()
    repos_root = cfg.repos_path(root)
    state_root = cfg.state_path(root)
    repo_path = repo_dir(repos_root, project)

    console.print(f"[bold]同步仓库[/bold] {project.name} …")
    commit = sync_repo(
        project, repos_root, pull=pull, no_sync=no_sync, force_clone=force_clone
    )
    if commit == "local":
        console.print(
            "[yellow]提示[/yellow] 使用本地文档目录（无 .git）。"
            "完整官方文档请删除 data/repos/fastapi 后执行 "
            "[bold]python -m docpulse ingest --force-clone[/bold]"
        )
    save_commit(commit_path(state_root, project.name), commit)

    md_files = _iter_doc_files(repo_path, project.docs_glob)
    if not md_files:
        console.print(f"[yellow]未找到文档[/yellow] glob={project.docs_glob} @ {repo_path}")
        return {"files": 0, "parents": 0, "children": 0, "skipped": 0}

    fp_path = fingerprints_path(state_root, project.name)
    chunk_file = chunks_path(state_root, project.name)
    old_fps = load_fingerprints(fp_path)
    old_chunks = load_chunks(chunk_file) if skip_unchanged else []

    new_fps: dict[str, FileFingerprint] = {}
    changed_chunks: list[ChunkRecord] = []
    unchanged_doc_ids: set[str] = set()
    stats = {"files": 0, "parents": 0, "children": 0, "skipped": 0}

    for file_path in md_files:
        rel = _rel_path(repo_path, file_path)
        content_hash = sha256_file(file_path)
        prev = old_fps.get(rel)

        if skip_unchanged and prev and prev.content_hash == content_hash:
            new_fps[rel] = prev
            unchanged_doc_ids.add(prev.doc_id)
            stats["skipped"] += 1
            continue

        stats["files"] += 1
        chunks, fp = _file_to_chunks(project, rel, file_path, commit, cfg)
        changed_chunks.extend(chunks)
        new_fps[rel] = fp
        for c in chunks:
            if c.level == "parent":
                stats["parents"] += 1
            else:
                stats["children"] += 1

    if skip_unchanged:
        # 增量合并策略（M1 指纹驱动，M3 将扩展到向量索引）：
        #   kept     = 未变文件对应的全部旧 chunks（按 doc_id 保留）
        #   changed  = 本次重算文件的 chunks
        #   去掉 kept 里与 changed 同 doc_id 的条目，避免同一文档出现两套块
        kept = [c for c in old_chunks if c.doc_id in unchanged_doc_ids]
        changed_doc_ids = {c.doc_id for c in changed_chunks}
        kept = [c for c in kept if c.doc_id not in changed_doc_ids]
        final_chunks = kept + changed_chunks
    else:
        final_chunks = changed_chunks

    save_fingerprints(fp_path, new_fps)
    save_chunks(chunk_file, final_chunks)
    return stats


def run_ingest(
    *,
    pull: bool = False,
    project_name: str | None = None,
    skip_unchanged: bool = True,
    no_sync: bool = False,
    force_clone: bool = False,
    config_path: str | None = None,
) -> None:
    """CLI ingest 入口：可处理 config 中的全部或单个 project。"""
    cfg = load_config(config_path)
    projects = cfg.projects
    if project_name:
        projects = [p for p in projects if p.name == project_name]
        if not projects:
            raise SystemExit(f"未找到项目配置: {project_name}")

    table = Table(title="Ingest 统计")
    table.add_column("project")
    table.add_column("files")
    table.add_column("parents")
    table.add_column("children")
    table.add_column("skipped")

    for project in projects:
        stats = process_project(
            project,
            cfg,
            pull=pull,
            skip_unchanged=skip_unchanged,
            no_sync=no_sync,
            force_clone=force_clone,
        )
        table.add_row(
            project.name,
            str(stats["files"]),
            str(stats["parents"]),
            str(stats["children"]),
            str(stats["skipped"]),
        )

    console.print(table)
    console.print("[green]完成[/green] 状态目录:", cfg.state_path(project_root()))
