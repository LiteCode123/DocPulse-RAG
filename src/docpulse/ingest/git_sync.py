"""
Git 仓库同步（clone / pull / 本地目录模式）。

返回当前 HEAD 的 hexsha，写入 last_commit.txt；
无 .git 时返回 "local"（用户手动解压 ZIP 的场景）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from docpulse.config import ProjectConfig


def repo_dir(repos_root: Path, project: ProjectConfig) -> Path:
    """例如 data/repos/fastapi。"""
    return repos_root / project.name


def _clone_attempt(url: str, dest: Path, branches: list[str]) -> str:
    """
    浅克隆 single-branch，依次尝试 branch 列表。
    失败则删除 dest 再试下一分支。
    """
    from git import Repo
    from git.exc import GitCommandError

    last_err: Optional[Exception] = None
    seen_branches: set[str] = set()
    for branch in branches:
        if branch in seen_branches:
            continue
        seen_branches.add(branch)
        try:
            repo = Repo.clone_from(
                url,
                dest,
                branch=branch,
                depth=1,
                single_branch=True,
            )
            return repo.head.commit.hexsha
        except GitCommandError as e:
            last_err = e
            if dest.exists():
                import shutil

                shutil.rmtree(dest, ignore_errors=True)
    raise GitCommandError(
        ["git", "clone"],
        128,
        f"clone 失败: {url} | {last_err}",
    ) from last_err


def _branch_candidates(project: ProjectConfig) -> list[str]:
    """配置 branch + 常见 master/main 回退。"""
    branches = [project.branch, "master", "main"]
    seen: set[str] = set()
    out: list[str] = []
    for b in branches:
        if b and b not in seen:
            seen.add(b)
            out.append(b)
    return out


def sync_repo(
    project: ProjectConfig,
    repos_root: Path,
    pull: bool = False,
    *,
    no_sync: bool = False,
    force_clone: bool = False,
) -> str:
    """
    Clone 或 pull 仓库，返回当前 HEAD commit（或 "local"）。

    分支逻辑简述：
      no_sync     — 仅用已有目录，要求路径存在
      已有 .git   — 可选 pull 后返回 HEAD
      无 .git 有 md — 视为 local，不删目录
      force_clone — 删掉目录后重新 clone
    """
    dest = repo_dir(repos_root, project)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # ── 模式 A：--no-sync，不碰网络，只用本地已有目录 ──────────────
    if no_sync:
        if not dest.exists():
            raise FileNotFoundError(
                f"--no-sync 需要已有仓库目录: {dest}\n"
                "请先 git clone 或复制文档到该路径。详见 docs/入库指南.md"
            )
        if (dest / ".git").exists():
            from git import Repo

            repo = Repo(dest)
            return repo.head.commit.hexsha
        return "local"

    # ── 模式 B：已有 Git 仓库 → 可选 pull 后返回 HEAD ─────────────
    if dest.exists() and (dest / ".git").exists():
        from git import Repo
        from git.exc import GitCommandError

        repo = Repo(dest)
        if pull:
            origin = repo.remotes.origin
            origin.fetch()
            try:
                origin.pull(project.branch)
            except GitCommandError:
                if project.branch != "main":
                    origin.pull("main")
        return repo.head.commit.hexsha

    if dest.exists():
        if force_clone:
            import shutil

            shutil.rmtree(dest, ignore_errors=True)
        elif no_sync or _has_markdown(dest):
            return "local"
        else:
            import shutil

            shutil.rmtree(dest, ignore_errors=True)

    # ── 模式 C：浅克隆（depth=1）主仓库 + 镜像 URL 依次尝试 ───────
    from git.exc import GitCommandError

    branches = _branch_candidates(project)
    errors: list[str] = []
    last_err: Optional[GitCommandError] = None
    for url in project.clone_urls():
        try:
            return _clone_attempt(url, dest, branches)
        except GitCommandError as e:
            last_err = e
            errors.append(str(e))

    manual = (
        f"\n\n无法从网络 clone {project.name}，已尝试:\n"
        + "\n".join(f"  - {u}" for u in project.clone_urls())
        + f"\n\n请任选一种方式:\n"
        f"  1) 配置代理后重试: python -m docpulse ingest --force-clone\n"
        f"  2) 浏览器下载 ZIP 解压到: {dest}\n"
        f"     见 docs/入库指南.md\n"
        f"  3) 手动 git clone 成功后: python -m docpulse ingest --no-sync\n"
    )
    raise GitCommandError(
        ["git", "clone"], 128, manual + "\n错误:\n" + "\n".join(errors)
    ) from last_err


def _has_markdown(dest: Path) -> bool:
    """目录里已有 .md 则不再强制 clone（本地文档包）。"""
    return any(dest.rglob("*.md"))
