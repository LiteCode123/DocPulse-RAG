"""
结构感知 Markdown 解析（M1）。

只把 ##、### 当作 Parent 章节边界；更低级标题留在 section body 内。
代码块、表格行不会被误切成新章节（由 chunker 按块处理）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 匹配 ATX 标题：# 到 ######
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class Section:
    """一个 ## / ### 章节（Parent 候选）。"""

    level: int  # 2 或 3
    title: str
    heading_path: str  # 面包屑，如 "Tutorial > First Steps"
    body: str  # 标题行之后到下一同级标题之前的正文


@dataclass
class ParsedDocument:
    rel_path: str
    sections: list[Section] = field(default_factory=list)


def _normalize_heading(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip())


def parse_markdown(rel_path: str, text: str, parent_heading_levels: set[int] | None = None) -> ParsedDocument:
    """
    结构感知解析：按 ##、### 切分为章节。
    代码块、表格行保持在 section body 内，由 chunker 负责不切断。

    算法要点：
      - stack 维护当前标题层级路径
      - 遇到 parent_heading_levels 中的标题时 flush 上一节并开始新节
      - 文档首个 ## 之前的内容可进入「引言」body（若 stack 非空）
    """
    if parent_heading_levels is None:
        parent_heading_levels = {2, 3}

    lines = text.splitlines()
    # 标题栈：维护当前文档的「面包屑路径」，类似 HTML 大纲或 DOM 树
    # 例：[(2, "Tutorial"), (3, "First Steps")] → heading_path = "Tutorial > First Steps"
    stack: list[tuple[int, str]] = []  # (level, title)
    sections: list[Section] = []

    current_level: int | None = None
    current_title = ""
    current_path = ""
    body_lines: list[str] = []

    def flush_section() -> None:
        """将当前累积的 body 写入 sections（仅当级别为 Parent 候选）。"""
        nonlocal body_lines
        if current_level is None:
            body_lines = []
            return
        if current_level not in parent_heading_levels:
            body_lines = []
            return
        body = "\n".join(body_lines).strip()
        if body or current_title:
            sections.append(
                Section(
                    level=current_level,
                    title=current_title,
                    heading_path=current_path,
                    body=body,
                )
            )
        body_lines = []

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            hashes, title = m.group(1), _normalize_heading(m.group(2))
            level = len(hashes)

            if level in parent_heading_levels:
                # 遇到新的 ##/###：先保存上一节 body，再开新节
                flush_section()
                # 标准「标题栈」操作：弹出 level >= 当前级的项（同级/子级标题结束）
                # 再 push 当前标题，保证 heading_path 始终反映从根到叶的层级
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                path_parts = [t for _, t in stack]
                current_level = level
                current_title = title
                current_path = " > ".join(path_parts)
                continue

            # #### 及以下：不作为新 Parent，写入当前 section 正文
            body_lines.append(line)
            continue

        if current_level is not None:
            body_lines.append(line)
        elif stack:
            # 文档在首个 ## 之前有引言
            body_lines.append(line)

    flush_section()
    return ParsedDocument(rel_path=rel_path, sections=sections)
