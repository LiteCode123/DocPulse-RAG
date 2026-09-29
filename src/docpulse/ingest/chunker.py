"""
Parent-Child 分块（M1 核心）。

Parent：每个 ##/### 章节一篇，带标题前缀，长度受 parent_max_chars 限制。
Child：在 Parent 内按「原子块」（段落/代码块/表格）合并，带字符 overlap。

检索时：用 child 向量/BM25 命中 → 取对应 parent 全文给 LLM。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docpulse.config import ChunkConfig
from docpulse.ingest.parser import ParsedDocument, Section

CODE_FENCE_RE = re.compile(r"^```")


@dataclass
class ParentChunk:
    section_title: str
    heading_path: str
    text: str


@dataclass
class ChildChunk:
    parent_heading_path: str
    child_index: int
    text: str


def _split_blocks(body: str) -> list[str]:
    """
    按段落 / 代码块 / 表格边界切分为「原子块」（atomic blocks）。

    设计动机（结构感知分块）：
      - 若按固定字符数硬切，容易把 ```python 代码块或 Markdown 表格拦腰截断，
        检索到的片段语义不完整，LLM 也无法正确理解 API 示例。
      - 本函数用有限状态机（FSM）扫描行：维护 in_code / in_table 标志，
        只在「安全边界」flush 缓冲区。

    状态转移简述：
      遇到 ``` → 进入/退出代码模式（整块保留）
      以 | 开头的行 → 表格模式（连续 | 行合并为一块）
      空行 → 段落边界，flush 当前 buf
    """
    lines = body.splitlines()
    blocks: list[str] = []
    buf: list[str] = []  # 当前正在累积的原子块
    in_code = False
    in_table = False

    def flush() -> None:
        nonlocal buf, in_table
        if buf:
            blocks.append("\n".join(buf).strip())
            buf = []
        in_table = False

    for line in lines:
        stripped = line.strip()
        if CODE_FENCE_RE.match(stripped):
            if in_code:
                buf.append(line)
                flush()
                in_code = False
            else:
                flush()
                in_code = True
                buf.append(line)
            continue

        if in_code:
            buf.append(line)
            continue

        is_table_line = "|" in line and stripped.startswith("|")
        if is_table_line:
            if not in_table:
                flush()
                in_table = True
            buf.append(line)
            continue

        if in_table:
            flush()

        if not stripped:
            flush()
            continue

        buf.append(line)

    flush()
    return [b for b in blocks if b]


def _merge_blocks_to_size(blocks: list[str], size: int, overlap: int) -> list[str]:
    """
    贪心合并：在不超过 size 的前提下，尽量把多个原子块拼成一个 child。

    算法：类似 bin packing 的在线贪心——能放下就合并，放不下则封存当前 piece 开新段。
    第二遍对相邻 piece 施加 overlap（字符级滑动窗口），缓解「答案恰好在块边界」的问题。
    """
    if not blocks:
        return []

    pieces: list[str] = []
    current = ""

    for block in blocks:
        if not current:
            current = block
            continue
        candidate = f"{current}\n\n{block}"
        if len(candidate) <= size:
            current = candidate
        else:
            pieces.append(current)
            current = block

    if current:
        pieces.append(current)

    if len(pieces) <= 1:
        return pieces

    # 应用字符级 overlap：下一段开头带上上一段末尾 overlap 字符
    with_overlap: list[str] = []
    for i, piece in enumerate(pieces):
        if i == 0:
            with_overlap.append(piece)
            continue
        prev = with_overlap[-1]
        tail = prev[-overlap:] if overlap > 0 and len(prev) > overlap else prev
        merged = f"{tail}\n\n{piece}" if tail else piece
        with_overlap.append(merged)

    return with_overlap


def build_parent_chunks(section: Section, cfg: ChunkConfig) -> ParentChunk:
    """构造 Parent 文本：可选 ## 标题 + body，超长截断。"""
    text = section.body.strip()
    if len(text) > cfg.parent_max_chars:
        text = text[: cfg.parent_max_chars] + "\n\n…"
    header = f"## {section.title}\n\n" if section.title else ""
    return ParentChunk(
        section_title=section.title,
        heading_path=section.heading_path,
        text=(header + text).strip(),
    )


def build_child_chunks(parent: ParentChunk, cfg: ChunkConfig) -> list[ChildChunk]:
    """在一个 Parent 内切出多个 Child。"""
    blocks = _split_blocks(parent.text)
    texts = _merge_blocks_to_size(blocks, cfg.child_size, cfg.child_overlap)
    if not texts:
        return []
    if len(texts) == 1 and len(texts[0]) > cfg.child_size * 2:
        # 单块过大时按字符硬切（已尽量保持代码块完整）
        texts = _hard_split(texts[0], cfg.child_size, cfg.child_overlap)
    return [
        ChildChunk(
            parent_heading_path=parent.heading_path,
            child_index=i,
            text=t,
        )
        for i, t in enumerate(texts)
    ]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """滑动窗口字符切分，步长 size - overlap。"""
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        out.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_document(doc: ParsedDocument, cfg: ChunkConfig) -> list[tuple[ParentChunk, list[ChildChunk]]]:
    """
    整篇文档 → 多组 (parent, children)。

    Parent-Child 双层索引（RAG 常见模式）：
      - Child 短、语义集中 → 用于向量/BM25 检索（高召回、定位准）
      - Parent 长、含完整章节 → 命中 child 后回溯 parent 给 LLM（上下文完整）
    """
    result: list[tuple[ParentChunk, list[ChildChunk]]] = []
    for section in doc.sections:
        parent = build_parent_chunks(section, cfg)
        children = build_child_chunks(parent, cfg)
        if children:
            result.append((parent, children))
    return result
