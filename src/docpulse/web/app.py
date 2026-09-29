"""
FastAPI Web 服务：检索 + DeepSeek 流式生成。

路由：
  GET  /              — 静态 index.html
  GET  /api/health    — 健康检查
  POST /api/query     — 非流式 JSON 问答
  POST /api/query/stream — SSE：meta → token* → done

SSE 事件类型见 stream_answer 与前端 parseSSEStream。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from docpulse import __version__
from docpulse.retrieval.engine import ContextSnippet, SourceRef, get_engine

load_dotenv()

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="DocPulse-RAG", version=__version__)


class QueryRequest(BaseModel):
    """与 CLI query 参数对齐的 HTTP 请求体。"""

    question: str = Field(..., min_length=1)
    project: str = "fastapi"
    use_hybrid: bool = True
    use_rerank: bool = True
    use_rewrite: bool = True


class SnippetOut(BaseModel):
    """检索片段（Parent 级），供前端侧栏展示。"""

    file_path: str
    heading_path: str
    section_title: str
    text: str
    score: float


class SourceOut(BaseModel):
    file_path: str
    heading_path: str
    section_title: str
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceOut]
    queries_used: List[str]
    snippets: List[SnippetOut]


def _snippet_out(s: ContextSnippet) -> SnippetOut:
    return SnippetOut(
        file_path=s.file_path,
        heading_path=s.heading_path,
        section_title=s.section_title,
        text=s.text,
        score=s.score,
    )


def _source_out(s: SourceRef) -> SourceOut:
    return SourceOut(
        file_path=s.file_path,
        heading_path=s.heading_path,
        section_title=s.section_title,
        chunk_id=s.chunk_id,
        score=s.score,
    )


def _handle_query_error(e: Exception) -> HTTPException:
    """索引缺失 → 503；未配置 API Key → 500。"""
    if isinstance(e, FileNotFoundError):
        return HTTPException(status_code=503, detail=str(e))
    if isinstance(e, RuntimeError):
        return HTTPException(status_code=500, detail=str(e))
    msg = str(e)
    if "index" in msg.lower() or "bm25" in msg.lower():
        return HTTPException(status_code=503, detail="请先运行: python -m docpulse index")
    return HTTPException(status_code=500, detail=msg)


def _sse(payload: dict) -> str:
    """Server-Sent Events 单行：data: {json}\\n\\n"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html_path = _STATIC / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>DocPulse-RAG</h1>"


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/api/query", response_model=QueryResponse)
def api_query(req: QueryRequest) -> QueryResponse:
    """同步等待完整答案（适合 API 调用，非 UI 主路径）。"""
    try:
        result = get_engine().ask(
            req.question,
            project=req.project,
            use_hybrid=req.use_hybrid,
            use_rerank=req.use_rerank,
            use_rewrite=req.use_rewrite,
        )
    except Exception as e:
        raise _handle_query_error(e) from e

    return QueryResponse(
        answer=result.answer,
        sources=[_source_out(s) for s in result.sources],
        queries_used=result.queries_used,
        snippets=[_snippet_out(s) for s in result.snippets],
    )


@app.post("/api/query/stream")
def api_query_stream(req: QueryRequest) -> StreamingResponse:
    """流式 SSE：先推送检索 meta，再逐 token 推送答案。"""

    def event_gen() -> Iterator[str]:
        """
        SSE 生成器：与前端 index.html 的 parseSSEStream 协议对应。

        时序：meta（检索完成）→ token×N（流式答案）→ done
        出错时发 error 事件，前端显示 detail。
        """
        try:
            gen = get_engine().stream_answer(
                req.question,
                project=req.project,
                use_hybrid=req.use_hybrid,
                use_rerank=req.use_rerank,
                use_rewrite=req.use_rewrite,
            )
            # 第一次 yield：检索结果元数据
            first = next(gen)
            snippets, sources, queries = first
            yield _sse(
                {
                    "type": "meta",
                    "snippets": [_snippet_out(s).model_dump() for s in snippets],
                    "sources": [_source_out(s).model_dump() for s in sources],
                    "queries_used": queries,
                }
            )
            for token in gen:
                yield _sse({"type": "token", "content": token})
            yield _sse({"type": "done"})
        except Exception as e:
            detail = str(e)
            if isinstance(e, HTTPException):
                detail = e.detail  # type: ignore
            yield _sse({"type": "error", "detail": detail})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲，保证实时流
        },
    )
