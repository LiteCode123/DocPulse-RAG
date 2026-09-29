"""
DocPulse-RAG: 文档脉冲检索增强系统。

推荐阅读顺序（学习代码时）：
  1. config.py / models.py          — 配置与数据结构
  2. ingest/parser.py + chunker.py  — 结构感知分块（Parent-Child）
  3. ingest/pipeline.py + indexer.py — 入库与索引
  4. retrieval/engine.py            — RAG 检索编排（核心）
  5. retrieval/hybrid.py + rerank.py — RRF 与 Cross-Encoder
  6. generation/answer.py           — LLM 生成
  7. web/app.py                     — HTTP/SSE 接口

包结构概览：
  ingest/     — M1 同步 Git、解析 Markdown、分块、写 chunks.jsonl
  ingest/indexer — M2 向量化 + BM25
  vectorstore/ — Chroma 持久化
  retrieval/  — 混合检索、RRF、重排
  generation/ — DeepSeek 流式/非流式回答
  web/        — FastAPI + SSE 前端
  cli.py      — Typer 命令行入口
"""

__version__ = "0.1.0"
