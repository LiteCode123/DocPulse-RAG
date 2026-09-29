"""
LLM 答案生成（DeepSeek OpenAI 兼容 API）。

环境变量：DEEPSEEK_API_KEY（或 OPENAI_API_KEY）
系统提示要求：仅依据上下文、中文回答、文末引用格式。
"""

from __future__ import annotations

import os
from typing import Generator, List, Optional, Tuple

from docpulse.config import AppConfig, load_config


def _get_client(cfg: AppConfig):
    """构造 OpenAI SDK 客户端，base_url 指向 DeepSeek。"""
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未设置 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入密钥。"
        )
    return OpenAI(api_key=api_key, base_url=cfg.llm.base_url)


def build_chat_messages(
    question: str, contexts: List[str]
) -> Tuple[str, str, Optional[str]]:
    """
    组装 system / user 消息。

    返回 (system, user, empty_message_if_no_context)。
    无检索结果时第三项为非空提示字符串，不调 LLM。
    """
    if not contexts:
        return "", "", "未检索到相关文档片段，请换种问法或确认已执行 ingest 与 index。"

    # 多段 Parent 上下文用分隔符拼接，便于模型区分不同来源
    context_text = "\n\n---\n\n".join(contexts)
    # Prompt 设计要点（RAG 生成阶段）：
    #   1. 角色 + 边界：仅依据上下文，减少幻觉
    #   2. 引用格式：便于用户核对，也为 M5 评测「引用命中率」留钩子
    #   3. 不足时明确拒答：避免模型用预训练知识瞎编
    system = (
        "你是 FastAPI 官方文档助手。仅根据提供的文档上下文回答，"
        "不要编造。回答使用中文。在文末列出引用，格式："
        "[来源: 文件路径 | 章节路径]。若上下文不足，明确说明不知道。"
    )
    user = f"问题：{question}\n\n文档上下文：\n{context_text}"
    return system, user, None


def generate_answer(
    question: str,
    contexts: List[str],
    cfg: Optional[AppConfig] = None,
) -> str:
    """非流式 chat.completions，返回完整文本。"""
    cfg = cfg or load_config()
    system, user, empty = build_chat_messages(question, contexts)
    if empty:
        return empty

    client = _get_client(cfg)
    resp = client.chat.completions.create(
        model=cfg.llm.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_answer_stream(
    question: str,
    contexts: List[str],
    cfg: Optional[AppConfig] = None,
) -> Generator[str, None, None]:
    """流式 chat.completions，逐 chunk yield delta 文本。"""
    cfg = cfg or load_config()
    system, user, empty = build_chat_messages(question, contexts)
    if empty:
        yield empty
        return

    client = _get_client(cfg)
    stream = client.chat.completions.create(
        model=cfg.llm.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
        stream=True,
    )
    # OpenAI 兼容流式 API：每个 chunk 可能只含少量 delta 文本
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta  # Web SSE 层逐 token 转发给浏览器
