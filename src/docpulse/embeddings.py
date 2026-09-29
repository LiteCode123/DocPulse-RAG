"""
文本向量化（Embedding）模块。

使用 sentence-transformers 加载 config 中的模型；
入库 index 与在线 dense 检索共用 get_embedding_model() 单例。
向量 L2 归一化，与 Chroma cosine 距离配合。
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from docpulse.config import AppConfig, load_config


def resolve_device(device: str) -> str:
    """将 config 中的 'auto' 解析为 cuda 或 cpu。"""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@lru_cache(maxsize=1)
def get_embedding_model(model_name: Optional[str] = None, device: Optional[str] = None):
    """
    懒加载 SentenceTransformer，进程内只建一次。
    首次调用会下载模型权重（如 bge-small-en-v1.5）。
    """
    cfg = load_config()
    name = model_name or cfg.embedding.model
    dev = resolve_device(device or cfg.embedding.device)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=dev)


def embed_texts(
    texts: List[str],
    *,
    model_name: Optional[str] = None,
    batch_size: int = 32,
) -> List[List[float]]:
    """
    批量将文本编码为浮点向量列表。
    normalize_embeddings=True 便于用余弦相似度检索。
    """
    model = get_embedding_model(model_name)
    # normalize_embeddings=True：向量 L2 归一化后，点积等价于余弦相似度
    # Chroma 使用 hnsw:space=cosine，与归一化向量配合是常见做法
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 64,
        normalize_embeddings=True,
    )
    return vectors.tolist()
