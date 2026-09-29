"""向量存储抽象入口；当前仅实现 ChromaStore。"""

from docpulse.vectorstore.chroma_store import ChromaStore, get_vector_store

__all__ = ["ChromaStore", "get_vector_store"]
