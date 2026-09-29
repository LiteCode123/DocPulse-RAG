"""
配置加载模块（M1/M2 共用入口）。

职责：
  - 定位项目根目录（含 config.yaml）
  - 用 Pydantic 将 YAML 解析为类型安全的 AppConfig
  - 提供路径辅助方法（repos、state）

数据流：config.yaml → load_config() → 各子模块（ingest / index / RAGEngine）
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def project_root() -> Path:
    """项目根目录（含 config.yaml）。优先当前工作目录，否则从本文件向上查找。"""
    cwd = Path.cwd()
    if (cwd / "config.yaml").exists():
        return cwd
    # 从 src/docpulse 向上找（便于在子目录执行 python -m docpulse）
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / "config.yaml").exists():
            return parent
    return cwd


class ProjectConfig(BaseModel):
    """单个文档源项目（如 fastapi 官方 docs）。"""

    name: str  # 项目标识，对应 data/repos/<name> 与 data/state/<name>
    repo: str  # 主 Git 仓库 URL
    branch: str = "master"
    docs_glob: str  # 相对仓库根的路径 glob，如 docs/en/docs/**/*.md
    repo_mirrors: list[str] = Field(default_factory=list)  # clone 失败时的镜像列表

    def clone_urls(self) -> list[str]:
        """主仓库 + 镜像，去重保序。"""
        seen: set[str] = set()
        urls: list[str] = []
        for u in [self.repo, *self.repo_mirrors]:
            if u and u not in seen:
                seen.add(u)
                urls.append(u)
        return urls


class ChunkConfig(BaseModel):
    """Parent-Child 分块尺寸（见 ingest/chunker.py）。"""

    parent_max_chars: int = 1500  # 每个 ##/### 章节 Parent 最大字符
    child_size: int = 400  # Child 目标长度
    child_overlap: int = 50  # 相邻 Child 字符重叠，减少边界截断


class PathsConfig(BaseModel):
    """磁盘目录布局。"""

    repos_dir: str = "data/repos"  # Git 克隆的文档仓库
    state_dir: str = "data/state"  # chunks.jsonl、BM25、指纹等


class RetrievalConfig(BaseModel):
    """在线检索超参（见 retrieval/engine.py）。"""

    dense_k: int = 50  # 向量检索每路召回数
    sparse_k: int = 50  # BM25 每路召回数
    rrf_k: int = 60  # RRF 融合常数 k（越大越平滑）
    rerank_k: int = 8  # Cross-Encoder 重排后保留的 child 数
    final_parent_n: int = 4  # 去重后送入 LLM 的 Parent 上下文条数


class EmbeddingConfig(BaseModel):
    """向量化模型（sentence-transformers）。"""

    model: str = "BAAI/bge-small-en-v1.5"
    device: str = "auto"  # auto → 有 CUDA 用 GPU


class RerankConfig(BaseModel):
    """重排模型（CrossEncoder）。"""

    model: str = "BAAI/bge-reranker-base"
    device: str = "auto"


class LLMConfig(BaseModel):
    """生成阶段 LLM（OpenAI 兼容 API，默认 DeepSeek）。"""

    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.3
    max_tokens: int = 2048


class VectorStoreConfig(BaseModel):
    """本地向量库（ChromaDB 持久化到磁盘，无需 Docker）。"""

    backend: str = "chroma"
    persist_dir: str = "data/state/chroma"
    collection: str = "docpulse_chunks"


class QueryRewriteConfig(BaseModel):
    """问句扩展：当前为规则扩展，M4 计划加 LLM Multi-Query。"""

    rule_expansions: int = 2  # 除原问外最多附加几条扩展问句
    llm_expansions: int = 0  # 预留


class AppConfig(BaseModel):
    """根配置对象，对应 config.yaml 顶层结构。"""

    projects: list[ProjectConfig]
    paths: PathsConfig = Field(default_factory=PathsConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    query_rewrite: QueryRewriteConfig = Field(default_factory=QueryRewriteConfig)

    def repos_path(self, root: Path | None = None) -> Path:
        """文档 Git 仓库存放根目录。"""
        return (root or project_root()) / self.paths.repos_dir

    def state_path(self, root: Path | None = None) -> Path:
        """解析产物与索引侧车文件根目录。"""
        return (root or project_root()) / self.paths.state_dir


@lru_cache  # 无参缓存：避免每次检索重复读盘解析 YAML
def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    加载并缓存 config.yaml。
    进程内多次调用返回同一 AppConfig 实例（@lru_cache）。
    """
    root = project_root()
    path = Path(config_path) if config_path else root / "config.yaml"
    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
