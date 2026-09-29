"""
命令行入口（Typer）。

典型流程：
  1. python -m docpulse ingest   — 拉文档、分块 → chunks.jsonl
  2. python -m docpulse index    — 向量 + BM25
  3. python -m docpulse query    — 终端问答
  4. python -m docpulse serve    — Web UI
"""

import typer
from typing import Optional
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

from docpulse.ingest.indexer import run_index
from docpulse.retrieval.engine import get_engine

# 从 .env 加载 DEEPSEEK_API_KEY 等环境变量
load_dotenv()

app = typer.Typer(
    name="docpulse",
    help="DocPulse-RAG：文档脉冲检索增强系统",
    no_args_is_help=True,
)
console = Console()


@app.command("ingest")
def cmd_ingest(
    pull: bool = typer.Option(False, "--pull", help="对已 clone 的仓库执行 git pull"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="仅处理指定项目"),
    full: bool = typer.Option(False, "--full", help="忽略指纹，全量重新解析所有文件"),
    no_sync: bool = typer.Option(
        False, "--no-sync", help="跳过 git clone/pull，使用 data/repos 已有目录"
    ),
    force_clone: bool = typer.Option(
        False,
        "--force-clone",
        help="删除非 git 的 data/repos/<project> 后重新 clone（获取完整官方文档）",
    ),
    config: Optional[str] = typer.Option(None, "--config", help="config.yaml 路径"),
) -> None:
    """同步 Git 仓库，解析 Markdown，生成 Parent-Child 块并写入 data/state。"""
    from docpulse.ingest.pipeline import run_ingest

    run_ingest(
        pull=pull,
        project_name=project,
        skip_unchanged=not full,
        no_sync=no_sync,
        force_clone=force_clone,
        config_path=config,
    )


@app.command("index")
def cmd_index(
    project: Optional[str] = typer.Option(None, "--project", "-p"),
    recreate: bool = typer.Option(
        False, "--recreate", help="删除并重建本地 Chroma collection 后全量写入"
    ),
    config: Optional[str] = typer.Option(None, "--config"),
) -> None:
    """将 child chunks 向量化写入本地 Chroma，并构建 BM25 索引。"""
    run_index(project_name=project, recreate=recreate, config_path=config)


# 消融开关：--no-hybrid / --no-rerank / --no-rewrite 用于对比各阶段贡献（M5 评测）
@app.command("query")
def cmd_query(
    question: str = typer.Argument(..., help="用户问题"),
    project: str = typer.Option("fastapi", "--project", "-p"),
    no_hybrid: bool = typer.Option(False, "--no-hybrid", help="关闭 BM25，仅向量检索"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="跳过 Cross-Encoder 重排"),
    no_rewrite: bool = typer.Option(False, "--no-rewrite", help="不做问句规则扩展"),
) -> None:
    """检索 + DeepSeek 生成带引用的回答。"""
    engine = get_engine()
    result = engine.ask(
        question,
        project=project,
        use_hybrid=not no_hybrid,
        use_rerank=not no_rerank,
        use_rewrite=not no_rewrite,
    )
    console.print(Markdown(result.answer))
    console.print("\n[bold]引用来源[/bold]")
    for s in result.sources:
        console.print(f"  • {s.file_path} — {s.heading_path} (score={s.score:.3f})")
    if result.queries_used:
        console.print(f"\n[dim]检索问句: {', '.join(result.queries_used)}[/dim]")


@app.command("serve")
def cmd_serve(
    host: str = typer.Option("127.0.0.1", help="监听地址"),
    port: int = typer.Option(8000, help="端口"),
    reload: bool = typer.Option(False, "--reload", help="开发热重载"),
) -> None:
    """启动 Web 问答服务（FastAPI + 静态页）。"""
    try:
        import uvicorn
    except ImportError as e:
        raise typer.Exit(1) from e
    rprint(f"[green]Web[/green] http://{host}:{port}")
    uvicorn.run(
        "docpulse.web.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
