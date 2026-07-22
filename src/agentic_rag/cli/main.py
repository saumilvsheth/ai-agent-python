"""
Typer-based CLI for ingest, query, stats, and serve commands.

Reuses the same service layer as the REST API so CLI and HTTP behave
consistently. Rich is used for formatted terminal output.
"""

from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel

from agentic_rag.agent.service import RAGAgent
from agentic_rag.api.service import index_stats, ingest_documents
from agentic_rag.config import settings

# Root Typer app; subcommands are registered with @app.command().
app = typer.Typer(
    name="agentic-rag",
    help="Agentic RAG CLI — ingest documents and query with an adaptive LangGraph agent.",
)
console = Console()


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest"),
    reset: bool = typer.Option(False, help="Clear the vector index before ingesting"),
) -> None:
    """
    Load files, chunk them, embed, and persist to the FAISS index.

    Example: agentic-rag ingest sample_docs --reset
    """
    result = ingest_documents(str(path), reset=reset)
    console.print(
        f"[green]Ingested[/green] {result.documents} documents → {result.chunks} chunks"
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the agent"),
) -> None:
    """
    Run the full LangGraph agent and print the answer.

    Also shows routing decision, rewrite count, and source files when present.
    """
    result = RAGAgent().query(question)
    console.print(Panel(result.answer, title="Answer", border_style="cyan"))
    if result.route:
        console.print(f"[dim]Route:[/dim] {result.route}")
    if result.rewrite_count:
        console.print(f"[dim]Query rewrites:[/dim] {result.rewrite_count}")
    if result.sources:
        console.print("[dim]Sources:[/dim]")
        for source in result.sources:
            console.print(f"  • {source}")


@app.command()
def stats() -> None:
    """Print the number of chunks currently stored in the vector index."""
    result = index_stats()
    console.print(f"Indexed chunks: [bold]{result.chunk_count}[/bold]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """
    Start the FastAPI server via uvicorn.

    reload=True watches files during development; keep False in production.
    """
    console.print(f"Serving on http://{host}:{port}  (model: {settings.openai_model})")
    uvicorn.run(
        "agentic_rag.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


# Allow running as `python -m agentic_rag.cli.main` during development.
if __name__ == "__main__":
    app()
