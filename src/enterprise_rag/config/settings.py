"""Configuração central.

Toda a plataforma é parametrizada por variáveis de ambiente com prefixo ``RAG_``.
Nada de constante mágica espalhada pelo código: se um número influencia
qualidade ou latência, ele mora aqui e aparece no relatório de avaliação.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VectorBackend = Literal["qdrant", "memory"]
EmbeddingBackend = Literal["sentence-transformers", "hashing"]
RerankerBackend = Literal["cross-encoder", "lexical", "none"]
LLMBackend = Literal["ollama", "echo"]
ChunkStrategy = Literal["fixed", "structural"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── aplicação ───────────────────────────────────────────
    app_name: str = "enterprise-rag"
    environment: str = "local"
    log_level: str = "INFO"
    log_json: bool = False

    # ── armazenamento ───────────────────────────────────────
    data_dir: Path = Path("./data")
    catalog_path: Path = Path("./data/catalog.sqlite3")

    vector_backend: VectorBackend = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "enterprise_rag"

    # ── embeddings ──────────────────────────────────────────
    embedding_backend: EmbeddingBackend = "hashing"
    dense_model: str = "intfloat/multilingual-e5-small"
    dense_dim: int = 384
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # ── chunking ────────────────────────────────────────────
    chunk_strategy: ChunkStrategy = "structural"
    chunk_size: int = 500          # em tokens aproximados
    chunk_overlap: int = 60
    chunk_min_size: int = 40

    # ── retrieval ───────────────────────────────────────────
    use_hybrid: bool = True
    top_k_candidates: int = 30
    top_k_final: int = 6
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    # ── reranking ───────────────────────────────────────────
    reranker_backend: RerankerBackend = "lexical"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_min_score: float = -1e9

    # ── context builder ─────────────────────────────────────
    max_context_tokens: int = 7000
    max_chunks_per_document: int = 3
    dedup_similarity: float = 0.92

    # ── geração ─────────────────────────────────────────────
    llm_backend: LLMBackend = "echo"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_temperature: float = 0.1
    ollama_num_ctx: int = 8192
    ollama_timeout: float = 120.0
    ollama_max_retries: int = 2

    # ── API ─────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_upload_mb: int = 50

    # ── observabilidade ─────────────────────────────────────
    otel_enabled: bool = False
    otel_endpoint: str | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def sparse_stats_path(self) -> Path:
        return self.data_dir / "sparse_stats.json"

    @property
    def memory_store_path(self) -> Path:
        return self.data_dir / "memory_store.json"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


def reset_settings_cache() -> None:
    """Usado em testes que trocam variáveis de ambiente."""
    get_settings.cache_clear()
