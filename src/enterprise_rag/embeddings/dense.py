"""Embeddings densos.

Dois backends atrás da mesma interface:

* `sentence-transformers` para qualidade real (padrão: multilingual-e5-small,
  que entende português sem custo de API);
* `hashing`, determinístico e sem download, usado em CI e em qualquer máquina
  sem GPU/internet. Não é bom para recall, mas mantém o pipeline inteiro
  executável e testável, o que é o ponto.

Trocar por Azure OpenAI/Foundry depois significa escrever uma terceira classe
com estes dois métodos.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

from enterprise_rag.config.logging import get_logger
from enterprise_rag.config.settings import Settings

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@runtime_checkable
class DenseEmbedder(Protocol):
    """Contrato mínimo. Documento e consulta são codificados separadamente
    porque modelos assimétricos (E5, BGE, GTE) exigem prefixos diferentes."""

    name: str
    dim: int

    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...

    def encode_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Hashing trick sobre unigramas e bigramas, com normalização L2.

    Determinístico entre execuções e processos: o mesmo texto sempre gera o
    mesmo vetor, o que torna as métricas de avaliação reprodutíveis em CI.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def encode_query(self, text: str) -> list[float]:
        return self._vector(text)


class SentenceTransformerEmbedder:
    """Wrapper com carregamento preguiçoso do modelo.

    O import fica dentro do construtor de propósito: importar torch custa
    segundos, e nem todo caminho de código (testes unitários, CLI de
    avaliação com backend hashing) precisa dele.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("carregando modelo denso", extra={"model": model_name, "device": device})
        self._model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.batch_size = batch_size
        # E5 e BGE são assimétricos: exigem prefixo diferente para query e passagem.
        lowered = model_name.lower()
        self._doc_prefix = "passage: " if "e5" in lowered else ""
        self._query_prefix = "query: " if "e5" in lowered else ""

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [self._doc_prefix + text for text in texts]
        vectors = self._model.encode(
            prepared,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vector.tolist() for vector in vectors]

    def encode_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            self._query_prefix + text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()


def build_dense_embedder(settings: Settings) -> DenseEmbedder:
    if settings.embedding_backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder(
                model_name=settings.dense_model,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
            )
        except ImportError:
            logger.warning(
                "sentence-transformers nao instalado; usando embedder hashing "
                "(instale com: pip install '.[ml]')"
            )
    return HashingEmbedder(dim=settings.dense_dim)
