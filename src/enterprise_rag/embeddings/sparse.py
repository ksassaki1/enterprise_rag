"""Embeddings esparsos (BM25 escrito à mão).

Por que não usar só o denso? Porque embedding falha exatamente onde documentos
corporativos são mais específicos: códigos de equipamento (`UHE-XPTO-01`),
normas (`NBR 6118`), números de contrato. Esses termos não têm vizinhança
semântica útil, mas têm sinal lexical altíssimo.

A implementação produz um vetor esparso `{indice: peso}` compatível com os
sparse vectors do Qdrant, de forma que o produto escalar
`query_vector . doc_vector` reproduz o score BM25:

* lado do documento: tf saturado por k1 e normalizado pelo comprimento (b);
* lado da consulta: IDF do termo.

Os índices vêm de hashing estável do token (sem vocabulário fixo), e as
estatísticas do corpus (df, número de documentos, comprimento médio) são
persistidas em JSON para que a busca seja idêntica entre processos.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path

from enterprise_rag.config.logging import get_logger
from enterprise_rag.config.settings import Settings
from enterprise_rag.models.chunk import SparseVector

logger = get_logger(__name__)

K1 = 1.5
B = 0.75
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*", re.UNICODE)

# Stopwords pt-BR + en. Curta de propósito: remover demais destrói consultas
# como "o que e a NBR 6118".
_STOPWORDS_RAW = (
    """
    a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob
    e ou mas que qual quais quando onde como porque pois se ao aos à às pelo pela
    pelos pelas este esta estes estas esse essa esses essas isso isto aquilo seu sua
    seus suas nosso nossa é sao são foi eram ser sendo ter tem têm há mais menos muito
    muita pouco entre sobre também já não sim qualquer cada todos todas
    the a an of in on at to for from by with without and or but is are was were be been
    being this that these those it its as what which when where how why do does did
    """
)


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


# Comparacao acontece sobre texto sem acento, entao as stopwords tambem sao normalizadas.
STOPWORDS: frozenset[str] = frozenset(strip_accents(word) for word in _STOPWORDS_RAW.split())


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenização compartilhada por BM25, reranker léxico e query analyzer.

    Preserva códigos com hífen/underscore/ponto (`uhe-xpto-01`, `nbr-6118`,
    `v1.2`) como um único token: quebrá-los destruiria o sinal lexical que
    justifica a busca esparsa.
    """
    lowered = strip_accents(text.lower())
    tokens = _TOKEN_RE.findall(lowered)
    result: list[str] = []
    for token in tokens:
        if len(token) < 2 and not token.isdigit():
            continue
        if remove_stopwords and token in STOPWORDS:
            continue
        result.append(token)
    return result


def token_index(token: str) -> int:
    """Índice estável de 31 bits para um token (hashing trick, sem vocabulário)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


class BM25SparseEncoder:
    """Codificador BM25 com estatísticas de corpus persistentes."""

    def __init__(self, stats_path: Path | None = None) -> None:
        self.stats_path = stats_path
        self._lock = threading.Lock()
        self.document_frequency: Counter[str] = Counter()
        self.document_count: int = 0
        self.total_length: int = 0
        if stats_path is not None and stats_path.exists():
            self.load()

    # ── estatísticas do corpus ──────────────────────────────────────────
    @property
    def average_length(self) -> float:
        if self.document_count == 0:
            return 1.0
        return max(1.0, self.total_length / self.document_count)

    def fit(self, texts: list[str]) -> None:
        """Incorpora novos documentos às estatísticas (ingestão incremental)."""
        with self._lock:
            for text in texts:
                tokens = tokenize(text)
                self.document_count += 1
                self.total_length += len(tokens)
                for token in set(tokens):
                    self.document_frequency[token] += 1

    def forget(self, texts: list[str]) -> None:
        """Remove documentos das estatísticas (usado ao deletar um documento)."""
        with self._lock:
            for text in texts:
                tokens = tokenize(text)
                self.document_count = max(0, self.document_count - 1)
                self.total_length = max(0, self.total_length - len(tokens))
                for token in set(tokens):
                    if self.document_frequency[token] <= 1:
                        del self.document_frequency[token]
                    else:
                        self.document_frequency[token] -= 1

    def idf(self, token: str) -> float:
        """IDF do BM25 (variante Lucene, sempre positiva)."""
        df = self.document_frequency.get(token, 0)
        n = max(self.document_count, 1)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    # ── codificação ─────────────────────────────────────────────────────
    def encode_document(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        if not tokens:
            return SparseVector()
        counts = Counter(tokens)
        length = len(tokens)
        norm = K1 * (1 - B + B * length / self.average_length)
        indices: list[int] = []
        values: list[float] = []
        for token, tf in counts.items():
            weight = tf * (K1 + 1) / (tf + norm)
            indices.append(token_index(token))
            values.append(round(weight, 6))
        return SparseVector(indices=indices, values=values)

    def encode_query(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        if not tokens:
            return SparseVector()
        indices: list[int] = []
        values: list[float] = []
        for token in dict.fromkeys(tokens):
            weight = self.idf(token)
            if weight <= 0:
                continue
            indices.append(token_index(token))
            values.append(round(weight, 6))
        return SparseVector(indices=indices, values=values)

    # ── persistência ────────────────────────────────────────────────────
    def save(self) -> None:
        if self.stats_path is None:
            return
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "document_count": self.document_count,
            "total_length": self.total_length,
            "document_frequency": dict(self.document_frequency),
        }
        tmp = self.stats_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.stats_path)

    def load(self) -> None:
        if self.stats_path is None or not self.stats_path.exists():
            return
        try:
            payload = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("estatisticas BM25 corrompidas; recomecando do zero")
            return
        self.document_count = int(payload.get("document_count", 0))
        self.total_length = int(payload.get("total_length", 0))
        self.document_frequency = Counter(payload.get("document_frequency", {}))


def build_sparse_encoder(settings: Settings) -> BM25SparseEncoder:
    return BM25SparseEncoder(stats_path=settings.sparse_stats_path)
