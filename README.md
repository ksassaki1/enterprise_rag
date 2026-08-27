# Enterprise RAG Platform

[![CI](https://github.com/your-user/enterprise-rag-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/your-user/enterprise-rag-platform/actions/workflows/ci.yml)
[![RAG Evaluation](https://github.com/your-user/enterprise-rag-platform/actions/workflows/evaluation.yml/badge.svg)](https://github.com/your-user/enterprise-rag-platform/actions/workflows/evaluation.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Plataforma local-first de RAG corporativo: ingestão de documentos,
recuperação híbrida, reranking, geração ancorada com citações, avaliação e
observabilidade.**

Roda inteira na sua máquina. Sem chave de API, sem serviço pago, sem dado
saindo da rede.

```
✓ Local-first, zero dependência de nuvem paga
✓ Recuperação híbrida (densa + BM25) com fusão RRF
✓ Filtro por metadados aplicado antes do vetor
✓ Reranking por cross-encoder (30 candidatos → 6 chunks)
✓ Indexação incremental por hash, com versionamento de documento
✓ Respostas com citação rastreável até arquivo, página e chunk
✓ Detecção explícita de "não encontrado"
✓ Pipeline de avaliação com golden dataset (Recall@K, MRR, nDCG, accuracy)
✓ Portão de regressão no CI
✓ Tracing por estágio e métricas em formato Prometheus
✓ Docker Compose, testes e CI prontos
```

---

## Por que este projeto existe

A maior parte dos projetos de RAG no GitHub é um `app.py` que lê um PDF e
conversa com ele. Este é um sistema corporativo em miniatura, e cada decisão
arquitetural aqui pode ser defendida em uma entrevista:

| Pergunta | Resposta |
| --- | --- |
| Por que busca híbrida? | [docs/retrieval.md](docs/retrieval.md#por-que-híbrido) |
| Por que RRF e não soma de scores? | [ADR 0002](docs/adr/0002-busca-hibrida-com-rrf.md) |
| Por que reranking, se já há top-k? | [docs/retrieval.md](docs/retrieval.md#reranking) |
| Por que separar retrieval de generation? | [docs/retrieval.md](docs/retrieval.md#search-separado-de-ask) |
| Como se mede recall aqui? | [docs/evaluation.md](docs/evaluation.md#métricas) |
| Como se detecta regressão? | [ADR 0004](docs/adr/0004-avaliacao-como-portao-de-ci.md) |
| Como atualizar um documento? | [Ingestão incremental](#ingestão-incremental) |
| Como trocar Qdrant/Ollama por AI Search/Foundry? | [ADR 0003](docs/adr/0003-qdrant-como-vector-store.md) |

---

## Arquitetura

```
                         ┌─────────────────┐
                         │     Usuário     │
                         └────────┬────────┘
                                  ▼
                         ┌─────────────────┐
                         │     FastAPI     │
                         │ /ask  /search   │
                         │   /documents    │
                         └────────┬────────┘
                                  ▼
                         ┌─────────────────┐
                         │ RAG Orchestrator│
                         └────────┬────────┘
                                  ▼
                   ┌──────────────────────────┐
                   │     Query Processing      │
                   │ rewrite / filtros / intent│
                   └──────────────┬───────────┘
                                  ▼
                         ┌─────────────────┐
                         │    Retriever    │
                         │  dense + sparse │
                         └────────┬────────┘
                              top 30
                                  ▼
                         ┌─────────────────┐
                         │    Reranker     │
                         └────────┬────────┘
                               top 6
                                  ▼
                         ┌─────────────────┐
                         │ Context Builder │
                         └────────┬────────┘
                                  ▼
                         ┌─────────────────┐
                         │  Ollama  (LLM)  │
                         └────────┬────────┘
                                  ▼
                    resposta + citações + trace
```

Pipeline de ingestão, fora do caminho da consulta:

```
PDF / DOCX / TXT / MD / XLSX / CSV
      │
      ▼
   Parser ──► Normalização ──► Metadados ──► Chunking
                                                │
                                       ┌────────┴────────┐
                                       ▼                 ▼
                                  dense emb.        sparse emb. (BM25)
                                       └────────┬────────┘
                                                ▼
                                   Qdrant + catálogo SQLite
```

Detalhes: [docs/architecture.md](docs/architecture.md).

---

## Stack

| Componente | Tecnologia | Alternativa determinística (CI) |
| --- | --- | --- |
| API | FastAPI + Uvicorn | — |
| LLM | Ollama (`qwen2.5:7b-instruct`) | gerador extrativo |
| Embeddings densos | sentence-transformers (`multilingual-e5-small`) | hashing trick |
| Busca esparsa | BM25 próprio, como sparse vector | mesmo |
| Vector DB | Qdrant (vetores nomeados) | store em memória (numpy) |
| Reranking | cross-encoder (`ms-marco-MiniLM-L-6-v2`) | reranker léxico |
| Metadados | SQLite (mesmo schema em PostgreSQL) | mesmo |
| PDF / DOCX / XLSX | PyMuPDF / python-docx / openpyxl | — |
| OCR | Tesseract (fallback para PDF digitalizado) | — |
| Observabilidade | tracing próprio + OpenTelemetry opcional | mesmo |
| Dashboard | Streamlit | — |
| Containers | Docker Compose | — |
| Testes / CI | pytest + ruff + mypy + GitHub Actions | — |

Cada dependência pesada tem um par leve e determinístico: é isso que permite
rodar o pipeline inteiro, com avaliação, em CI gratuito ([ADR 0005](docs/adr/0005-backends-deterministicos.md)).

---

## Começando

### Opção A — Docker Compose (completo)

```bash
git clone https://github.com/your-user/enterprise-rag-platform.git
cd enterprise-rag-platform
cp .env.example .env

docker compose up -d --build
docker compose --profile setup run --rm ollama-pull     # baixa o modelo (~4 GB)

docker compose exec api python scripts/ingest.py ./datasets/sample_documents
```

Sobe `qdrant`, `ollama`, `api` e `dashboard`:

| Serviço | URL |
| --- | --- |
| API (Swagger) | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| Qdrant | http://localhost:6333/dashboard |
| Métricas | http://localhost:8000/metrics |

### Opção B — Local, sem Docker e sem baixar modelo

Roda o sistema inteiro com os backends determinísticos. Serve para explorar o
código, rodar os testes e ver o pipeline funcionando em segundos:

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

export RAG_VECTOR_BACKEND=memory RAG_EMBEDDING_BACKEND=hashing \
       RAG_RERANKER_BACKEND=lexical RAG_LLM_BACKEND=echo

python scripts/ingest.py ./datasets/sample_documents
python scripts/evaluate.py
uvicorn apps.api.main:app --reload
```

### Opção C — Local com qualidade real

```bash
pip install -e ".[dev,ml,vectordb,parsers,dashboard]"
docker compose up -d qdrant ollama
ollama pull qwen2.5:7b-instruct

cp .env.example .env      # já vem apontando para qdrant + ollama
python scripts/ingest.py ./datasets/sample_documents
make api
```

---

## Usando a API

### `POST /ask` — RAG completo

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{
  "question": "Qual a altura maxima da barragem da Usina Serra Azul?"
}' | jq
```

```jsonc
{
  "trace_id": "9f2c1a...",
  "question": "Qual a altura maxima da barragem da Usina Serra Azul?",
  "answer": "A altura máxima da barragem é de 76 metros, medida entre a fundação e a crista. [FONTE 1]",
  "answer_found": true,
  "sources": [
    {
      "document": "relatorio-tecnico-2024.md",
      "document_id": "brazil/usina-serra-azul/relatorio-tecnico-2024",
      "chunk_id": "brazil/usina-serra-azul/relatorio-tecnico-2024#0003",
      "page": null,
      "section": "Características da barragem",
      "score": 0.912,
      "excerpt": "A barragem é do tipo gravidade em concreto compactado a rolo...",
      "used_in_answer": true
    }
  ],
  "retrieval": {
    "dense_candidates": 30,
    "sparse_candidates": 28,
    "fused_candidates": 30,
    "reranked": 6,
    "filters_applied": {},
    "strategy": "hybrid"
  },
  "context": {
    "chunks_in_context": 6,
    "context_tokens": 1284,
    "documents_in_context": 3,
    "dropped_duplicates": 1,
    "dropped_by_document_cap": 2
  },
  "metrics": {
    "total_latency_ms": 1834.2,
    "query_processing_ms": 0.4,
    "retrieval_latency_ms": 213.1,
    "rerank_latency_ms": 208.7,
    "context_build_ms": 9.8,
    "generation_latency_ms": 1402.2
  },
  "model": "qwen2.5:7b-instruct"
}
```

Nunca só `{"answer": "76 metros"}`. Proveniência e latência por estágio vêm
junto, porque é isso que torna o sistema depurável.

### `POST /search` — só recuperação

O endpoint mais útil em produção: responde se o chunk certo foi recuperado,
sem gastar um token de LLM.

```bash
curl -s localhost:8000/search -H 'content-type: application/json' -d '{
  "query": "potencia instalada",
  "top_k": 5,
  "filters": {"country": "Brazil"}
}' | jq '.results[] | {document, section, dense_score, sparse_score, rerank_score}'
```

Resposta errada? A pergunta é sempre a mesma: **foi o retrieval ou foi o LLM?**
`/search` responde em uma requisição.

### Endpoints

| Método | Rota | Descrição |
| --- | --- | --- |
| `POST` | `/ask` | RAG completo com citações e métricas |
| `POST` | `/search` | apenas recuperação, com scores de cada estágio |
| `POST` | `/documents` | upload e ingestão (aceita metadados como sidecar) |
| `GET` | `/documents` | inventário do que está indexado |
| `GET` | `/documents/{id}` | detalhe, versão e contagem de chunks |
| `DELETE` | `/documents/{id}` | remove do índice, do catálogo e das estatísticas BM25 |
| `POST` | `/evaluate` | roda o golden dataset contra a configuração em execução |
| `GET` | `/health` | estado de cada dependência e configuração efetiva |
| `GET` | `/metrics` | exposição no text format do Prometheus |
| `GET` | `/stats` | índice, consultas recentes, P50/P95 |

---

## O objeto fundamental: `Chunk`

Tudo circula como `Chunk`. Schema canônico é o que permite filtro, citação e
deduplicação sem gambiarra depois:

```python
class Chunk:
    chunk_id: str          # brazil/usina-serra-azul/relatorio-2024#0042
    document_id: str       # brazil/usina-serra-azul/relatorio-2024
    content: str
    source: str
    filename: str
    file_type: str
    page: int | None       # sem isso não existe citação verificável
    section: str | None
    position: int
    created_at: datetime
    metadata: dict         # country, project, document_type, year, ...
    content_hash: str      # deduplicação
    token_count: int       # orçamento de contexto
```

---

## Ingestão incremental

```
relatorio.pdf
   hash anterior: abc123
   hash atual:    abc123   →  não reprocessa

relatorio.pdf (editado)
   hash anterior: abc123
   hash atual:    def456   →  apaga chunks da v1, indexa v2, version += 1
```

Em acervo corporativo, reindexar tudo a cada execução é o custo dominante e a
causa mais comum de "o RAG está caro". O SHA-256 por arquivo resolve isso, e o
`document_id` derivado do caminho garante que a substituição seja idempotente:
nenhuma resposta cita uma versão que não existe mais.

Metadados vêm de três fontes, em ordem crescente de prioridade: convenção de
caminho (`brazil/...` → `country=Brazil`), frontmatter do documento e sidecar
explícito (`arquivo.pdf.meta.json` ou `_metadata.json` do diretório).

---

## Avaliação

```bash
python scripts/evaluate.py
```

```
==============================================================
RAG EVALUATION
==============================================================
Dataset            golden_dataset.jsonl  (16 perguntas)
Embeddings         intfloat/multilingual-e5-small
Chunking           structural (500/60)
Reranker           cross-encoder/ms-marco-MiniLM-L-6-v2
LLM                qwen2.5:7b-instruct

Retrieval                    Generation
  Recall@1         ...         Corretas         ...
  Recall@5         ...         Incorretas       ...
  Recall@10        ...         Nao encontrado   ...
  MRR              ...         Accuracy         ...
  nDCG@10          ...         Citacao valida   ...

Performance
  P50              ...s
  P95              ...s
==============================================================
```

> Os números não vêm preenchidos aqui de propósito: eles dependem da sua
> máquina, do seu modelo e do seu corpus. Publicar números inventados
> destruiria exatamente o argumento deste projeto. Rode e preencha os seus.

### Benchmark comparativo

```bash
python scripts/benchmark.py
```

Indexa o mesmo corpus com configurações diferentes, cada uma isolada, e
compara `fixed-500`, `fixed-1000`, `structural`, `structural-no-rerank` e
`structural-dense-only`. A afirmação defensável não é "estrutural é melhor" --
é **"eu medi"**.

### Portão de regressão

```bash
python scripts/evaluate.py --update-baseline           # grava reports/baseline.json
python scripts/evaluate.py --baseline reports/baseline.json
```

```
git push
   ↓
lint + mypy
   ↓
pytest (unit + integration + evaluation)
   ↓
docker build
   ↓
ingest + evaluate
   ↓
comparação com baseline   →  falha se Recall@10 cair além de 0.03
   ↓
comentário no PR com a tabela
```

Mudar `chunk_size` de 500 para 800 deixa de ser achismo e passa a ser um número
no diff. Detalhes: [docs/evaluation.md](docs/evaluation.md).

---

## Observabilidade

Cada pergunta gera um `trace_id` e uma árvore de spans:

```
trace_id 9f2c1a...
   ├── query_processing      0.4 ms
   ├── retrieval           213.1 ms
   ├── reranking           208.7 ms
   ├── context_build         9.8 ms
   └── generation         1402.2 ms
```

Métricas expostas em `/metrics` (text format do Prometheus):

```
rag_requests_total{endpoint,status}
rag_latency_seconds{endpoint}
rag_stage_latency_seconds{stage,endpoint}
rag_retrieved_chunks{endpoint}
rag_context_tokens
rag_answer_not_found_total
rag_documents_indexed_total{file_type}
rag_indexed_chunks
```

Com `RAG_OTEL_ENABLED=true` os mesmos spans saem via OTLP, sem mudar o código
de negócio.

---

## Estrutura do repositório

```
enterprise-rag-platform/
├── apps/
│   ├── api/main.py                 FastAPI: /ask /search /documents /metrics
│   └── dashboard/app.py            Streamlit operacional
├── src/enterprise_rag/
│   ├── config/                     Settings (env RAG_*) e logging estruturado
│   ├── models/                     Chunk, Document, contratos de resposta
│   ├── ingestion/
│   │   ├── parsers/                pdf, docx, txt/md, xlsx/csv (+ OCR)
│   │   ├── chunker.py              fixed | structural
│   │   ├── metadata.py             caminho + frontmatter + sidecar
│   │   ├── deduplication.py        SHA-256, hash de chunk, Jaccard
│   │   └── pipeline.py             ingestão incremental e idempotente
│   ├── embeddings/                 denso (ST/hashing) + esparso (BM25)
│   ├── storage/                    Qdrant | memória + catálogo SQLite
│   ├── retrieval/                  dense, sparse, hybrid (RRF), filtros, reranker
│   ├── orchestration/              query analyzer, context builder, RAG
│   ├── generation/                 cliente Ollama, prompts, "não encontrado"
│   ├── evaluation/                 métricas, runner, portão de regressão
│   ├── observability/              tracing por estágio, métricas Prometheus
│   └── container.py                composition root
├── scripts/                        ingest.py, evaluate.py, benchmark.py
├── tests/                          unit, integration, evaluation
├── datasets/                       documentos sintéticos + golden dataset
├── docs/                           arquitetura, retrieval, avaliação, segurança, ADRs
├── reports/                        baseline.json versionado
├── docker-compose.yml
└── .github/workflows/              ci.yml, evaluation.yml
```

---

## Desenvolvimento

```bash
make install-dev     # instala com extras de dev e ML
make lint            # ruff
make typecheck       # mypy
make test            # pytest: unit + integration + evaluation
make ingest          # indexa os documentos de exemplo
make evaluate        # golden dataset + comparação com baseline
make benchmark       # compara estratégias
make api             # uvicorn com reload
make dashboard       # streamlit
```

A suíte roda com os backends determinísticos: sem GPU, sem download de modelo,
sem serviço externo. Os testes de integração exercitam o pipeline completo --
ingestão, busca híbrida, filtro, reranking, contexto, citação e a API HTTP.

### Configuração

Toda a plataforma é parametrizada por variáveis `RAG_*` (veja
[.env.example](.env.example)). As que mais importam:

| Variável | Padrão | Efeito |
| --- | --- | --- |
| `RAG_VECTOR_BACKEND` | `qdrant` | `qdrant` ou `memory` |
| `RAG_EMBEDDING_BACKEND` | `sentence-transformers` | ou `hashing` |
| `RAG_RERANKER_BACKEND` | `cross-encoder` | `lexical` ou `none` |
| `RAG_LLM_BACKEND` | `ollama` | ou `echo` |
| `RAG_CHUNK_STRATEGY` | `structural` | ou `fixed` |
| `RAG_TOP_K_CANDIDATES` | `30` | candidatos antes do reranker |
| `RAG_TOP_K_FINAL` | `6` | chunks que entram no contexto |
| `RAG_MAX_CHUNKS_PER_DOCUMENT` | `3` | impede um documento de dominar |

---

## Roadmap

O projeto foi construído em incrementos, e cada um deles é uma versão que
funciona sozinha:

| Versão | Escopo | Estado |
| --- | --- | --- |
| **v0.1** Foundation | FastAPI, Ollama, Qdrant, Docker, PDF → chunks → embeddings → resposta | concluído |
| **v0.2** Retrieval | denso + esparso + fusão RRF + filtro por metadados | concluído |
| **v0.3** Quality | reranking, context builder, deduplicação, "não encontrado", citações | concluído |
| **v0.4** Evaluation | golden dataset, Recall@K, MRR, nDCG, accuracy, latência, benchmark | concluído |
| **v0.5** Productionization | logging estruturado, tracing, métricas, healthcheck, retry, timeout, CI, portão de regressão | concluído |
| **v1.0** Enterprise | RBAC, coleções, multi-tenant, auditoria completa, versionamento com histórico, juiz-LLM fora do portão | planejado |

Lacunas conhecidas (e deliberadas) estão listadas em
[docs/security.md](docs/security.md#superfície-atual): não há autenticação nem
autorização por documento. É uma plataforma de referência, não um produto
endurecido -- e esconder isso seria o pior problema de segurança do repositório.

---

## Documentos de exemplo

Todo o conteúdo de `datasets/sample_documents/` é **sintético e fictício**:
uma usina hidrelétrica inventada, uma planta no Chile, uma política de
segurança e um procedimento de manutenção, todos criados para este repositório.
Nenhum dado corresponde a empreendimento, organização ou pessoa real, e nada
foi derivado de material proprietário de terceiros.

---

## Licença

[MIT](LICENSE).
