# Arquitetura

## Visão geral

```
                         ┌─────────────────┐
                         │     Usuário     │
                         └────────┬────────┘
                                  ▼
                         ┌─────────────────┐
                         │     FastAPI     │
                         │ /ask /search    │
                         │ /documents      │
                         └────────┬────────┘
                                  ▼
                         ┌─────────────────┐
                         │ RAG Orchestrator│
                         └────────┬────────┘
                                  ▼
                    ┌─────────────────────────┐
                    │     Query Processing     │
                    │ rewrite / filtros / intenção │
                    └─────────────┬───────────┘
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
                         │  Ollama (LLM)   │
                         └────────┬────────┘
                                  ▼
                       resposta + citações + trace
```

Pipeline de preparação, executado fora do caminho da consulta:

```
PDF / DOCX / TXT / MD / XLSX / CSV
          │
          ▼
       Parser  ──────► blocos (texto + página + seção + tipo)
          │
          ▼
   Normalização (NFKC, hífen de quebra, espaços)
          │
          ▼
 Extração de metadados (caminho, frontmatter, sidecar)
          │
          ▼
      Chunking (fixed | structural)
          │
     ┌────┴────┐
     ▼         ▼
 dense emb.  sparse emb. (BM25)
     │         │
     └────┬────┘
          ▼
   Qdrant (vetores nomeados) + SQLite (catálogo)
```

## Camadas

| Camada | Pacote | Responsabilidade |
| --- | --- | --- |
| Configuração | `config/` | `Settings` (env `RAG_*`) e logging estruturado |
| Modelos | `models/` | `Chunk`, `Document`, contratos de resposta |
| Ingestão | `ingestion/` | parsers, chunking, metadados, deduplicação, pipeline |
| Embeddings | `embeddings/` | denso (sentence-transformers/hashing) e esparso (BM25) |
| Armazenamento | `storage/` | vector store (Qdrant/memória) e catálogo SQLite |
| Recuperação | `retrieval/` | denso, esparso, fusão RRF, filtros, reranking |
| Orquestração | `orchestration/` | análise da consulta, contexto, fluxo RAG |
| Geração | `generation/` | cliente Ollama, prompts, detecção de "não encontrado" |
| Avaliação | `evaluation/` | métricas de recuperação e geração, portão de regressão |
| Observabilidade | `observability/` | tracing por estágio e métricas Prometheus |

`container.py` é o *composition root*: API, CLIs e dashboard usam o mesmo
objeto montado no mesmo lugar. Isso não é cerimônia -- é o que garante que o
número medido pela avaliação descreve o sistema que roda em produção.

> Diferença em relação ao esboço original: `storage/` e `container.py` não
> estavam previstos. Foram extraídos quando ficou claro que o vector store
> precisava de duas implementações e que quatro pontos de entrada montavam o
> mesmo grafo de objetos.

## Decisões estruturais

**Núcleo de retrieval escrito à mão, sem LangChain/LlamaIndex.** A fusão RRF, o
funil de reranking e a montagem de contexto são as partes que explicam a
qualidade do sistema. Escondê-las atrás de uma abstração de terceiros tornaria
impossível justificar cada escolha, e é justamente isso que este projeto
pretende demonstrar. Bibliotecas continuam bem-vindas nas bordas (parsing,
embeddings, cliente HTTP).

**Todo backend tem duas implementações.** Vector store (Qdrant | memória),
embeddings (sentence-transformers | hashing), reranker (cross-encoder |
léxico), LLM (Ollama | extrativo). Os pares "leves" existem para que o sistema
inteiro rode em CI gratuito e em qualquer laptop -- e, de quebra, provam que as
interfaces realmente isolam as dependências. Trocar Qdrant por Azure AI Search
ou Ollama por Foundry é escrever uma terceira classe.

**Retrieval e generation são endpoints separados.** Ver [retrieval.md](retrieval.md).

**Falha degrada, não derruba.** Qdrant fora do ar cai para o store em memória
com aviso; `sentence-transformers` ausente cai para hashing; LLM fora do ar
devolve 503 em `/ask` mas mantém `/search` funcionando. O `/health` mostra o
estado real de cada dependência em vez de um booleano único.

## Fluxo de uma pergunta

1. `query_processing` -- filtros inferidos, consulta reescrita por canal.
2. `retrieval` -- busca densa e esparsa em paralelo lógico, fusão por RRF.
3. `reranking` -- cross-encoder reduz 30 candidatos a 6.
4. `context_build` -- orçamento de tokens, teto por documento, deduplicação.
5. `generation` -- prompt ancorado, com marcador de "não encontrado".
6. resposta -- texto, citações resolvidas para `chunk_id`/página, e o trace.

Cada etapa é um span; as durações voltam no campo `metrics` da resposta e
alimentam `rag_stage_latency_seconds`.
