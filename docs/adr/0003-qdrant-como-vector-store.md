# ADR 0003 — Qdrant como vector store, atrás de uma interface

**Status:** aceito

## Contexto

O projeto precisa de busca vetorial densa **e** esparsa, filtro por metadados
indexado, gratuito e executável localmente.

## Decisão

Qdrant como backend padrão, usando vetores nomeados (`dense` + `bm25`) no mesmo
ponto, atrás do protocolo `VectorStore` (`storage/base.py`).

## Alternativas consideradas

| Opção | Por que não |
| --- | --- |
| pgvector | Excelente para denso; busca esparsa exigiria `tsvector` e uma segunda consulta com fusão manual fora do banco |
| Chroma | Simples, mas sem sparse vectors nativos nem filtro indexado maduro |
| FAISS | Biblioteca, não banco: sem payload, sem filtro, sem persistência gerenciada |
| Elasticsearch | Resolveria os dois canais, mas pesa demais para rodar em laptop |

## Consequências

* Um único ponto por chunk carrega os dois vetores: o mesmo chunk é
  recuperável pelos dois caminhos, e não há um segundo banco para sincronizar.
* Índices de payload precisam ser criados explicitamente para os campos
  filtráveis -- sem eles o filtro vira varredura.
* Os ids precisam ser UUID ou inteiro; derivamos por `uuid5` do `chunk_id`,
  o que mantém a ingestão idempotente.
* A interface permite um segundo backend em memória, que é o que torna o CI
  gratuito viável (ADR 0005).
