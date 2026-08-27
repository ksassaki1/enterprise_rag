# Recuperação

## Por que híbrido

Pergunta real de um acervo técnico:

> Qual a potência instalada do empreendimento UHE-SRA-01?

Ela tem duas metades, e cada uma quebra um dos dois métodos:

| Metade | Denso | Esparso (BM25) |
| --- | --- | --- |
| "potência instalada" ~ "capacidade de geração" | acerta | erra (vocabulário diferente) |
| `UHE-SRA-01` vs `UHE-SRA-02` | erra (vetores quase idênticos) | acerta (IDF altíssimo) |

Um embedding de 384 dimensões não tem incentivo para separar dois códigos que
diferem em um dígito. O BM25 separa. O inverso também vale: BM25 não sabe que
"capacidade instalada" e "potência instalada" são a mesma coisa.

Rodar os dois e fundir custa uma busca a mais e resolve as duas falhas.

## Fusão: RRF, não soma de scores

Cosseno vive em `[0, 1]`; BM25 vive em `[0, +inf)` e depende do corpus.
Normalizar exigiria calibração por consulta e continuaria instável quando uma
das listas voltasse vazia. O RRF usa apenas **posição**:

```
score(d) = Σ_listas  peso / (k + rank_lista(d))
```

com `k = 60` (valor do artigo original de Cormack et al., 2009). O efeito
prático: aparecer em ambas as listas vale mais do que ser o primeiro em uma só
-- concordância entre dois recuperadores independentes é o sinal mais forte
disponível.

```
Denso              Esparso            RRF (k=60)
1. A               1. C               1. A   (1/61 + 1/62)
2. B               2. A               2. C   (1/63 + 1/61)
3. C               3. D               3. B
                                      4. D
```

Implementação: `retrieval/hybrid.py`. Testes: `tests/unit/test_hybrid.py`.

## BM25 como vetor esparso

O BM25 é implementado à mão (`embeddings/sparse.py`) e emitido no formato de
sparse vector do Qdrant, de modo que o produto escalar
`query · documento` reproduz o score:

* documento: `tf · (k1 + 1) / (tf + k1 · (1 - b + b · |d| / avgdl))`
* consulta: `idf(termo)`

com `k1 = 1.5`, `b = 0.75`. Os índices vêm de hashing estável do token (sem
vocabulário fixo); `df`, número de documentos e comprimento médio são
persistidos em `data/sparse_stats.json` e atualizados de forma incremental na
ingestão -- inclusive no `forget()`, quando um documento é removido.

O tokenizador preserva códigos com hífen, underscore e ponto
(`uhe-sra-01`, `nbr-6118`, `v1.2`) como um único token. Quebrá-los destruiria
exatamente o sinal que justifica ter um canal lexical.

## Filtro de metadados

Filtrar antes do vetor é mais barato e mais preciso do que filtrar depois:

```
"Quais barragens brasileiras têm mais de 50 metros?"
        │
        ▼
{"country": "Brazil"}          ← inferido por regra
        │
        ▼
busca vetorial  WHERE metadata.country == "Brazil"
```

Sem o filtro, metade do top-30 é gasto com documentos de outro país e nunca
chega ao reranker. `retrieval/filters.py` avalia o mesmo conjunto de condições
em memória e o traduz para o `Filter` nativo do Qdrant; `QdrantVectorStore`
cria os índices de payload dos campos filtráveis -- sem índice, o filtro vira
varredura e o ganho desaparece.

Filtro passado pela API sempre vence o inferido: inferência erra, e um filtro
errado descarta silenciosamente o documento certo.

## Reranking

```
30 candidatos → cross-encoder → 6 chunks
```

Recuperação e reranking resolvem problemas diferentes:

| | Recuperação | Reranking |
| --- | --- | --- |
| Como compara | vetores pré-computados | lê consulta e chunk juntos |
| Custo | ~milissegundos sobre milhões | ~200 ms sobre dezenas |
| Otimiza | recall | precisão |

Rodar o cross-encoder sobre a coleção inteira seria inviável; sobre 30
candidatos ele costuma ser a maior melhoria de qualidade por linha de código do
projeto. O preço é latência -- por isso `scripts/benchmark.py` imprime
qualidade e P50 lado a lado, e existe a configuração `structural-no-rerank`
para medir exatamente quanto o reranker entrega.

`rerank_min_score` define um piso: candidato abaixo dele é descartado, o que
alimenta a detecção de "não encontrado" em vez de forçar uma resposta ruim.

## `/search` separado de `/ask`

Este é o endpoint mais útil do sistema em produção, e o mais ignorado em
tutoriais. Quando uma resposta sai errada, existem exatamente duas hipóteses:

```
resposta errada
      │
      ├── o chunk certo foi recuperado?  → não → problema de RETRIEVAL
      │                                          (chunking, embedding, filtro, top-k)
      └──                                 sim → problema de GERAÇÃO
                                                (prompt, contexto, modelo)
```

`POST /search` responde essa pergunta em uma requisição, sem gastar um token de
LLM, e devolve os scores de cada estágio (`dense_score`, `sparse_score`,
`fusion_score`, `rerank_score`) para que dê para ver *em que ponto* o chunk
certo caiu no ranking.

## Parâmetros que importam

| Parâmetro | Padrão | Efeito |
| --- | --- | --- |
| `RAG_TOP_K_CANDIDATES` | 30 | recall bruto; alimenta o reranker |
| `RAG_TOP_K_FINAL` | 6 | precisão e tamanho do contexto |
| `RAG_RRF_K` | 60 | quanto o topo de cada lista pesa |
| `RAG_DENSE_WEIGHT` / `RAG_SPARSE_WEIGHT` | 1.0 / 1.0 | equilíbrio semântico ↔ lexical |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | 500 / 60 | granularidade da recuperação |
| `RAG_MAX_CHUNKS_PER_DOCUMENT` | 3 | impede um documento de dominar o contexto |

Nenhum desses números é sagrado. Meça antes de mudar:
`python scripts/benchmark.py`.
