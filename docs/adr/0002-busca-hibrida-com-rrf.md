# ADR 0002 — Busca híbrida com Reciprocal Rank Fusion

**Status:** aceito

## Contexto

Busca densa erra em códigos, siglas e números de norma: `UHE-SRA-01` e
`UHE-SRA-02` produzem vetores quase idênticos. Busca lexical erra em
paráfrase: "potência instalada" não casa com "capacidade de geração".
Documentos técnicos têm as duas coisas na mesma frase.

## Decisão

Executar as duas buscas e fundir por Reciprocal Rank Fusion com `k = 60`.

## Alternativas consideradas

**Só denso.** Mais simples, e falha exatamente nas consultas mais específicas
-- as que mais aparecem em acervo corporativo.

**Soma ponderada de scores normalizados.** Cosseno e BM25 não são comparáveis;
BM25 não tem limite superior e depende do corpus. Normalizar exigiria
calibração por consulta e continuaria instável quando uma das listas voltasse
vazia.

**Reranker resolvendo tudo.** O reranker só reordena o que foi recuperado. Se o
chunk certo não está no top-30, nenhum cross-encoder o traz de volta.

## Consequências

* Uma busca a mais por consulta (~60 ms no store em memória).
* Estatísticas do BM25 precisam ser persistidas e mantidas incrementalmente,
  inclusive na remoção de documentos.
* O sistema fica robusto aos dois modos de falha, e `RAG_USE_HYBRID=false`
  permite medir exatamente quanto isso vale (`scripts/benchmark.py`).
