# ADR 0001 — Núcleo de retrieval próprio, sem framework de RAG

**Status:** aceito

## Contexto

LangChain e LlamaIndex resolvem em poucas linhas o que aqui ocupa alguns
módulos: carregar documento, dividir, indexar, recuperar, responder.

## Decisão

Escrever o núcleo -- chunking, fusão híbrida, reranking, montagem de contexto,
prompt e citação -- diretamente, sem framework de RAG como dependência central.
Bibliotecas continuam sendo usadas nas bordas: PyMuPDF, python-docx,
sentence-transformers, qdrant-client, httpx.

## Justificativa

O valor deste projeto está em conseguir explicar cada decisão: por que RRF e
não soma de scores, por que 30 candidatos antes do reranker, por que separar
`/search` de `/ask`, o que acontece com o orçamento de contexto quando um
documento domina o top-k. Um framework esconde exatamente essas escolhas.

Além disso, a abstração de terceiros teria custado o que ela sempre custa:
depurar através de camadas, versões incompatíveis e limites que aparecem tarde.

## Consequências

* Mais código para manter (~2.500 linhas de núcleo).
* Cada componente é testável isoladamente, sem mocks de framework.
* Trocar Qdrant/Ollama por AI Search/Foundry é implementar uma interface, não
  migrar de framework.
* Perde-se acesso imediato ao ecossistema de integrações prontas. Aceitável:
  o escopo é documentos corporativos em disco.
