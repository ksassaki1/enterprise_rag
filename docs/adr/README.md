# Architecture Decision Records

Registro curto de cada decisão relevante: o contexto, a escolha e o que ela
custa. O objetivo não é documentação cerimonial -- é conseguir responder, meses
depois ou em uma entrevista, *por que* o sistema é assim e o que a alternativa
teria custado.

| ADR | Decisão |
| --- | --- |
| [0001](0001-core-de-retrieval-proprio.md) | Núcleo de retrieval próprio, sem framework de RAG |
| [0002](0002-busca-hibrida-com-rrf.md) | Busca híbrida com Reciprocal Rank Fusion |
| [0003](0003-qdrant-como-vector-store.md) | Qdrant como vector store, atrás de uma interface |
| [0004](0004-avaliacao-como-portao-de-ci.md) | Avaliação como portão de CI |
| [0005](0005-backends-deterministicos.md) | Backends determinísticos para testes e CI |
