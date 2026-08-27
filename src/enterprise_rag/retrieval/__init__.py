"""Camada de recuperação.

Sem imports no `__init__` de propósito: `storage` depende de
`retrieval.filters`, e importar os retrievers aqui criaria um ciclo.
Importe sempre o módulo específico (`from enterprise_rag.retrieval.hybrid import ...`).
"""
