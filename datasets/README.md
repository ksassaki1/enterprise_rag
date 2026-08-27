# Datasets

## `sample_documents/`

Documentos sintéticos e fictícios usados para demonstrar a plataforma. Veja o
README daquele diretório.

## `golden_dataset.jsonl`

Conjunto de avaliação. Uma pergunta por linha:

```json
{
  "id": "Q001",
  "question": "Qual e a potencia instalada da Usina Serra Azul?",
  "expected_answer": "potencia instalada de 850 MW",
  "expected_documents": ["brazil/usina-serra-azul/relatorio-tecnico-2024"],
  "filters": {"country": "Brazil"},
  "unanswerable": false,
  "tags": ["numerico", "geracao"]
}
```

Campos:

| campo | uso |
| --- | --- |
| `expected_documents` | mede recuperação (Recall@K, MRR, nDCG). São `document_id`s -- o caminho relativo sem extensão. |
| `expected_answer` | mede geração (containment e token F1). Frase curta com os termos que a resposta precisa conter. |
| `filters` | força um filtro de metadados, para testar o caminho de filtragem. |
| `unanswerable` | pergunta sem resposta na base: acertar é **recusar**. Mede honestidade, não conhecimento. |
| `tags` | agrupamento para análise (numérico, código, política, filtro...). |

Um golden dataset pequeno e honesto vale mais que um grande e automático:
estas 16 perguntas foram escritas olhando os documentos, e a `Q016` existe
justamente para punir o sistema que inventa uma resposta plausível.

Para rodar:

```bash
python scripts/evaluate.py
```
