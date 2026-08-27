# Avaliação

## O princípio

A afirmação forte deste projeto não é "usei chunking estrutural e reranking".
É:

> Eu medi as alternativas no mesmo dataset, com o mesmo código que roda em
> produção, e o CI reprova o PR que piorar o número.

## Golden dataset

`datasets/golden_dataset.jsonl`, uma pergunta por linha:

```json
{
  "id": "Q002",
  "question": "Qual a altura maxima da barragem da Usina Serra Azul?",
  "expected_answer": "altura maxima da barragem e de 76 metros",
  "expected_documents": ["brazil/usina-serra-azul/relatorio-tecnico-2024"],
  "tags": ["numerico", "barragem"]
}
```

16 perguntas escritas à mão sobre os documentos sintéticos, cobrindo:
valores numéricos, códigos de ativo, limites operacionais, políticas, filtro
por país e **uma pergunta sem resposta na base** (`Q016`, `unanswerable`) --
essa existe para punir o sistema que inventa uma resposta plausível.

Dataset pequeno e honesto vale mais que grande e automático: cada linha foi
escrita olhando o documento.

## Métricas

### Recuperação (`evaluation/retrieval.py`)

| Métrica | Pergunta que responde |
| --- | --- |
| Recall@K | o documento certo chegou ao LLM? |
| Precision@5 | quanto lixo entrou no contexto? |
| MRR | quão no topo ele chegou? |
| nDCG@10 | e quando há mais de um documento correto? |

Recall@K é a métrica que governa `top_k_candidates`. Se ela está baixa, mexer
no prompt é perder tempo.

### Geração (`evaluation/generation.py`)

| Métrica | O que mede |
| --- | --- |
| accuracy (containment) | os termos da resposta esperada aparecem? |
| token F1 | sobreposição parcial, para respostas longas |
| not-found rate | com que frequência o sistema admite não saber |
| valid citation rate | a citação aponta para o documento certo? |

Não há juiz-LLM no portão de CI, e isso é uma escolha: um juiz baseado no mesmo
modelo local traria variância, custaria minutos por execução e tornaria o
portão não determinístico. Para respostas curtas e factuais (`850 MW`,
`76 metros`, `18 meses`), verificação determinística mede o que importa em
milissegundos. Juiz-LLM é o passo natural **fora** do portão, para perguntas
dissertativas.

Uma nota sobre a not-found rate: ela não deve ser minimizada. Próxima de zero
significa que o modelo está inventando em vez de recusar; alta demais significa
recall ruim. É uma métrica de calibração, não de erro.

## Executando

```bash
python scripts/ingest.py ./datasets/sample_documents
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

Retrieval
  Recall@1         ...
  Recall@5         ...
  Recall@10        ...
  MRR              ...

Generation
  Corretas         ...
  Nao encontrado   ...
  Accuracy         ...
  Citacao valida   ...

Performance
  P50              ...s
  P95              ...s
==============================================================
```

O cabeçalho traz a configuração efetiva de propósito: uma métrica sem a
configuração que a produziu não descreve nada.

> Os números não vêm preenchidos neste repositório. Eles dependem da sua
> máquina, do seu modelo e do seu corpus -- publicar números inventados
> destruiria justamente o ponto do documento. Rode e preencha os seus.

## Benchmark comparativo

```bash
python scripts/benchmark.py
```

Indexa o mesmo corpus com configurações diferentes, cada uma em um índice
isolado, e compara:

```
configuracao              Recall@5   Recall@10     MRR   nDCG@10      P50
fixed-500                     ...         ...      ...       ...      ...s
fixed-1000                    ...         ...      ...       ...      ...s
structural                    ...         ...      ...       ...      ...s
structural-no-rerank          ...         ...      ...       ...      ...s
structural-dense-only         ...         ...      ...       ...      ...s
```

As duas últimas linhas são as mais informativas: elas medem quanto o reranker e
o canal esparso realmente entregam neste corpus, em vez de assumir que
entregam.

## Portão de regressão

```bash
python scripts/evaluate.py --update-baseline   # grava reports/baseline.json
python scripts/evaluate.py --baseline reports/baseline.json
```

Tolerâncias em `evaluation/regression.py`:

| Métrica | Tolerância | Direção |
| --- | --- | --- |
| recall@5, recall@10, mrr | 0.03 absoluto | maior é melhor |
| accuracy | 0.05 absoluto | maior é melhor |
| not_found_rate | 0.05 absoluto | menor é melhor |
| p95_latency_ms | 35% relativo | menor é melhor |

Tolerância existe porque nem toda variação é regressão (CPU vs GPU, versão de
modelo, ruído de máquina de CI). O que não pode acontecer é a qualidade cair
sem ninguém ver.

No CI (`.github/workflows/evaluation.yml`), o PR que muda `src/`, `datasets/`
ou `scripts/`:

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
comparação com baseline  → falha se regrediu
   ↓
comentário no PR com a tabela
```

Quando a mudança **melhora** o número, atualize o baseline no mesmo PR. A linha
alterada em `reports/baseline.json` é a evidência da melhoria.
