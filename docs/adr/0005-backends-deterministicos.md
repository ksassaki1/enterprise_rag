# ADR 0005 — Backends determinísticos para testes e CI

**Status:** aceito

## Contexto

O portão de regressão do ADR 0004 só é útil se rodar em todo PR. Runner
gratuito do GitHub Actions não tem GPU, e baixar `torch` + modelo de embedding
+ cross-encoder + modelo do Ollama a cada execução custaria dezenas de minutos
-- e traria variação entre execuções.

## Decisão

Cada dependência pesada tem um par determinístico e leve, selecionado por
variável de ambiente:

| Componente | Produção | CI / teste |
| --- | --- | --- |
| Vector store | Qdrant | `InMemoryVectorStore` (numpy + JSON) |
| Embedding denso | sentence-transformers | `HashingEmbedder` (hashing trick) |
| Reranker | cross-encoder | `LexicalReranker` (cobertura de termos) |
| LLM | Ollama | `EchoLLM` (extrativo, citando a fonte) |

## Justificativa

O ponto não é fingir qualidade: o embedder por hashing tem recall muito pior
que o E5, e o `EchoLLM` não é um modelo de linguagem. O ponto é que **todo o
resto do caminho é o mesmo código**: mesmo chunking, mesmo BM25, mesma fusão
RRF, mesmos filtros, mesmo orçamento de contexto, mesmo formato de citação.

Se um parser quebrar, se o canal esparso for desligado por engano, se o teto
por documento parar de funcionar ou se o formato de citação mudar, a suíte
acusa -- em segundos, de graça, em toda máquina.

## Consequências

* CI completo em menos de dois minutos, sem custo e sem GPU.
* Os números do CI **não** são os números de produção. O relatório sempre
  imprime a configuração avaliada, e os limiares do CI são um piso grosseiro;
  o baseline de qualidade real é gerado localmente com os backends de verdade.
* Cada interface ganhou uma segunda implementação, o que é a melhor prova de
  que as abstrações não vazam.
