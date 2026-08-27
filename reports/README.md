# Relatórios

`evaluation.json` é gerado a cada execução de `scripts/evaluate.py` e **não** é
versionado.

`baseline.json` **é** versionado: ele é o contrato de qualidade do repositório.
O workflow `evaluation.yml` compara cada PR contra ele e reprova a mudança se
Recall@10, MRR, accuracy ou P95 saírem da tolerância definida em
`src/enterprise_rag/evaluation/regression.py`.

O baseline não vem pronto no repositório de propósito: ele descreve **a sua**
máquina, o seu modelo e o seu corpus. Gere o seu na primeira execução:

```bash
python scripts/ingest.py ./datasets/sample_documents
python scripts/evaluate.py --update-baseline
git add reports/baseline.json
```

Quando uma mudança melhora a qualidade, atualize o baseline no mesmo PR --
com o número novo visível no diff. É essa linha no diff que transforma
"acho que melhorou" em "melhorou de 0.84 para 0.91".
