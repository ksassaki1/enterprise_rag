# ADR 0004 — Avaliação como portão de CI

**Status:** aceito

## Contexto

Em RAG, mudar `chunk_size` de 500 para 800 ou trocar o modelo de embedding
pode melhorar duas consultas e piorar dez. Sem medição, a discussão vira
opinião; com medição manual, ninguém repete o teste no PR seguinte.

## Decisão

Um golden dataset versionado, um baseline versionado
(`reports/baseline.json`) e um job de CI que roda a avaliação em todo PR que
toque `src/`, `datasets/` ou `scripts/`, falhando quando alguma métrica sai da
tolerância.

## Justificativa

Testes unitários provam que a função faz o que foi escrito. Só a avaliação
prova que o *sistema* continua respondendo bem. São garantias diferentes, e a
segunda é a que importa para o usuário.

## Consequências

* Toda mudança de qualidade fica visível como número no PR, não como
  afirmação no corpo do commit.
* O baseline precisa ser atualizado conscientemente quando a mudança melhora
  -- e essa linha no diff é a evidência.
* Tolerâncias precisam ser calibradas: apertadas demais geram falso positivo,
  frouxas demais deixam passar regressão.
* Exige que a avaliação rode sem GPU e sem serviço externo (ADR 0005).
