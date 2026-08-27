"""Prompts de geração.

Três exigências, e cada uma existe por um modo de falha concreto:

* **responder só pelo contexto** -- sem isso o modelo completa com o que
  "sabe", e a alucinação chega bem escrita e com números plausíveis;
* **admitir quando não sabe** -- um marcador explícito transforma "não
  encontrado" em métrica (`answer_not_found_total`) em vez de uma resposta
  errada silenciosa;
* **citar `[FONTE n]`** -- é o que permite ligar cada afirmação de volta ao
  `chunk_id`, à página e ao arquivo. Resposta sem citação, em ambiente
  corporativo, não é auditável e portanto não é utilizável.
"""

from __future__ import annotations

import re

NOT_FOUND_MARKER = "RESPOSTA_NAO_ENCONTRADA"

SYSTEM_PROMPT = f"""Voce e um assistente corporativo de documentacao tecnica.

Regras obrigatorias:
1. Responda EXCLUSIVAMENTE com base no contexto fornecido. Nunca use
   conhecimento externo, mesmo que voce tenha certeza da resposta.
2. Se o contexto nao contiver a informacao necessaria, responda exatamente
   com o marcador {NOT_FOUND_MARKER} seguido de uma frase curta explicando o
   que faltou.
3. Cite as fontes usadas no formato [FONTE n], logo apos a afirmacao que elas
   sustentam. Toda afirmacao factual precisa de pelo menos uma citacao.
4. Nao invente numeros, datas, nomes ou unidades. Reproduza os valores
   exatamente como aparecem no contexto.
5. Seja direto e objetivo. Responda no idioma da pergunta.
"""

USER_TEMPLATE = """Contexto:

{context}

---

Pergunta: {question}

Responda usando apenas o contexto acima e cite as fontes no formato [FONTE n]."""

NO_CONTEXT_TEMPLATE = """Nenhum documento relevante foi recuperado para a pergunta abaixo.

Pergunta: {question}

Responda exatamente com {marker} e explique em uma frase que a base de
conhecimento nao contem essa informacao."""

CITATION_RE = re.compile(r"\[FONTE\s+(\d+)\]", re.IGNORECASE)


def build_user_prompt(question: str, context: str) -> str:
    if not context.strip():
        return NO_CONTEXT_TEMPLATE.format(question=question, marker=NOT_FOUND_MARKER)
    return USER_TEMPLATE.format(context=context, question=question)


def extract_citations(answer: str) -> list[int]:
    """Índices de fonte citados, na ordem em que aparecem e sem repetição."""
    seen: dict[int, None] = {}
    for match in CITATION_RE.finditer(answer):
        seen.setdefault(int(match.group(1)), None)
    return list(seen)


def is_not_found(answer: str) -> bool:
    return NOT_FOUND_MARKER.lower() in answer.lower()


def strip_marker(answer: str) -> str:
    return re.sub(NOT_FOUND_MARKER, "", answer, flags=re.IGNORECASE).strip(" :-\n") or (
        "Nao encontrei essa informacao nos documentos indexados."
    )
