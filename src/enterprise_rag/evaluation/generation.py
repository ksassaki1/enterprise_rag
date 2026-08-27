"""Métricas de geração.

Sem juiz-LLM aqui, e por escolha: um juiz baseado no mesmo modelo local
introduz variância, custa minutos por execução e transformaria o portão de
regressão do CI em algo não determinístico. Para um golden dataset com
respostas curtas e factuais (números, unidades, nomes), verificações
determinísticas medem o que importa e rodam em milissegundos:

* **containment** -- os tokens da resposta esperada aparecem na resposta?
  É o sinal certo para "850 MW" ou "76 metros".
* **token F1** -- sobreposição parcial, para respostas mais longas.
* **not-found rate** -- com que frequência o sistema admite não saber. Alto
  demais indica recall ruim; próximo de zero indica que o modelo está
  inventando em vez de recusar.
* **citações válidas** -- a resposta cita fontes que existem no contexto e que
  pertencem aos documentos esperados. Resposta certa por acaso, com citação
  errada, é uma resposta errada em ambiente corporativo.

Um juiz-LLM continua sendo o passo natural depois disso, rodando fora do
portão de CI, sobre perguntas dissertativas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from enterprise_rag.embeddings.sparse import tokenize


def token_f1(prediction: str, reference: str) -> float:
    predicted = tokenize(prediction, remove_stopwords=False)
    expected = tokenize(reference, remove_stopwords=False)
    if not predicted or not expected:
        return 0.0
    common = set(predicted) & set(expected)
    if not common:
        return 0.0
    precision = len(common) / len(set(predicted))
    recall = len(common) / len(set(expected))
    return 2 * precision * recall / (precision + recall)


def contains_answer(prediction: str, reference: str, threshold: float = 0.8) -> bool:
    """Verdadeiro quando a maior parte dos tokens da resposta esperada aparece.

    Tolera diferença de redação ("A altura maxima e de 76 metros" contém
    "76 metros") sem aceitar qualquer texto que apenas mencione o assunto.
    """
    expected = tokenize(reference, remove_stopwords=False)
    if not expected:
        return False
    predicted = set(tokenize(prediction, remove_stopwords=False))
    hits = sum(1 for token in expected if token in predicted)
    return hits / len(expected) >= threshold


@dataclass
class GenerationMetrics:
    questions: int = 0
    correct: int = 0
    incorrect: int = 0
    not_found: int = 0
    avg_token_f1: float = 0.0
    citation_rate: float = 0.0
    valid_citation_rate: float = 0.0
    failures: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.questions if self.questions else 0.0

    @property
    def not_found_rate(self) -> float:
        return self.not_found / self.questions if self.questions else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "questions": self.questions,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "not_found": self.not_found,
            "accuracy": round(self.accuracy, 4),
            "not_found_rate": round(self.not_found_rate, 4),
            "avg_token_f1": round(self.avg_token_f1, 4),
            "citation_rate": round(self.citation_rate, 4),
            "valid_citation_rate": round(self.valid_citation_rate, 4),
            "failures": self.failures,
        }


@dataclass
class GenerationCase:
    question_id: str
    answer: str
    expected_answer: str
    answer_found: bool
    cited_documents: list[str]
    expected_documents: list[str]


def aggregate_generation(cases: list[GenerationCase]) -> GenerationMetrics:
    from enterprise_rag.evaluation.retrieval import matches

    metrics = GenerationMetrics(questions=len(cases))
    if not cases:
        return metrics

    f1_total = 0.0
    cited_total = 0
    valid_citation_total = 0

    for case in cases:
        f1_total += token_f1(case.answer, case.expected_answer)

        if not case.answer_found:
            metrics.not_found += 1
            metrics.failures.append(f"{case.question_id}: nao encontrado")
        elif not case.expected_answer.strip():
            # pergunta sem gabarito textual (ex.: nao-respondivel): recusar ja e acertar
            metrics.correct += 1
        elif contains_answer(case.answer, case.expected_answer):
            metrics.correct += 1
        else:
            metrics.incorrect += 1
            metrics.failures.append(f"{case.question_id}: resposta divergente")

        if case.cited_documents:
            cited_total += 1
            expected = set(case.expected_documents)
            if expected and any(matches(doc, expected) for doc in case.cited_documents):
                valid_citation_total += 1

    total = len(cases)
    metrics.avg_token_f1 = f1_total / total
    metrics.citation_rate = cited_total / total
    metrics.valid_citation_rate = valid_citation_total / total
    return metrics
