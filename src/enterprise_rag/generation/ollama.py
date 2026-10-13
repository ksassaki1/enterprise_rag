"""Camada de geração.

`OllamaLLM` fala com o servidor local por HTTP (`/api/chat`), com timeout e
retry explícitos: um LLM local sob carga é a peça mais lenta e mais instável
do sistema, e tratar isso como detalhe é o que faz a API inteira travar.

`EchoLLM` é uma implementação extrativa determinística: escolhe do contexto as
frases com maior sobreposição com a pergunta e as devolve já citadas. Não é um
modelo de linguagem e não pretende ser -- ela existe para que os testes e a
avaliação de recuperação rodem em CI sem GPU, isolando a qualidade do
retrieval da qualidade do gerador.
"""

from __future__ import annotations

import re
import time
from typing import Any, Protocol, runtime_checkable

import httpx

from enterprise_rag.config.logging import get_logger
from enterprise_rag.config.settings import Settings
from enterprise_rag.embeddings.sparse import tokenize
from enterprise_rag.generation.prompts import NOT_FOUND_MARKER, SYSTEM_PROMPT

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """Falha de geracao apos as tentativas configuradas."""


@runtime_checkable
class LLM(Protocol):
    name: str

    def chat(self, system: str, user: str) -> str: ...

    def complete(self, prompt: str, max_tokens: int = 512) -> str: ...

    def health(self) -> bool: ...


class OllamaLLM:
    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b-instruct",
        temperature: float = 0.1,
        num_ctx: int = 8192,
        timeout: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.url = url.rstrip("/")
        self.name = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(f"{self.url}{path}", json=payload)
                    response.raise_for_status()
                    return dict(response.json())
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                # backoff exponencial curto: o erro comum aqui e o modelo ainda
                # carregando na memoria, que resolve em segundos
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
        raise LLMError(f"Ollama indisponivel em {self.url}: {last_error}")

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        data = self._post("/api/chat", payload)
        return str(data.get("message", {}).get("content", "")).strip()

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        payload = {
            "model": self.name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }
        data = self._post("/api/generate", payload)
        return str(data.get("response", "")).strip()

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.url}/api/tags")
                response.raise_for_status()
                models = {item.get("name", "") for item in response.json().get("models", [])}
                # o Ollama registra "modelo:tag"; aceitamos o nome sem a tag
                return any(name.split(":")[0] == self.name.split(":")[0] for name in models)
        except (httpx.HTTPError, ValueError):
            return False


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class EchoLLM:
    """Gerador extrativo determinístico (sem modelo)."""

    name = "echo"

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def chat(self, system: str, user: str) -> str:
        context, question = _split_prompt(user)
        if not context.strip():
            return f"{NOT_FOUND_MARKER}: nenhum contexto recuperado."

        query_tokens = set(tokenize(question))
        scored: list[tuple[float, int, str]] = []
        for source_index, block in _iter_sources(context):
            for sentence in SENTENCE_RE.split(block):
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                tokens = set(tokenize(sentence))
                if not tokens or not query_tokens:
                    continue
                overlap = len(query_tokens & tokens) / len(query_tokens)
                if overlap > 0:
                    scored.append((overlap, source_index, sentence))

        if not scored:
            return f"{NOT_FOUND_MARKER}: o contexto recuperado nao responde a pergunta."

        scored.sort(key=lambda item: item[0], reverse=True)
        parts = [
            f"{sentence} [FONTE {source_index}]"
            for _, source_index, sentence in scored[: self.max_sentences]
        ]
        return " ".join(parts)

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        return "{}"

    def health(self) -> bool:
        return True


def _split_prompt(user: str) -> tuple[str, str]:
    # A linha separadora pode aparecer também em tabelas Markdown do contexto.
    # Só o último delimitador separa as fontes da pergunta; a instrução final
    # não deve entrar nos termos usados para escolher uma frase da resposta.
    context, separator, tail = user.rpartition("\n\n---\n\n")
    if not separator:
        context, tail = "", user
    question = tail.partition("Pergunta:")[2].split("\n\n", 1)[0].strip()
    return context.removeprefix("Contexto:").strip(), question


def _iter_sources(context: str) -> list[tuple[int, str]]:
    """Quebra o bloco de contexto de volta em (indice_da_fonte, texto)."""
    parts = re.split(r"\[FONTE\s+(\d+)\]", context)
    sources: list[tuple[int, str]] = []
    for index in range(1, len(parts) - 1, 2):
        sources.append((int(parts[index]), parts[index + 1]))
    return sources or [(1, context)]


def build_llm(settings: Settings) -> LLM:
    if settings.llm_backend == "ollama":
        return OllamaLLM(
            url=settings.ollama_url,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            num_ctx=settings.ollama_num_ctx,
            timeout=settings.ollama_timeout,
            max_retries=settings.ollama_max_retries,
        )
    return EchoLLM()


__all__ = ["LLM", "LLMError", "OllamaLLM", "EchoLLM", "build_llm", "SYSTEM_PROMPT"]
