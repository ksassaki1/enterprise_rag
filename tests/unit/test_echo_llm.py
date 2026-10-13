"""Regressões do parser do prompt usado pelo gerador determinístico."""

from __future__ import annotations

from enterprise_rag.generation.ollama import EchoLLM, _split_prompt
from enterprise_rag.generation.prompts import build_user_prompt, is_not_found


def test_pergunta_nao_inclui_instrucoes_ao_gerador():
    question = "Qual e o valor do seguro contratado?"
    context = "[FONTE 1]\nO contexto acima descreve apenas o formato das fontes."
    prompt = build_user_prompt(question, context)
    assert _split_prompt(prompt) == (context, question)
    assert is_not_found(EchoLLM().chat("", prompt))


def test_tabela_markdown_nao_corta_as_fontes_do_prompt():
    context = "[FONTE 1]\n| tag | potencia |\n| --- | --- |\n| UG-01 | 212.5 MW |"
    question = "Qual a potencia da UG-01?"
    prompt = build_user_prompt(question, context)
    assert _split_prompt(prompt) == (context, question)
    answer = EchoLLM().chat("", prompt)
    assert "212.5 MW" in answer
    assert "[FONTE 1]" in answer


def test_prompt_sem_contexto_devolve_marcador_de_nao_encontrado():
    prompt = build_user_prompt("Qual a altura?", "")
    assert _split_prompt(prompt) == ("", "Qual a altura?")
    assert is_not_found(EchoLLM().chat("", prompt))
