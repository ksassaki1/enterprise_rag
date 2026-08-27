# Segurança e privacidade

## Postura

Local-first por padrão. Documentos, embeddings, índice e geração ficam na
máquina que roda a plataforma: nenhum conteúdo sai para API de terceiros,
porque não existe chamada para fora. É a diferença entre poder e não poder
processar documento classificado como restrito.

## Superfície atual

| Área | Estado |
| --- | --- |
| Autenticação da API | **não implementada** -- veja abaixo |
| Autorização por documento | **não implementada** (ver roadmap: RBAC/coleções) |
| Isolamento multi-tenant | **não implementado** |
| Trilha de auditoria | consultas registradas em `query_log` (SQLite) |
| Dados em trânsito | HTTP local; TLS é responsabilidade do proxy à frente |
| Dados em repouso | sem criptografia; delegada ao disco/volume |
| Upload | limite de tamanho, nome de arquivo saneado, gravação isolada |

Este repositório é uma plataforma de referência, não um produto endurecido. As
lacunas estão listadas explicitamente porque escondê-las seria o problema de
segurança mais grave do projeto.

## Antes de expor além de `localhost`

1. **Autenticação.** Um middleware de API key ou OIDC no FastAPI. Nunca exponha
   `/documents` (upload e delete) sem autenticação.
2. **CORS.** `RAG_CORS_ORIGINS` vem como `*` para desenvolvimento. Restrinja.
3. **Rate limiting.** `/ask` é o endpoint caro: uma rajada de requisições
   satura o LLM e derruba o serviço para todos.
4. **TLS.** Termine em um proxy (Caddy, nginx, Traefik).
5. **Qdrant.** Suba com API key e não publique a porta 6333 na rede.

## Riscos específicos de RAG

**Injeção de prompt via documento.** Um documento ingerido pode conter texto
como "ignore as instruções anteriores". Mitigações presentes: as instruções
ficam na mensagem *system*, o contexto entra como blocos numerados e
delimitados, e o prompt exige citação -- o que torna a resposta auditável.
Mitigação ausente: sanitização ativa do conteúdo recuperado. Trate documento
ingerido como entrada não confiável.

**Vazamento por recuperação.** Sem RBAC, qualquer pessoa com acesso à API
alcança qualquer chunk indexado. Se o acervo tiver níveis de confidencialidade
diferentes, use coleções separadas por nível até o RBAC existir -- filtro de
metadados **não** é controle de acesso: ele é aplicado por conveniência de
relevância e pode ser sobrescrito pelo cliente.

**Alucinação com aparência de citação.** O modelo pode citar `[FONTE 2]` para
uma afirmação que a fonte 2 não sustenta. Por isso a resposta devolve o trecho
de cada citação (`excerpt`) e o `chunk_id`: a verificação é possível. A
métrica `valid_citation_rate` acompanha isso no agregado.

**Retenção.** `query_log` guarda as perguntas, que podem conter dados
sensíveis. Defina uma política de expurgo antes de rodar com usuários reais.

## Propriedade intelectual dos documentos de exemplo

Todo o conteúdo de `datasets/sample_documents/` é sintético e fictício. Nenhum
dado corresponde a empreendimento, organização, política, procedimento ou
pessoa real, e nada foi derivado de material proprietário de terceiros.
