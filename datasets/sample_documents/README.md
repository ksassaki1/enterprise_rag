# Documentos de exemplo

Todo o conteúdo deste diretório é **sintético e fictício**, criado apenas para
exercitar a plataforma: nenhum dado corresponde a empreendimento, organização,
política ou pessoa real.

A estrutura de pastas é intencional -- ela alimenta a inferência de metadados:

```
brazil/          -> country=Brazil   (via _metadata.json)
chile/           -> country=Chile
*.md             -> frontmatter define project, document_type, year, asset_code
*.csv            -> planilha, um bloco a cada 25 linhas com cabecalho repetido
```

Para indexar:

```bash
python scripts/ingest.py ./datasets/sample_documents
```
