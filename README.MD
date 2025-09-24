# Chat com Documentos (LangChain + Ollama)

Este projeto permite criar um chatbot capaz de responder perguntas sobre arquivos PDF ou DOCX carregados pelo usuário. O projeto utiliza [LangChain](https://www.langchain.com/) para manipulação de documentos e chunking, [FAISS](https://github.com/facebookresearch/faiss) para indexação de vetores, e [Ollama](https://ollama.com/) para gerar respostas com modelos locais. A interface é feita com [Streamlit](https://streamlit.io/).

## Funcionalidades

- Upload de múltiplos arquivos PDF ou DOCX.
- Seleção de modelo Ollama (`gemma3:270m`, `gemma3:1b`, `qwen3:4b`).
- Chunking automático do conteúdo dos documentos.
- Busca simples por relevância via overlap de palavras.
- Resposta do modelo exibida no Streamlit.
- Cadeia de pensamento (opcional) exibida em um expander para modelos que suportam.

## Pré-requisitos

- Python 3.10+
- [Ollama](https://ollama.com/) instalado e configurado localmente.
- Modelos desejados instalados no Ollama (`gemma3:270m`, `gemma3:1b`, `qwen3:4b`).

## Instalação

1. Clonar o repositório:

```bash
git clone <URL_DO_REPO>
cd chat-docs
````

2. Instalar dependências:

```bash
pip install -r requirements.txt
```

3. Certificar-se de que os modelos Ollama estão instalados:

```bash
ollama pull gemma3:1b
ollama pull gemma3:270m
ollama pull qwen3:4b
```

## Como usar

1. Rodar o app Streamlit:

```bash
streamlit run app.py
```

2. Abrir o navegador no endereço informado pelo Streamlit.
3. Selecionar o modelo, fazer upload dos arquivos e digitar a pergunta.
4. A resposta aparecerá no campo de resposta; se houver cadeia de pensamento, clique no expander para visualizar.

## Observações

* O modelo **qwen3** possui janela de contexto muito grande e pode fornecer respostas mais completas.
* A cadeia de pensamento só é exibida se o modelo retornar esse tipo de resposta.
* As bibliotecas utilizadas possuem licenças permissivas ([LangChain MIT](https://github.com/hwchase17/langchain/blob/master/LICENSE), [FAISS MIT](https://github.com/facebookresearch/faiss/blob/main/LICENSE), [Streamlit Apache 2.0](https://github.com/streamlit/streamlit/blob/develop/LICENSE)).

## Referências

* [Ollama](https://ollama.com/) – plataforma para rodar modelos de linguagem localmente.
* [LangChain](https://www.langchain.com/) – framework para construção de aplicações com LLMs.
* [FAISS](https://github.com/facebookresearch/faiss) – biblioteca de indexação e busca por vetores.
* [Streamlit](https://streamlit.io/) – framework para criar interfaces web em Python de forma rápida.
* [PyPDF](https://pypi.org/project/pypdf/) – biblioteca para manipulação de PDFs em Python.
* [python-docx](https://pypi.org/project/python-docx/) – biblioteca para manipulação de arquivos DOCX em Python.

```

---


```
