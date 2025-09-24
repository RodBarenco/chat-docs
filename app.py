# app.py
import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

# ---------------- Configurações ----------------
MODEL_CONTEXT = {
    "gemma3:270m": 32000,
    "gemma3:1b": 32000,
    "qwen3:4b": 256000,
}

CHUNK_SIZE = 1000        # caracteres por chunk
CHUNK_OVERLAP = 100      # sobreposição entre chunks
TOP_K_CHUNKS = 6         # quantos chunks pegar para a pergunta

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Chat com Documentos (LangChain + Ollama)", layout="wide")
st.title("📚 Chat com Documentos usando LangChain + Ollama")

# Seleção de modelo
model_choice = st.selectbox("Escolha o modelo:", list(MODEL_CONTEXT.keys()), index=1)
st.write(f"Janela de contexto do modelo: {MODEL_CONTEXT[model_choice]} tokens aprox.")

# Upload de arquivos
uploaded_files = st.file_uploader(
    "Envie arquivos (PDF ou DOCX)", type=["pdf", "docx"], accept_multiple_files=True
)

# Pergunta
question = st.text_area("Pergunta sobre os documentos:", height=120)

# Container para mostrar a resposta
response_container = st.empty()

# ---------------- Lógica ----------------
if st.button("Perguntar"):
    if not uploaded_files:
        st.warning("Envie pelo menos um arquivo.")
    elif not question.strip():
        st.warning("Digite uma pergunta.")
    else:
        # ----------- Ler arquivos ----------
        docs = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            if file.type == "application/pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = Docx2txtLoader(tmp_path)

            docs.extend(loader.load())
            os.remove(tmp_path)

        if not docs:
            st.error("Nenhum texto extraído dos arquivos.")
        else:
            # ----------- Chunking ----------
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_documents(docs)

            # ----------- Ranking simples ----------
            def rank_chunks(chunks, question, top_k=TOP_K_CHUNKS):
                q_words = set(question.lower().split())
                scored = []
                for c in chunks:
                    c_words = set(c.page_content.lower().split())
                    score = len(q_words & c_words)
                    scored.append((score, c))
                scored.sort(reverse=True, key=lambda x: x[0])
                return [c.page_content for score, c in scored[:top_k]]

            top_chunks = rank_chunks(chunks, question, top_k=TOP_K_CHUNKS)

            # ----------- Montar prompt ----------
            prompt_text = "\n\n".join(top_chunks)
            prompt = f"Você é um assistente. Responda com base nos documentos abaixo:\n\n{prompt_text}\n\nPergunta: {question}"

            # ----------- Chamar modelo Ollama ----------
            llm = Ollama(model=model_choice)
            with st.spinner(f"Consultando {model_choice}..."):
                try:
                    full_response = llm.invoke(prompt)
                    
                    # Separar resposta final de cadeia de pensamento (se existir)
                    if "<think>" in full_response and "</think>" in full_response:
                        start = full_response.find("<think>")
                        end = full_response.find("</think>") + len("</think>")
                        chain_of_thought = full_response[start:end]
                        resposta_final = (full_response[:start] + full_response[end:]).strip()
                    else:
                        chain_of_thought = None
                        resposta_final = full_response

                    # ----------- Mostrar no Streamlit ----------
                    response_container.markdown("### 🤖 Resposta do Modelo")
                    response_container.write(resposta_final)

                    # Mostrar cadeia de pensamento opcional
                    if chain_of_thought:
                        with st.expander("Mostrar cadeia de pensamento (opcional)"):
                            st.text(chain_of_thought)

                except Exception as e:
                    response_container.error(f"Erro ao consultar o modelo: {e}")
