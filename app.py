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
    "granite3.3:2b": 128000,
}

CHUNK_SIZE = 1000        # caracteres por chunk
CHUNK_OVERLAP = 100      # sobreposição entre chunks
TOP_K_CHUNKS = 6         # quantos chunks pegar para a pergunta

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Chat com Documentos (LangChain + Ollama)", layout="wide")
st.title("📚 Chat com Documentos usando LangChain + Ollama")

tabs = st.tabs(["Chat com arquivos", "Chat livre / híbrido"])

# ---------------- Funções auxiliares ----------------
def carregar_documentos(uploaded_files):
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
    return docs

def chunk_and_rank(docs, question, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, top_k=TOP_K_CHUNKS):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    q_words = set(question.lower().split())
    scored = []
    for c in chunks:
        c_words = set(c.page_content.lower().split())
        score = len(q_words & c_words)
        scored.append((score, c))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c.page_content for score, c in scored[:top_k]]

def gerar_resposta(model_choice, prompt):
    llm = Ollama(model=model_choice)
    full_response = llm.invoke(prompt)
    
    if "<think>" in full_response and "</think>" in full_response:
        start = full_response.find("<think>")
        end = full_response.find("</think>") + len("</think>")
        chain_of_thought = full_response[start:end]
        resposta_final = (full_response[:start] + full_response[end:]).strip()
    else:
        chain_of_thought = None
        resposta_final = full_response

    return resposta_final, chain_of_thought

# ---------------- Aba 1: Chat com arquivos ----------------
with tabs[0]:
    st.subheader("Chat com documentos carregados")
    model_choice_tab1 = st.selectbox("Escolha o modelo:", list(MODEL_CONTEXT.keys()), index=1, key="model_tab1")
    st.write(f"Janela de contexto do modelo: {MODEL_CONTEXT[model_choice_tab1]} tokens aprox.")

    uploaded_files_tab1 = st.file_uploader(
        "Envie arquivos (PDF ou DOCX)", type=["pdf", "docx"], accept_multiple_files=True, key="uploaded_files_tab1"
    )

    question_tab1 = st.text_area("Pergunta sobre os documentos:", height=120, key="question_tab1")
    response_container_tab1 = st.empty()

    if st.button("Perguntar", key="btn_tab1"):
        if not uploaded_files_tab1:
            st.warning("Envie pelo menos um arquivo.")
        elif not question_tab1.strip():
            st.warning("Digite uma pergunta.")
        else:
            docs_tab1 = carregar_documentos(uploaded_files_tab1)
            if not docs_tab1:
                st.error("Nenhum texto extraído dos arquivos.")
            else:
                top_chunks_tab1 = chunk_and_rank(docs_tab1, question_tab1)
                prompt_tab1 = f"Você é um assistente. Responda com base nos documentos abaixo:\n\n{'\n\n'.join(top_chunks_tab1)}\n\nPergunta: {question_tab1}"
                with st.spinner(f"Consultando {model_choice_tab1}..."):
                    try:
                        resposta_final, chain_of_thought = gerar_resposta(model_choice_tab1, prompt_tab1)
                        response_container_tab1.markdown("### 🤖 Resposta do Modelo")
                        response_container_tab1.write(resposta_final)
                        if chain_of_thought:
                            with st.expander("Mostrar cadeia de pensamento (opcional)"):
                                st.text(chain_of_thought)
                    except Exception as e:
                        response_container_tab1.error(f"Erro ao consultar o modelo: {e}")

# ---------------- Aba 2: Chat livre / híbrido ----------------
with tabs[1]:
    st.subheader("Chat livre / híbrido")

    # Inicializar estado da sessão
    if "chat_history_tab2" not in st.session_state:
        st.session_state.chat_history_tab2 = []

    if "docs_tab2" not in st.session_state:
        st.session_state.docs_tab2 = []
        
    # Seleção de modelo e informações
    model_choice_tab2 = st.selectbox("Escolha o modelo:", list(MODEL_CONTEXT.keys()), index=1, key="model_tab2")
    st.write(f"Janela de contexto do modelo: {MODEL_CONTEXT[model_choice_tab2]} tokens aprox.")

    # Container para o chat
    chat_container = st.container(height=400, border=True) # Diminui um pouco a altura

    # Renderiza o histórico dentro do container
    with chat_container:
        for chat in st.session_state.chat_history_tab2:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])
                if "think" in chat and chat["think"]:
                    with st.expander("Mostrar cadeia de pensamento (opcional)"):
                        st.text(chat["think"])
    
    # Colunas para a caixa de upload e a entrada de texto
    col1, col2 = st.columns([10, 1]) # Inverte a ordem das colunas

    with col1:
        user_input = st.chat_input("Digite sua mensagem...", key="user_input_tab2")

    with col2:
        with st.popover("📎"):
            uploaded_files_tab2 = st.file_uploader(
                "Adicione arquivos para contexto opcional",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                key="uploaded_files_tab2_widget",
                help="Os arquivos adicionados aqui serão usados para dar contexto às suas perguntas. Eles são processados apenas uma vez."
            )
            if uploaded_files_tab2:
                with st.spinner("Processando documentos..."):
                    st.session_state.docs_tab2 = carregar_documentos(uploaded_files_tab2)

    if user_input:
        # Adiciona a mensagem do usuário ao histórico ANTES de chamar o modelo
        st.session_state.chat_history_tab2.append({"role": "user", "content": user_input})
        
        # Monta prompt com contexto dos arquivos
        prompt_parts = []
        if st.session_state.docs_tab2:
            top_chunks_tab2 = chunk_and_rank(st.session_state.docs_tab2, user_input)
            if top_chunks_tab2:
                prompt_parts.append(f"Baseado nos seguintes documentos:\n\n{'\n\n'.join(top_chunks_tab2)}")
        
        # Adiciona histórico da conversa ao prompt para manter contexto
        history_prompt = "Histórico do chat:\n"
        for chat in st.session_state.chat_history_tab2[-8:]: 
            history_prompt += f"{chat['role']}: {chat['content']}\n"
        prompt_parts.append(history_prompt)
        prompt_parts.append(f"{user_input}")
        
        full_prompt = "\n\n".join(prompt_parts)

        # Chama o modelo e exibe a resposta
        with st.spinner("Consultando..."):
            try:
                resposta_final, chain_of_thought = gerar_resposta(model_choice_tab2, full_prompt)
                st.session_state.chat_history_tab2.append({"role": "assistant", "content": resposta_final, "think": chain_of_thought})
                st.rerun() # Força o Streamlit a rerenderizar
            except Exception as e:
                st.error(f"Erro ao consultar o modelo: {e}")