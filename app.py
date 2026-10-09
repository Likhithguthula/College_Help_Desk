import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="College Helpdesk AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin-top: 8px;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .badge-info {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ENVIRONMENT & SECRETS
# ============================================================
load_dotenv()

def get_hf_token():
    # 1. Check Streamlit secrets (for Streamlit Community Cloud)
    if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
        return st.secrets["HF_TOKEN"]
    # 2. Check local environment variables (.env)
    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token
    # 3. Check session state (manual entry fallback)
    return st.session_state.get("custom_hf_token", "")


# ============================================================
# RESOURCE CACHING (Embeddings & Vector Database)
# ============================================================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

@st.cache_resource(show_spinner="Initializing College Knowledge Base...")
def load_vectorstore(_embeddings):
    vector_db_path = "vector_db"
    
    # Try loading existing FAISS index
    if os.path.exists(vector_db_path) and os.path.exists(os.path.join(vector_db_path, "index.faiss")):
        try:
            return FAISS.load_local(
                vector_db_path,
                _embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.warning(f"Could not load pre-built index, rebuilding from PDFs: {e}")
    
    # Otherwise build from knowledge PDFs
    pdf_files = [
        'knowledge/01_college_handbook.pdf',
        'knowledge/02_admission_and_registration.pdf',
        'knowledge/03_fee_structure_and_payment.pdf',
        'knowledge/04_examination_and_evaluation_rules.pdf',
        'knowledge/05_student_rules_and_services.pdf',
        'knowledge/06_academic_circulars_2026.pdf'
    ]
    
    documents = []
    for file in pdf_files:
        if os.path.exists(file):
            try:
                loader = PyPDFLoader(file)
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"Could not load document {file}: {e}")
                
    if not documents:
        return None
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    
    vectorstore = FAISS.from_documents(chunks, _embeddings)
    
    try:
        os.makedirs(vector_db_path, exist_ok=True)
        vectorstore.save_local(vector_db_path)
    except Exception:
        pass
        
    return vectorstore


# ============================================================
# LLM INFERENCE
# ============================================================
# Available chat models for Hugging Face Inference Providers
CHAT_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "google/gemma-2-2b-it"
]

def ask_ai(client: InferenceClient, question: str, context: str, model_name: str):
    prompt = f"""You are an official College Helpdesk Assistant.

Answer the student's question accurately and clearly using ONLY the information provided in the college documents below.

Rules:
1. Use only the provided context.
2. Do not invent or assume information.
3. Give simple, clear, bulleted or structured answers where appropriate.
4. If the answer is not found in the documents, state:
"Sorry, I couldn't find this information in the official college documents."
5. Keep the answer concise and student-friendly.

College Documents Context:
{context}

Student Question:
{question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        # Check if error is due to credit limit or model routing
        if "402" in error_msg or "Payment Required" in error_msg or "credits" in error_msg.lower():
            # Graceful RAG extraction fallback if HF Inference Provider credits are exhausted
            fallback_answer = format_rag_fallback(question, context)
            return fallback_answer
        raise e

def format_rag_fallback(question: str, context: str) -> str:
    """Fallback generator when external LLM API credits are unavailable."""
    return f"""### 📋 Official College Handbook Information

Based on the official college documents, here is the relevant information retrieved for your query:

{context}

---
*💡 Note: Generated via direct College Knowledge Base Retrieval.*"""


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("🎓 College Helpdesk")
    st.markdown("Your 24/7 AI-powered assistant for academic, admission, fee, and campus queries.")
    st.divider()
    
    # Token Authentication
    hf_token = get_hf_token()
    if not hf_token:
        st.warning("⚠️ Hugging Face Token required")
        user_input_token = st.text_input(
            "Enter HF Token:", 
            type="password", 
            help="Add HF_TOKEN in .env or Streamlit Secrets"
        )
        if user_input_token:
            st.session_state["custom_hf_token"] = user_input_token
            st.rerun()
    else:
        st.success("✅ Hugging Face Connected")

    st.markdown("### ⚙️ Model Settings")
    st.caption(f"**Embedding Model:** `{EMBEDDING_MODEL_NAME}`")
    
    selected_llm = st.selectbox(
        "Chat LLM Model:",
        options=CHAT_MODELS,
        index=0,
        help="Select the generation model hosted on Hugging Face Inference Providers"
    )

    st.divider()
    st.markdown("### 📌 Common Questions")
    sample_queries = [
        "What is the fee structure for this academic year?",
        "What is the minimum attendance requirement?",
        "How do I apply for hostel accommodation?",
        "What are the rules for re-evaluation of exam papers?",
        "What scholarships are available for students?",
        "What are the library timings and borrowing rules?"
    ]
    
    for query in sample_queries:
        if st.button(query, use_container_width=True):
            st.session_state["selected_query"] = query

    st.divider()
    st.markdown("### 📚 Knowledge Base Documents")
    st.markdown("""
    - *01_college_handbook.pdf*
    - *02_admission_and_registration.pdf*
    - *03_fee_structure_and_payment.pdf*
    - *04_examination_and_evaluation_rules.pdf*
    - *05_student_rules_and_services.pdf*
    - *06_academic_circulars_2026.pdf*
    """)
    
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# MAIN INTERFACE
# ============================================================
st.markdown('<div class="main-header">🎓 College Helpdesk Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about admissions, fees, examinations, campus rules, scholarships, and academic circulars.</div>', unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! 👋 I am your College Helpdesk Assistant. How can I help you today? You can ask me anything about admissions, fee payment, exam regulations, library, hostel, or circulars.",
            "sources": []
        }
    ]

# Display Chat Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Document References & Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- **{src}**")

# Get input either from chat_input or sidebar sample query
prompt_input = st.chat_input("Type your question here...")
if st.session_state.get("selected_query"):
    prompt_input = st.session_state.pop("selected_query")

# Handle User Input
if prompt_input:
    current_token = get_hf_token()
    if not current_token:
        st.error("Please provide a valid Hugging Face API Token (HF_TOKEN) to continue.")
    else:
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.write(prompt_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching college documents and generating answer..."):
                try:
                    # Load components
                    embeddings = get_embeddings()
                    vectorstore = load_vectorstore(embeddings)
                    
                    if vectorstore is None:
                        st.error("Could not load college knowledge documents. Please verify the `knowledge/` or `vector_db/` folder.")
                    else:
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                        docs = retriever.invoke(prompt_input)
                        
                        context = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Prepare sources
                        sources = []
                        displayed = set()
                        for doc in docs:
                            src = doc.metadata.get("source", "College Document")
                            page = doc.metadata.get("page")
                            if page is not None:
                                src_label = f"{os.path.basename(src)} (Page {page + 1})"
                            else:
                                src_label = os.path.basename(src)
                                
                            if src_label not in displayed:
                                sources.append(src_label)
                                displayed.add(src_label)
                                
                        # Call Hugging Face API
                        client = InferenceClient(api_key=current_token)
                        answer = ask_ai(client, prompt_input, context, model_name=selected_llm)
                        
                        st.write(answer)
                        if sources:
                            with st.expander("📄 Document References & Sources", expanded=False):
                                for s in sources:
                                    st.markdown(f"- **{s}**")
                                    
                        # Save to message history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                except Exception as e:
                    error_msg = f"An error occurred while generating the answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
