# 🎓 College Helpdesk AI Chatbot

An intelligent, RAG-powered (Retrieval-Augmented Generation) Streamlit application that provides instant, accurate answers to student questions regarding admissions, fees, examinations, campus rules, scholarships, and academic circulars based on official college documents.

**Domain:** GenAI

**Team Lead:** Guthula Likhith & Badrinadh

---

## 🚀 Key Architecture & Features

- **Document QA (RAG)**: Retrieves precise college policy excerpts from official handbook documents using FAISS vector indexing and sentence embeddings.
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` for generating embeddings.
- **Chat LLM Options**: Supports top models hosted on Hugging Face Inference Providers (e.g. `Qwen/Qwen2.5-72B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`).
- **Resilient Fallback Mode**: If external LLM API credits are exhausted, the system automatically provides direct relevant handbook passages with citation references.
- **Streamlit Web Interface**: Fast, responsive chat UI with quick question suggestions, document references, and chat history management.
- **Deployment Ready**: Fully configured for Streamlit Community Cloud and local environments.

---

## 🛠️ Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Likhithguthula/College_Help_Desk.git
   cd College_Help_Desk
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the project root:
   ```env
   HF_TOKEN=your_huggingface_token_here
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   ```

4. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deploy to Streamlit Community Cloud (Free)

1. Push your changes to your GitHub repository:
   ```bash
   git add .
   git commit -m "Fix HF token, embedding model, and chat LLM inference"
   git push origin main
   ```

2. Open [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**.
4. Configure the deployment settings:
   - **Repository**: `Likhithguthula/College_Help_Desk`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Advanced settings"** -> **"Secrets"**, and paste:
   ```toml
   HF_TOKEN = "your_huggingface_token_here"
   ```
6. Click **"Deploy"**!

---

## 📚 Knowledge Base Documents

The chatbot indexes and queries the following official PDFs located in the `knowledge/` directory:
- `01_college_handbook.pdf`
- `02_admission_and_registration.pdf`
- `03_fee_structure_and_payment.pdf`
- `04_examination_and_evaluation_rules.pdf`
- `05_student_rules_and_services.pdf`
- `06_academic_circulars_2026.pdf`
