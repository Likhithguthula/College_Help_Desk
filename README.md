# 🎓 College Helpdesk AI Chatbot

An intelligent, RAG-powered (Retrieval-Augmented Generation) Streamlit application that provides instant, accurate answers to student questions regarding admissions, fees, examinations, campus rules, scholarships, and academic circulars based on official college documents.

---

## 🚀 Features

- **Document QA (RAG)**: Uses FAISS vector store and HuggingFace Sentence Transformers to retrieve precise college policy excerpts.
- **Inference with Qwen2.5**: Generates clear, helpful answers strictly grounded in college handbook documents.
- **Streamlit Web Interface**: Fast, responsive chat UI with question suggestions, document citations, and chat history.
- **Ready for Deployment**: Optimized for Streamlit Community Cloud with secret management and caching.

---

## 🛠️ Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Likhithguthula/College_Help_Desk.git
   cd College-Helpdesk-Chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   HF_TOKEN=your_huggingface_token_here
   ```

4. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Deploy to Streamlit Community Cloud (Free)

1. Push this repository to GitHub:
   ```bash
   git add .
   git commit -m "Configure Streamlit app and dependencies for deployment"
   git push origin main
   ```

2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New app"**.
4. Select:
   - **Repository**: `Likhithguthula/College_Help_Desk`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Advanced settings"** -> **"Secrets"**, and add:
   ```toml
   HF_TOKEN = "your_huggingface_token_here"
   ```
6. Click **"Deploy"**!
