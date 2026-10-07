import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing in .env file")


# ============================================================
# LOAD COLLEGE DOCUMENTS
# ============================================================

documents = []

files = [
    'knowledge/01_college_handbook.pdf',
    'knowledge/02_admission_and_registration.pdf',
    'knowledge/03_fee_structure_and_payment.pdf',
    'knowledge/04_examination_and_evaluation_rules.pdf',
    'knowledge/05_student_rules_and_services.pdf',
    'knowledge/06_academic_circulars_2026.pdf'
]

for file in files:

    loader = PyPDFLoader(file)

    documents.extend(
        loader.load()
    )


print("Total pages loaded:", len(documents))


# ============================================================
# TEXT CHUNKING
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Total chunks created:", len(chunks))


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CREATE VECTOR DATABASE AUTOMATICALLY
# ============================================================

vector_db_path = "vector_db"

if not os.path.exists(vector_db_path):

    print("\nCreating vector database...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(
        vector_db_path
    )

    print("Vector database created successfully.")

else:

    print("\nVector database already exists.")

    vectorstore = FAISS.load_local(
        vector_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


# ============================================================
# CREATE RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

client = InferenceClient(
    provider="auto",
    token=HF_TOKEN
)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


# ============================================================
# AI FUNCTION
# ============================================================

def ask_ai(question, context):

    prompt = f"""
You are a College Helpdesk Assistant.

Answer the student's question ONLY using the information
provided in the college documents below.

Rules:

1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not invent information.
4. Give simple and easy-to-understand answers.
5. If the answer is not available in the documents, say:

"Sorry, I couldn't find this information in the
college documents."

6. Keep the answer directly related to the student's question.

College Documents Context:
{context}

Student Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500,
        temperature=0.2
    )

    return response.choices[0].message.content


# ============================================================
# CHAT LOOP
# ============================================================

print("\n")
print("=" * 60)
print("        COLLEGE HELPDESK CHATBOT")
print("=" * 60)

print("""
You can ask questions about:

- Admissions
- Registration
- Fees
- Examinations
- Attendance
- College Rules
- Library
- Hostel
- Scholarships
- Academic Circulars

Type 'exit' or 'stop' to close the chatbot.
""")

print("=" * 60)


while True:

    question = input("\nStudent: ")

    # Exit
    if question.lower().strip() in ["exit", "stop"]:

        print("\nThank you for using the College Helpdesk Chatbot.")
        break


    # Empty question
    if not question.strip():

        print("Please enter a question.")

        continue


    # ========================================================
    # RETRIEVE RELEVANT DOCUMENTS
    # ========================================================

    docs = retriever.invoke(question)


    # ========================================================
    # CREATE CONTEXT
    # ========================================================

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    try:

        answer = ask_ai(
            question,
            context
        )

        print("\nAssistant:")
        print("-" * 60)
        print(answer)
        print("-" * 60)


        # ====================================================
        # DISPLAY SOURCES
        # ====================================================

        print("\nSources:")

        displayed_sources = set()

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown Document"
            )

            page = doc.metadata.get("page")

            if page is not None:

                source_name = (
                    f"{os.path.basename(source)} "
                    f"(Page {page + 1})"
                )

            else:

                source_name = os.path.basename(source)


            if source_name not in displayed_sources:

                print(f"- {source_name}")

                displayed_sources.add(source_name)


    except Exception as e:

        print("\nError:", e)
