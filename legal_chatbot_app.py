import os
import streamlit as st
import warnings
import logging

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

# =======================
# Secrets (Streamlit Cloud)
# =======================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY غير موجود في Secrets")
    st.stop()

# =======================
# Clean logs
# =======================
warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


# =======================
# Setup
# =======================

PDF_DIR = "laws_pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

# =======================
# Streamlit Config
# =======================
st.set_page_config(page_title="⚖️ Legal AI Assistant", layout="wide")

st.title("⚖️ Legal AI Assistant")
st.caption("مساعد قانوني ذكي للمحامين | LangChain 1.2.7 + Groq")

# =======================
# Sidebar – Upload PDFs
# =======================
st.sidebar.header("📤 رفع ملفات قانونية")

uploaded_files = st.sidebar.file_uploader(
    "ارفع ملفات PDF",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(PDF_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("✅ تم رفع الملفات بنجاح")

# =======================
# Load LLM
# =======================
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=GROQ_API_KEY,
    temperature=0
)

# =======================
# Load & Index PDFs
# =======================
@st.cache_resource
def load_vectorstore():
    loader = PyPDFDirectoryLoader(PDF_DIR)
    documents = loader.load()

    if len(documents) == 0:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    docs = splitter.split_documents(documents)

    if len(docs) == 0:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(docs, embeddings)

# Clear cache if new files uploaded
if uploaded_files:
    st.cache_resource.clear()

vectorstore = load_vectorstore()

if vectorstore is None:
    st.info("📂 ارفع ملفات PDF قانونية من الشريط الجانبي للبدء")
    st.stop()

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# =======================
# Prompt
# =======================
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
أنت مساعد قانوني محترف يساعد المحامين فقط.

حلل السؤال باستخدام النص القانوني المتاح تحليلاً داخليًا.
لا تذكر أي تفكير أو خطوات تحليل.
أخرج فقط النتيجة النهائية بشكل منظم.

المطلوب في الإجابة:
- رقم المادة القانونية
- رقم الصفحة
- ملخص قانوني دقيق
- شرح مبسط وواضح

النص القانوني:
{context}

السؤال:
{question}

الإجابة:
"""
)

# =======================
# RAG Chain
# =======================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# =======================
# UI – Ask Question
# =======================
query = st.text_input(
    "📝 اسأل سؤال قانوني:",
    placeholder="مثال: ما هي عقوبة التزوير في المحررات الرسمية؟"
)

if st.button("🔍 اسأل"):
    if not query.strip():
        st.warning("من فضلك اكتب سؤال قانوني")
    else:
        with st.spinner("جاري تحليل القوانين..."):
            response = rag_chain.invoke(query)
            source_docs = retriever.invoke(query)

        st.subheader("📌 الإجابة القانونية")
        st.write(response.content)

        with st.expander("📄 الصفحات المستخدمة"):
            for doc in source_docs:
                st.markdown(
                    f"**صفحة:** {doc.metadata.get('page', 'غير معروف')}"
                )





