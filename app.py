import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Streamlit UI setup
st.set_page_config(page_title="AI Compliance Agent", layout="centered")
st.title("🛡️Welcome to SATYENDRA AI Compliance Agent")
st.write("Upload a compliance PDF document and ask questions about it!")

# Upload a PDF file
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded PDF locally
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split the PDF
    loader = PyPDFLoader("uploaded.pdf")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    # Embed and create FAISS vector DB
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embedding_model)
    retriever = vectorstore.as_retriever()

    # User query
    query = st.text_input("💬 Arre bhaya question puchna re yaha jo document chipkaya hai tune - Ask a question about the PDF:")

    if query:
        results = retriever.invoke(query)  # ✅ works with new LangChain version
        if results:
            st.subheader("📌 Answer")
            st.write(results[0].page_content)
        else:
            st.warning("No relevant content found.")
else:
    st.info("Please upload a PDF file to get started.")
