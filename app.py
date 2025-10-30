import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import easyocr
from pdf2image import convert_from_bytes
from PIL import Image
import io

# Streamlit UI
st.set_page_config(page_title="AI Compliance Agent", layout="centered")
st.title("🛡️ SATYENDRA AI Compliance Agent")
st.write("Upload a .txt or .pdf file (even scanned PDFs) and ask questions about it!")

uploaded_file = st.file_uploader("📄 Upload your document", type=["txt", "pdf"])

def extract_text_from_pdf(file_bytes):
    """Try to extract text directly from PDF (non-scanned)."""
    text = ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception:
        pass
    return text.strip()

def extract_text_with_ocr(file_bytes):
    """Extract text from scanned PDFs using EasyOCR."""
    st.info("🧠 Detected scanned PDF — running OCR (this may take a bit)...")
    text = ""
    try:
        images = convert_from_bytes(file_bytes)  # Convert PDF pages to images
        reader = easyocr.Reader(["en"], gpu=False)
        for img in images:
            result = reader.readtext(np.array(img), detail=0)
            text += " ".join(result) + "\n"
    except Exception as e:
        st.error(f"OCR extraction failed: {e}")
    return text.strip()

def process_text(text):
    """Split and embed text into FAISS for retrieval."""
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore.as_retriever()

if uploaded_file:
    file_bytes = uploaded_file.read()
    file_ext = uploaded_file.name.split(".")[-1].lower()

    extracted_text = ""

    if file_ext == "txt":
        extracted_text = file_bytes.decode("utf-8", errors="ignore").strip()
    elif file_ext == "pdf":
        extracted_text = extract_text_from_pdf(file_bytes)
        if not extracted_text:
            extracted_text = extract_text_with_ocr(file_bytes)

    if not extracted_text:
        st.error("❌ Could not extract any text from the file. Try another document.")
    else:
        retriever = process_text(extracted_text)
        query = st.text_input("💬 Ask a question about your document:")

        if query:
            results = retriever.invoke(query)
            if results:
                st.subheader("📌 Answer")
                st.write(results[0].page_content)
            else:
                st.warning("No relevant content found.")
else:
    st.info("Please upload a .txt or .pdf file to get started.")
