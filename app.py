import streamlit as st
import io
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import easyocr
import pdfplumber
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="AI Compliance Agent", layout="centered")
st.title("🛡️ SATYENDRA AI Compliance Agent")
st.write("Upload a .txt or .pdf file (even scanned PDFs) and ask questions about it!")

uploaded_file = st.file_uploader("📄 Upload your document", type=["txt", "pdf"])

# ----------- TEXT EXTRACTION ------------
def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text
    except Exception as e:
        st.warning(f"Text extraction failed: {e}")
    return text.strip()

def extract_text_with_ocr(file_bytes):
    st.info("🧠 Detected scanned PDF — running OCR (may take a minute)...")
    text = ""
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        reader = easyocr.Reader(["en"], gpu=False)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            result = reader.readtext(np.array(img), detail=0)
            text += " ".join(result) + "\n"
    except Exception as e:
        st.error(f"OCR failed: {e}")
    return text.strip()

# ----------- TEXT PROCESSING ------------
def process_text(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    if not docs:
        raise ValueError("No text chunks to embed. (File may be empty or OCR failed.)")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore.as_retriever()

# ----------- APP LOGIC ------------
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
        st.error("❌ No readable text found in this file. Try uploading a different one.")
    else:
        try:
            retriever = process_text(extracted_text)
            query = st.text_input("💬 Ask a question about your document:")
            if query:
                results = retriever.invoke(query)
                if results:
                    st.subheader("📌 Answer")
                    st.write(results[0].page_content)
                else:
                    st.warning("No relevant information found.")
        except Exception as e:
            st.error(f"⚠️ Processing error: {e}")
else:
    st.info("Please upload a .txt or .pdf file to get started.")
