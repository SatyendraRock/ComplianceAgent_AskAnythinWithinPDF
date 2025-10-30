import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pdf2image import convert_from_path
import pytesseract

# ------------- STREAMLIT UI SETUP -------------
st.set_page_config(page_title="AI Compliance Agent", layout="centered")
st.title("🛡️ Welcome to SATYENDRA AI Compliance Agent")
st.write("Upload a .txt or .pdf file (even scanned PDFs) and ask questions about it!")

# ------------- FILE UPLOAD -------------
uploaded_file = st.file_uploader("📄 Upload a .txt or .pdf file", type=["txt", "pdf"])

def extract_text_from_scanned_pdf(pdf_path):
    """Extract text from scanned PDF using OCR"""
    st.info("🔍 Scanned PDF detected. Running OCR... (this may take a few seconds)")
    text_content = ""
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)
        for img in images:
            text_content += pytesseract.image_to_string(img)
    except Exception as e:
        st.error(f"OCR extraction failed: {e}")
    return text_content.strip()

def load_document(file_path, file_type):
    """Load and return text content from txt or pdf"""
    if file_type == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        return docs

    elif file_type == "pdf":
        # Try normal PDF extraction
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        docs = [d for d in docs if d.page_content.strip()]

        # If no text found, use OCR
        if not docs:
            ocr_text = extract_text_from_scanned_pdf(file_path)
            if ocr_text:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_txt:
                    tmp_txt.write(ocr_text.encode("utf-8"))
                    tmp_txt_path = tmp_txt.name
                loader = TextLoader(tmp_txt_path, encoding="utf-8")
                docs = loader.load()
        return docs

if uploaded_file:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Detect file type
    ext = uploaded_file.name.split(".")[-1].lower()

    # Load and process document
    docs = load_document(tmp_path, ext)

    if not docs:
        st.error("❌ No readable text found. Try another file.")
    else:
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        splits = [s for s in splits if s.page_content.strip()]

        if not splits:
            st.error("❌ No valid text chunks extracted.")
        else:
            # Create embeddings and FAISS index
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(splits, embedding_model)
            retriever = vectorstore.as_retriever()

            # Ask user question
            query = st.text_input("💬 Ask a question about the document:")

            if query:
                results = retriever.invoke(query)
                if results:
                    st.subheader("📌 Answer")
                    st.write(results[0].page_content)
                else:
                    st.warning("No relevant content found.")
else:
    st.info("Please upload a .txt or .pdf file to get started.")
