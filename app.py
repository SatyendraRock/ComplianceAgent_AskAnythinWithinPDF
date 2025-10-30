import streamlit as st
from io import BytesIO
import pdfplumber
import fitz  # PyMuPDF
import easyocr
from PIL import Image
import numpy as np
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.title("📄 Document Search with OCR and FAISS")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
if uploaded_file is not None:
    try:
        # Read text content based on file type
        raw_text = ""
        if uploaded_file.type == "text/plain":
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            # Try extracting text with pdfplumber
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        raw_text += page_text + "\n"
                    else:
                        # Use OCR for scanned PDFs
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        page = doc.load_page(page.page_number - 1)
                        pix = page.get_pixmap(dpi=150)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        reader = easyocr.Reader(['en'], gpu=False)
                        result = reader.readtext(np.array(img), detail=0)
                        raw_text += " ".join(result) + "\n"
        else:
            st.error("Unsupported file type.")
            st.stop()

        if not raw_text.strip():
            st.error("No extractable text found in the document.")
            st.stop()

    except Exception as e:
        st.error(f"Error extracting text: {e}")
        st.stop()

    # Split text
    try:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(raw_text)
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        st.stop()

    # Embed and index
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        faiss_index = FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        st.error(f"Error creating embeddings or FAISS index: {e}")
        st.stop()

    # Query
    query = st.text_input("Enter a query:")
    if query:
        try:
            results = faiss_index.similarity_search(query, k=5)
            if results:
                st.subheader("Top matching document excerpts:")
                for i, doc in enumerate(results, start=1):
                    st.write(f"**Result {i}:** {doc.page_content}")
            else:
                st.info("No matching text found.")
        except Exception as e:
            st.error(f"Search error: {e}")
