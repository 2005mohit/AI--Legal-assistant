import streamlit as st
import os
import faiss
import pickle
import numpy as np

import pdfplumber
import docx
import pandas as pd

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# STREAMLIT CONFIG

st.set_page_config(
    page_title="AI Legal Assistant",
    layout="centered"
)

st.title(" AI Legal Assistant")
st.caption(" Legal Document Q&A ")

# LOAD FAISS + METADATA

@st.cache_resource
def load_faiss():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FAISS_PATH = os.path.join(BASE_DIR, "model", "index.faiss")
    META_PATH  = os.path.join(BASE_DIR, "model", "metadata.pkl")

    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

index, metadata = load_faiss()

# LOAD MODELS (CACHED)

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return embed_model, tokenizer, model

embed_model, tokenizer, model = load_models()

# GENERATE ANSWER DIRECTLY (no pipeline)

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        repetition_penalty=2.5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# RAG FUNCTIONS (INDEXED DOCS)

def retrieve(query, top_k=3):
    emb = embed_model.encode([query])
    _, ids = index.search(np.array(emb).astype("float32"), top_k)
    return [metadata[i]["text"] for i in ids[0]]

def answer_from_index(query):
    chunks = retrieve(query)
    answers = []

    for c in chunks:
        prompt = f"""
Answer the question based on the legal text below:

{c[:1000]}

Question:
{query}
"""
        res = generate(prompt)
        answers.append(res)

    return "\n\n".join(answers)

# FILE EXTRACTION

def extract_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_docx(file):
    d = docx.Document(file)
    return "\n".join(p.text for p in d.paragraphs)

def extract_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

def answer_from_uploaded_doc(text, question):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    embeddings = embed_model.encode(chunks)
    temp_index = faiss.IndexFlatL2(embeddings.shape[1])
    temp_index.add(np.array(embeddings).astype("float32"))

    q_emb = embed_model.encode([question])
    _, ids = temp_index.search(np.array(q_emb).astype("float32"), 3)

    answers = []
    for i in ids[0]:
        prompt = f"""
Answer using the document content below:

{chunks[i]}

Question:
{question}
"""
        res = generate(prompt)
        answers.append(res)

    return "\n\n".join(answers)


# SIDEBAR (UPLOAD)

with st.sidebar:
    st.header(" Upload Document (Optional)")
    uploaded_file = st.file_uploader(
        "PDF / DOCX / XLSX",
        type=["pdf", "docx", "xlsx"]
    )

    if st.button(" Clear Chat"):
        st.session_state.messages = []

# CHAT STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# CHAT INPUT

prompt = st.chat_input("Ask a legal question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if uploaded_file:
                    if uploaded_file.name.endswith(".pdf"):
                        text = extract_pdf(uploaded_file)
                    elif uploaded_file.name.endswith(".docx"):
                        text = extract_docx(uploaded_file)
                    else:
                        text = extract_excel(uploaded_file)

                    answer = answer_from_uploaded_doc(text, prompt)
                else:
                    answer = answer_from_index(prompt)

            except Exception as e:
                answer = f" Error: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
