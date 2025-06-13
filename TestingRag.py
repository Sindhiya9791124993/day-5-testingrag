import os
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# --- HARDCODE GOOGLE API KEY (TEMPORARY TESTING PURPOSE ONLY) ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyABT-OfEjwUM40w-Jhkc5V1ZcZ6MWEAvsk"

# --- Streamlit UI ---
st.set_page_config(page_title="Gemini PDF QA", layout="centered")
st.title("📄 PDF Q&A with Gemini 2.0 Flash")
st.markdown("Upload a PDF and ask questions. Answers are powered by Gemini, without embeddings.")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
user_question = st.text_input("Enter your question")

if uploaded_file and user_question:
    # --- Save PDF to temp file ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # --- Load and extract full text from PDF ---
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    full_text = "\n\n".join(doc.page_content for doc in documents)

    # --- Gemini LLM ---
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # --- Prompt Template ---
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the following document content to answer the question.

    Document:
    {context}

    Question:
    {question}

    Answer:""")

    # --- Format input and call LLM ---
    final_prompt = prompt.format_messages(context=full_text, question=user_question)
    with st.spinner("Generating answer..."):
        response = llm.invoke(final_prompt)

    # --- Display result ---
    st.subheader("📌 Answer")
    st.write(response.content)

    st.subheader("📚 Raw Extracted Document Preview")
    st.write(full_text[:1000] + "...")  # Optional: limit to first 1000 chars
