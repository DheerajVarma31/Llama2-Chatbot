import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Prompt template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def load_llm():
    return CTransformers(
        model=r"C:\Users\dheer\Llama2-Chatbot\model\llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        config={
            'max_new_tokens': 256,
            'temperature': 0.5
        }
    )

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, prompt, db)

# --- Streamlit App Start ---
st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.title("🩺 Medical Q&A Bot")

# Initialize session state
if "chain" not in st.session_state:
    with st.spinner("Initializing chatbot..."):
        st.session_state.chain = qa_bot()
        st.session_state.chat_history = []

# Chat UI
user_query = st.chat_input("Ask your medical question...")

if user_query:
    st.session_state.chat_history.append(("user", user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    try:
        chain = st.session_state.chain
        res = chain.invoke({"query": user_query})
        answer = res.get("result", "Sorry, I couldn't find an answer.")

        sources = res.get("source_documents", [])
        if sources:
            unique_sources = {
                f"{doc.metadata.get('source', 'unknown').split(os.sep)[-1]} (page {doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))})"
                for doc in sources
            }
            answer += "\n\n**Sources:**\n" + "\n".join(unique_sources)
        else:
            answer += "\n\n_No sources found._"

        st.session_state.chat_history.append(("bot", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        st.session_state.chat_history.append(("bot", error_msg))
        with st.chat_message("assistant"):
            st.error(error_msg)

# Display chat history (on refresh)
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

