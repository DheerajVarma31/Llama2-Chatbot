import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Custom Prompt Template
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Provide a complete and helpful answer to the user's question. Be detailed and accurate.
Helpful answer:

"""

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm():
    return CTransformers(
        model=r"C:\Users\dheer\Llama2-Chatbot\model\llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        config={
            'max_new_tokens': 512,     # 🔼 Increased from 256
            'temperature': 0.7,        # 🔼 Slightly higher for more creative/longer responses
            'top_k': 50,
            'top_p': 0.95,
            'repetition_penalty': 1.1
        }
    )


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    prompt = set_custom_prompt()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 4}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

# --- Streamlit App Start ---
st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.title("🩺 Medical Q&A Bot")

# Initialize chatbot only once
if "chain" not in st.session_state:
    with st.spinner("Initializing Medical Bot..."):
        st.session_state.chain = qa_bot()
        st.success("Hello! How can I help you today?")

# Input from user
user_query = st.chat_input("Type your question here...")

if user_query:
    user_query = user_query.strip().lower()

    # Exit condition
    if user_query in ["exit", "quit", "bye", "goodbye"]:
        st.chat_message("assistant").write("Goodbye! 👋 Stay healthy.")
    else:
        st.chat_message("user").write(user_query)

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

            st.chat_message("assistant").write(answer)

        except Exception as e:
            st.chat_message("assistant").error(f"❌ Error: {str(e)}")
