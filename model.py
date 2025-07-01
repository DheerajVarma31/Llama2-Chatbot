import os
import torch
import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

# Load LLM with optional GPU layers
def load_llm():
    return CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        config={"gpu_layers": 100},  # Reduce to 20–50 if you get CUDA OOM errors
        max_new_tokens=512,
        temperature=0.5
    )

# QA Bot setup
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': DEVICE}
    )

    # FAISS on CPU (Windows-safe)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = load_llm()
    prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, prompt, db)

# Chainlit start handler
@cl.on_chat_start
async def start():
    chain = qa_bot()
    cl.user_session.set("chain", chain)
    await cl.Message(content="Hi, Welcome to Medical Bot. What is your query?").send()

# Chainlit message handler
@cl.on_message
async def main(message: cl.Message):
    user_input = message.content.strip().lower()

    if user_input in ["exit", "quit", "bye", "goodbye"]:
        await cl.Message(content="Goodbye! 👋 Feel free to come back anytime.").send()
        return

    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="⚠️ Error: Chain not initialized.").send()
        return

    try:
        res = await chain.ainvoke({"query": message.content})
        answer = res.get("result", "Sorry, I couldn't find an answer.")

        sources = res.get("source_documents", [])
        if sources:
            unique_sources = {
                f"{doc.metadata.get('source', 'unknown').split(os.sep)[-1]} (page {doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))})"
                for doc in sources
            }
            answer += "\n\nSources:\n" + "\n".join(unique_sources)
        else:
            answer += "\n\nNo sources found."

        await cl.Message(content=answer).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()
