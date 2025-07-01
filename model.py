from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
# Optional: Use LangChain's streaming handler if needed
from langchain.callbacks.streaming_stdout_async import AsyncStreamingStdOutCallbackHandler

import chainlit as cl
import os

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "model/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

import os
import chainlit as cl
from chainlit.step import Step
from chainlit.callbacks import AsyncLangchainCallbackHandler

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content.strip().lower()

    # Handle polite exit
    if user_input in ["exit", "quit", "bye", "goodbye"]:
        await cl.Message(content="Goodbye! 👋 Feel free to come back anytime.").send()
        return

    chain = cl.user_session.get("chain")
    if not chain:
        await cl.Message(content="⚠️ Error: Chain not loaded. Please restart the chat.").send()
        return

    cb = AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    try:
        async with Step(name="Processing your question..."):
            res = await chain.ainvoke({"query": user_input})

        answer = res.get("result", "Sorry, I couldn't find an answer.")

        sources = res.get("source_documents", [])
        if sources:
            source_texts = {
                f"{doc.metadata.get('source', 'unknown').split(os.sep)[-1]} (page {doc.metadata.get('page_label', 'N/A')})"
                for doc in sources
            }
            answer += "\n\nSources:\n" + "\n".join(source_texts)
        else:
            answer += "\n\nNo sources found."

        await cl.Message(content=answer).send()

    except Exception as e:
        await cl.Message(content=f"❌ An error occurred:\n{str(e)}").send()
