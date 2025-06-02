# streamlit_rag_chatbot.py

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import os

# load api
load_dotenv("api.env")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load FAISS index and retriever
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.load_local(folder_path="faiss_index",
                               embeddings=embedding_model,
                               allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Define prompt template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    The following is a friendly conversation between a human and an AI.
    The AI is talkative and provides structured message with relevant and 
    specific details from its context about Orangetheory Fitness.

    Context:
    {context}

    Human: {question}
    AI:
    """
)

# Define memory
memory = ConversationBufferMemory(memory_key="chat_history",
                                  return_messages=True,
                                  output_key='answer')

# Build Conversational QA chain with memory
llm = ChatOpenAI(model="gpt-4", temperature=0.1, max_tokens=1024)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    return_source_documents=True,
    output_key='answer'
)

# Streamlit UI
st.set_page_config(page_title="Orangetheory Chatbot", page_icon="🧡")
st.title("🧡 Orangetheory Q&A Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about Orangetheory:", placeholder="e.g. What is the orange zone?")

if query:
    result = qa_chain.invoke({"question": query})
    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("AI", result["answer"]))

    st.markdown("### 📌 Answer")
    st.write(result["answer"])

    with st.expander("📄 Source Documents"):
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Source {i+1}** - {doc.metadata.get('source', 'Unknown')}")
            st.write(doc.page_content[:500] + "...")

    st.markdown("---")
    st.markdown("### 💬 Chat History")
    for speaker, message in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {message}")
