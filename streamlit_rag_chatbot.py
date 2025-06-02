# streamlit_rag_chatbot.py

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from fpdf import FPDF
import base64

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

# 💬 Styled Chat History with Avatars
st.markdown("---")
st.markdown("### 💬 Chat History")
for speaker, message in st.session_state.chat_history:
    avatar = "🧑" if speaker == "You" else "🤖"
    bubble_color = "#F0F8FF" if speaker == "You" else "#E6FFE6"
    st.markdown(f"""
    <div style='background-color:{bubble_color};padding:10px;border-radius:10px;margin-bottom:10px;'>
        <strong>{avatar} {speaker}:</strong><br>
        <div style='margin-left:10px'>{message}</div>
    </div>
    """, unsafe_allow_html=True)

# 📥 Export chat to PDF
if st.button("📄 Export Q&A to PDF"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Orangetheory Q&A Chat Log", ln=True, align="C")
    pdf.ln(10)
    for speaker, msg in st.session_state.chat_history:
        pdf.multi_cell(0, 10, f"{speaker}: {msg}\n")

    pdf_output_path = "chat_log.pdf"
    pdf.output(pdf_output_path)

    with open(pdf_output_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="chat_log.pdf">📥 Click here to download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
