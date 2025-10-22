# streamlit_rag_chatbot.py  — modernized for LangChain >= 0.2/0.3
import os
import base64
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF

# LangChain / OpenAI (new split packages)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# -------------------------
# Env & API key
# -------------------------
load_dotenv("api.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------------
# Load FAISS index & retriever
# 注意：为兼容你已生成的索引，这里继续使用 text-embedding-ada-002
# 如果将来重建索引，可考虑 text-embedding-3-small
# -------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.load_local(
    folder_path="faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True  # 读取自有索引文件所需
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# LLM
# -------------------------
llm = ChatOpenAI(model="gpt-4", temperature=0.1, max_tokens=1024)

# -------------------------
# Prompts & Chains (替代 ConversationalRetrievalChain)
# 1) 历史感知的查询改写
# 2) 文档融合回答（stuff）
# 3) 检索链拼装
# -------------------------
contextualize_q = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the user's latest message, "
     "rewrite the user message into a standalone search query about Orangetheory Fitness."
     " Do not answer the question, only rewrite it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful Orangetheory Fitness expert. "
     "Use the following context to answer the user's question. "
     "If the answer is not in the context, say you don't know.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=contextualize_q
)

doc_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    doc_chain
)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Orangetheory Chatbot", page_icon="🧡")
st.title("🧡 Orangetheory Q&A Chatbot")

if "chat_history" not in st.session_state:
    # 保存为 [(role, text)]，role ∈ {"You","AI"}
    st.session_state.chat_history = []

def to_lc_messages(hist):
    """[(role, text)] -> [HumanMessage/AIMessage]"""
    out = []
    for role, text in hist:
        if role == "You":
            out.append(HumanMessage(content=text))
        else:
            out.append(AIMessage(content=text))
    return out

query = st.text_input(
    "Ask a question about Orangetheory:",
    placeholder="e.g. What is the orange zone?"
)

if query:
    lc_history = to_lc_messages(st.session_state.chat_history)
    result = rag_chain.invoke({"input": query, "chat_history": lc_history})

    # 兼容不同版本输出键名：优先 'answer'，其次 'output_text'
    answer = result.get("answer") or result.get("output_text") or ""
    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("AI", answer))

    st.markdown("### 📌 Answer")
    st.write(answer)

    # 展示来源文档（create_retrieval_chain 通常返回 'context' 为 List[Document]）
    with st.expander("📄 Source Documents"):
        docs = result.get("context", [])
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source", "Unknown")
            st.markdown(f"**Source {i+1}** - {src}")
            st.write((doc.page_content or "")[:500] + "...")

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
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        href = f'<a href="data:application/pdf;base64,{base64_pdf}" download="chat_log.pdf">📥 Click here to download PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
