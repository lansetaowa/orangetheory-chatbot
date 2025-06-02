# 🧡 Orangetheory RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot built with LangChain, OpenAI, FAISS, and Streamlit. It provides context-aware answers based on official Orangetheory Fitness documentation and Wikipedia articles.

> 🔗 [Live App on Streamlit Cloud](https://orangetheory-chatbot.streamlit.app)

---

## 🧠 Features

- 💬 Natural language Q&A about Orangetheory Fitness
- 📄 Based on real documents retrieved from Orangetheory.com and Wikipedia
- 🔎 Uses FAISS for fast semantic similarity search
- 🧠 GPT-4 powered conversational agent
- 🧾 Styled multi-turn chat history with avatars
- 📥 Export full chat log to downloadable PDF

---

## 🚀 Technologies

| Component           | Stack                     |
|---------------------|---------------------------|
| LLM                 | OpenAI GPT-4     |
| Vector Search       | FAISS (local)             |
| Embedding Model     | OpenAI `text-embedding-ada-002` |
| UI Framework        | Streamlit                 |
| AI Engine Framework | LangChain                 |

---

## 🧰 How It Works

1. Documents are scraped from Orangetheory’s official site and Wikipedia.
2. Text is split and embedded using OpenAI Embeddings.
3. FAISS builds a vector index for similarity search.
4. User questions retrieve top relevant chunks from the knowledge base.
5. A custom prompt feeds retrieved chunks + user query into the LLM.
6. AI generates detailed, context-rich answers with document references.

---

## 🛠 Local Setup

### 1. Clone this repo

```bash
git clone https://github.com/lansetaowa/orangetheory-chatbot.git
cd orangetheory-chatbot
