# ⌚ Timely Customer Support Bot

> An AI-powered customer support chatbot for **Timely** — a Pakistani watches & accessories e-commerce store. Built with LangChain, Groq, and Streamlit.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge)](https://customer-support-bot-22jbd88mzjkrwjrjwgczjl.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Py--saqlain-181717?style=for-the-badge&logo=github)](https://github.com/Py-saqlain/Customer-support-bot)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-1C3C3C?style=for-the-badge)](https://langchain.com)

---

## 🌐 Live Demo

👉 **[Try it here](https://customer-support-bot-22jbd88mzjkrwjrjwgczjl.streamlit.app)**

---

## 🤖 What This Bot Does

This is a production-ready AI customer support agent that:

- 💬 **Answers customer questions** from a real FAQ knowledge base
- 🧠 **Remembers full conversation** — no repeating yourself
- 📄 **Reads from PDF** — answers only from verified Timely data, never makes things up
- 😤 **Detects frustrated customers** — escalates to human agent automatically
- 🇵🇰 **Bilingual** — works in both Urdu and English
- 🌐 **Deployed live** — accessible from any device including mobile

---

## 🏗️ Architecture

```
User Message
     ↓
Escalation Detection (LLM)
     ↓
Is customer angry?
     ├── YES → Empathy message + Human agent contact
     └── NO  → RAG Pipeline
                    ↓
             Search FAISS Vector DB
                    ↓
             Retrieve top 3 relevant FAQ chunks
                    ↓
             Inject into prompt with chat history
                    ↓
             Groq LLM generates response
                    ↓
             Answer displayed in Streamlit UI
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq API — `llama-3.3-70b-versatile` (Free) |
| **Framework** | LangChain |
| **Vector Database** | FAISS (local) |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Memory** | LangChain `InMemoryChatMessageHistory` |
| **PDF Loader** | PyPDFLoader |
| **UI** | Streamlit |
| **Deployment** | Streamlit Cloud |

---

## ✨ Key Features Explained

### 📄 RAG (Retrieval Augmented Generation)
Instead of relying on general AI knowledge, the bot reads from a custom **Timely FAQ PDF** covering return policy, warranty, delivery, payment methods, and products. It never makes up answers.

### 🧠 Conversation Memory
Uses `RunnableWithMessageHistory` to maintain full conversation context. The bot remembers your name, previous questions, and complaint history throughout the session.

### 😤 Smart Escalation Detection
A separate LLM chain analyzes every message for signs of frustration, anger, fraud mentions, or legal threats. If detected, the bot immediately shows empathy and connects the customer to a human agent — before even attempting to answer.

### 🌍 Bilingual Support
Escalation detection and responses work in both **English and Urdu**, making it suitable for Pakistani market customers.

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Py-saqlain/Customer-support-bot.git
cd Customer-support-bot
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free key at [console.groq.com](https://console.groq.com)

### 5. Generate FAQ PDF
```bash
python create_faq.py
```

### 6. Run the app
```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser 🎉

---

## 📁 Project Structure

```
Customer-support-bot/
│
├── streamlit_app.py      # Main Streamlit UI app
├── app.py                # Terminal version
├── create_faq.py         # Generates Timely FAQ PDF
├── Timely_faq.pdf        # Knowledge base
├── requirements.txt      # All dependencies
├── .env                  # API keys (never commit this!)
└── .gitignore
```

---

## 💬 Example Conversations

**Normal Query:**
```
User: What is your return policy?
Bot:  You can return any product within 7 days of delivery.
      The product must be unused and in original packaging...
```

**Escalation:**
```
User: This is fraud! You took my money and sent nothing!
Bot:  I completely understand your frustration and sincerely
      apologize for the inconvenience caused. 😔
      
      📞 Please call: 0327-0337903
      📧 Or email:   support@timely.pk
```

**Memory:**
```
User: My name is Ahmed
Bot:  Nice to meet you Ahmed!
User: What is my name?
Bot:  Your name is Ahmed! How can I help you today?
```

---

## 📦 Requirements

```txt
langchain
langchain-groq
langchain-core
langchain-community
langchain-text-splitters
langchain-huggingface
python-dotenv
streamlit
pypdf
faiss-cpu
sentence-transformers
```

---

## 🎓 What I Learned Building This

- **RAG pipeline** — connecting PDF knowledge to LLMs
- **Vector databases** — how FAISS stores and searches embeddings
- **LangChain memory** — managing stateful conversations
- **Prompt engineering** — designing system prompts for specific behaviors
- **Streamlit deployment** — building and deploying AI apps fast

---

## 👨‍💻 Author

**Saqlain** — BS Software Engineering @ PUCIT Lahore

[![GitHub](https://img.shields.io/badge/GitHub-Py--saqlain-181717?style=flat&logo=github)](https://github.com/Py-saqlain)

---

## ⭐ If you found this useful, give it a star!
