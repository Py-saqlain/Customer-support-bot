import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


# @st.cache_resource — runs ONCE, cached forever
# Without this: PDF reloads + model downloads every message!
# Everything heavy goes inside this function

@st.cache_resource
def load_bot():
    # Load and chunk PDF
    loader = PyPDFLoader("Timely_faq.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    # Embeddings & FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile"
    )

    # Main prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly and professional customer support 
         agent for Timely - a Pakistani Watches e-commerce store.
         
         Use the following information from Timely FAQ to answer the customer:
         {context}
         
         Rules:
         - Answer ONLY from the context provided above
         - If the answer is not in the context, say "I don't have that info, 
           please call 0327-0337903"
         - Be polite, concise, and helpful
         - Always respond in the same language the customer uses"""),
        ("placeholder", "{chat_history}"),
        ("human", "{user_message}")
    ])

    # format_docs helper
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG chain
    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["user_message"])),
            "user_message": RunnablePassthrough() | (lambda x: x["user_message"]),
            "chat_history": RunnablePassthrough() | (lambda x: x.get("chat_history", []))
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Memory
    store = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_memory = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="user_message",
        history_messages_key="chat_history"
    )

    # Escalation chain
    escalation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an escalation detector for Timely customer support.
         
         Analyze the customer message and reply with ONLY one word:
         YES - if customer shows any of these signs:
           - Extreme anger or frustration
           - Mentions fraud, scam, or cheat
           - Legal threats or police mention
           - Completely lost trust
           - Using abusive language
           - Desperate or begging tone
         
         NO - if customer is:
           - Asking a normal question
           - Mildly frustrated but still calm
           - Just following up politely
         
         Reply with ONLY the word YES or NO. Nothing else."""),
        ("human", "{user_message}")
    ])

    escalation_chain = escalation_prompt | llm | StrOutputParser()

    return chain_with_memory, escalation_chain



# PAGE CONFIG — browser tab title and icon
# Must be first Streamlit command in the script

st.set_page_config(
    page_title="Timely Support Bot",
    page_icon="⌚",
    layout="centered"
)
# header
st.title("Timely Customer Support")
st.caption("Powered by AI — Ask me anything about orders, watches, and policies!")
st.divider()


# LOAD BOT — cached, runs once

with st.spinner("Loading Timely knowledge base..."):
    chain_with_memory, escalation_chain = load_bot()


# SESSION STATE — persists chat history between reruns
# First run: creates empty messages list
# Every rerun: messages list already exists, skip creation

if "messages" not in st.session_state:
    st.session_state.messages = []


# DISPLAY CHAT HISTORY
# Loops through all previous messages and renders them
# This runs every rerun to redraw the full chat

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# CHAT INPUT — text box at bottom of screen
# Returns None if user hasn't typed anything yet
# Returns the typed text when user hits Enter

user_input = st.chat_input("Type your message here...")

if user_input:

    # Show user message immediately in chat
    with st.chat_message("user"):
        st.write(user_input)

    # Save user message to session state
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Escalation check first
    escalation_result = escalation_chain.invoke(
        {"user_message": user_input}
    ).strip().upper()

    if escalation_result == "YES":
        # Escalation response
        escalation_message = """I completely understand your frustration and 
I sincerely apologize for the inconvenience caused. 

This situation needs immediate personal attention.
I am connecting you to a human agent right now.

 **Please call:** 0327-0337903
 **Or email:** support@timely.pk

Our team will resolve this for you as soon as possible.
Thank you for your patience. """

        with st.chat_message("assistant"):
            st.write(escalation_message)

        st.session_state.messages.append({
            "role": "assistant",
            "content": escalation_message
        })

    else:
        # Normal RAG response with spinner
        with st.spinner("Thinking..."):
            response = chain_with_memory.invoke(
                {"user_message": user_input},
                config={"configurable": {"session_id": "Timely_user_1"}}
            )

        with st.chat_message("assistant"):
            st.write(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })


# streamlit return policy problems: 
# solution(st.cache & st.spinner)
# Every time user sends a message:
# Streamlit RERUNS the entire app.py from line 1!

# This means:
#  All variables reset to zero
#  Chat history disappears
#  PDF reloads every message (slow!)
#  Model downloads again (very slow!)