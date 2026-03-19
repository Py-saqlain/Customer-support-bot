import os
import warnings
warnings.filterwarnings("ignore")

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

# STEP 1: Load API key
load_dotenv()

# STEP 2: Build Knowledge Base
print("Loading Timely knowledge base...")

loader = PyPDFLoader("Timely_faq.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"PDF split into {len(chunks)} chunks")
# converting text to numbers coz our computer understands numbers only
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Knowledge base ready!\n")

# STEP 3: Initialize LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# STEP 4: Main Prompt Template
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

# STEP 5: format_docs helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# STEP 6: RAG Chain
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

# STEP 7: Memory
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

# STEP 8: Escalation Detection Chain (detecting human emotions)
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
       - Repeated complaints about same issue
     
     NO - if customer is:
       - Asking a normal question
       - Mildly frustrated but still calm
       - Just following up politely
     
     Reply with ONLY the word YES or NO. Nothing else."""),
    ("human", "{user_message}")
])

escalation_chain = escalation_prompt | llm | StrOutputParser()

# STEP 9: Chat loop with Escalation
print("=" * 40)
print("  Timely Customer Support Bot ")
print("=" * 40)
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "quit":
        print("Bot: Thank you for contacting Timely. Goodbye! 👋")
        break

    # Escalation check FIRST — before RAG
    escalation_result = escalation_chain.invoke(
        {"user_message": user_input}
    ).strip().upper()

    if escalation_result == "YES":
        print("""
Bot: I completely understand your frustration and I sincerely 
     apologize for the inconvenience caused. 

     This situation needs immediate personal attention.
     I am connecting you to a human agent right now.

      Please call: 0327-0337903
      Or email:   support@timely.pk

     Our team will resolve this for you as soon as possible.
     Thank you for your patience. 
""")
    else:
        response = chain_with_memory.invoke(
            {"user_message": user_input},
            config={"configurable": {"session_id": "Timely_user_1"}}
        )
        print(f"\nBot: {response}\n")
