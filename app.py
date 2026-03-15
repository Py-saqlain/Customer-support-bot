import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# STEP 1: Load API key
load_dotenv()

# STEP 2: Initialize LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# STEP 3: Prompt Template (with chat_history placeholder)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly and professional customer support 
     agent for ShopEase - a Pakistani electronics e-commerce store.
     You help customers with orders, products, returns, and complaints.
     Always be polite, concise, and helpful.
     If you don't know something, say you'll look into it."""),
    ("placeholder", "{chat_history}"),  # full history injected here
    ("human", "{user_message}")
])

# STEP 4: Base chain
chain = prompt | llm | StrOutputParser()

# STEP 5: Memory store
# Dictionary to hold conversation history per session
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Returns existing history or creates new one for this session"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap chain with memory
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="user_message",
    history_messages_key="chat_history"
)

# STEP 6: Chat loop
print("=" * 40)
print("  ShopEase Customer Support Bot 🛒")
print("=" * 40)
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "quit":
        print("Bot: Thank you for contacting ShopEase. Goodbye! 👋")
        break

    response = chain_with_memory.invoke(
        {"user_message": user_input},
        config={"configurable": {"session_id": "shopease_user_1"}}
    )
    print(f"\nBot: {response}\n")
