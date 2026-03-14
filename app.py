import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# STEP 1: Load API key
load_dotenv()

# STEP 2: Initialize LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# STEP 3: Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly and professional customer support 
     agent for ShopEase - a Pakistani electronics e-commerce store.
     You help customers with orders, products, returns, and complaints.
     Always be polite, concise, and helpful.
     If you don't know something, say you'll look into it."""),
    ("human", "{user_message}")
])

# STEP 4: Chain( input + Thinking + output)
chain = prompt | llm | StrOutputParser()

# STEP 5: Chat loop
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

    response = chain.invoke({"user_message": user_input})
    print(f"\nBot: {response}\n")
