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

# load Api key
load_dotenv()

# step 2
print("Loading Timely knowledge base...")

# Naming the generated PDF 
loader = PyPDFLoader("Timely_faq.pdf")
documents = loader.load()

# splitting the text every 500 characters/chunk and 50 characters from end of chunk 1 to start of chunk 2
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"PDF split into {len(chunks)} chunks")

# embeddings convert text to numbers for computer readibilty with help of AI model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS stores all chunk vectors locally on your machine

vectorstore = FAISS.from_documents(chunks, embeddings)


# when user asks anything, retriever finds 3 best matching chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Knowledge base ready!\n")


# STEP 3: Initialize LLM — same as before

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)


# STEP 4: Prompt Template has 3 placeholders


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


# STEP 5: x["user_message"] → get what user typed
          #retriever.invoke(...) → search FAISS, find 3 relevant chunks
         #format_docs(...) → join chunks into one clean string

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# STEP 6: RAG Chain RunnablePassthrough()just passes the input through unchanged. Just extracts `user_message` from the input dictionary and passes it forward

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


# STEP 7
store = {} # empty dict living in ram

#user exists in store? → return their history
#user doesn't exist?  → create new empty history, return it
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
 # gets the history of user
chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="user_message",
    history_messages_key="chat_history"
)


# STEP 8: Chat loop with RAG
print("=" * 40)
print("  Timely Customer Support Bot ⌚")
print("=" * 40)
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "quit":
        print("Bot: Thank you for contacting Timely. Goodbye! 👋")
        break

    response = chain_with_memory.invoke(
        {"user_message": user_input},
        config={"configurable": {"session_id": "Timely_user_1"}}
    )
    print(f"\nBot: {response}\n")