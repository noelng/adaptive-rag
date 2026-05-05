from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama

load_dotenv()

# chat model initialization
# llm_model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0
# )

llm_model = ChatOllama(
    model="gemma3:1b",
    temperature=0
)

# embedding model initialization
embed_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)