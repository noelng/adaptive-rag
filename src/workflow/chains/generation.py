# from langchain import hub
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from src.models.model import llm_model
from langchain_core.prompts import ChatPromptTemplate

llm = llm_model
# prompt = hub.pull("rlm/rag-prompt")
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
#     ("user", "Context: {context}\n\nQuestion: {question}")
# ])
client = Client()
prompt = client.pull_prompt("rlm/rag-prompt")
generation_chain = prompt | llm | StrOutputParser()

# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question} 
# Context: {context} 
# Answer: