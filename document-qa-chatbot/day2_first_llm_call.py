import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature = 0
)
print("making first LLM call..\n")
response = llm.invoke("wht is RAG in one sentence ?")
res = llm.invoke("how is president of india ?")
print("response:..")
print(res.content)
print(response.content)