import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import GoogleGenerativeAI
# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("GOOGLE_API_KEY")
api_key2 = os.getenv("NVIDIA_API_KEY")

if api_key and api_key2:
    # Use the API key
    print("API key loaded successfully!")
else:
    print("API key not found.")

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash")
llm2 = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
# llm2 = ChatNVIDIA(model="meta/llama-4-maverick-17b-128e-instruct") 

# result = llm.invoke("what's the population of India in 2025?")
result = llm2.invoke("calculate 5+5")
print(result)