from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash"
)

messages = [
    SystemMessage("your are aan expert in socail media content strategy"),
    HumanMessage("give ideas creat a post on fashion on insta")]

result = llm.invoke(messages)
print(result)