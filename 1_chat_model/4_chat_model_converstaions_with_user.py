from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA


from dotenv import load_dotenv
load_dotenv()
# llm = GoogleGenerativeAI(
#     model="gemini-2.5-flash"
# )

model = ChatNVIDIA(model="deepseek-ai/deepseek-r1")
# model = GoogleGenerativeAI(
#     model="gemini-2.5-flash"
# )

chat_history = []
system_message = SystemMessage("you are an expert ai assistant")
chat_history.append(system_message)

# chat Loop

while True:
    query = input("You: ")
    if query=="exit":
     break
    chat_history.append(HumanMessage(content=query)) #add user message

    # get ai response using hiatory
    result = model.invoke(chat_history)
    response = result
    chat_history.append(response)  # add ai response
    print(f"AI: {response}")


print("========message history========")
print(chat_history)
