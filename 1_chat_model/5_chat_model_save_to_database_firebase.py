from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import GoogleGenerativeAI
from google.cloud import firestore
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from dotenv import load_dotenv
load_dotenv()
PROJECT_ID = "langchain1-f050e"
SESSION_ID = "agentics15"
# SESSION_ID = "agentics14"
COLLECTON_NAME = "chat_messages_history"


print("initizalizing firestore....")
client = firestore.Client(project=PROJECT_ID)

print("initializing firestore chjat messages history")

chat_history =FirestoreChatMessageHistory(
    session_id = SESSION_ID,
    collection=COLLECTON_NAME,
    client=client

)
print("initialized chat history")
print(f"current chat histroty: {chat_history.messages}")

# model = ChatNVIDIA(model="deepseek-ai/deepseek-r1")
model = GoogleGenerativeAI(
    model="gemini-2.5-flash"
)
print("starting chat loop")


while True:
    human_input= input("You: ")
    if human_input.lower() == "exit":
        break
    chat_history.add_user_message(human_input)
    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response)
    print(f"AI: {ai_response}")

print("========message history========")
print(chat_history)