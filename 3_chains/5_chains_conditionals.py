from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_nvidia_ai_endpoints import  ChatNVIDIA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# model = GoogleGenerativeAI(model="gemini-2.5-flash")
model = ChatNVIDIA(model="openai/gpt-oss-120b")

positive_feedback_temp = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful agent."),
        ("human", "Generate a thank you note for this positive feedback: {feedback}.")

    ]
)

negative_feedback_temp = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful agent."),
        ("human", "Generate a response addressing this negative feedback: {feedback}.") # Corrected: "positive" to "negative"
    ]
)

neutral_feedback_temp = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful agent."),
        ("human", "Generate a response for this neutral feedback: {feedback}.") # Corrected: "thank you note" to a more general response
    ]
)

escalate_feedbck_template = ChatPromptTemplate.from_messages( # Corrected: use from_messages for consistency
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a message to escalate this feedback to a human agent: {feedback}.")
    ]
)

classification_template = ChatPromptTemplate.from_messages( # Corrected: use from_messages
    [
        ("system", "You are a helpful assistant."),
        ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}.") # Corrected: "potive" to "positive"
    ]
)

branches = RunnableBranch(
    (
        lambda x: "positive" in x.lower(), # Corrected: removed space and added .lower() for robustness
        positive_feedback_temp | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x.lower(), # Corrected: removed space and added .lower()
        negative_feedback_temp | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x.lower(), # Corrected: removed space and added .lower()
        neutral_feedback_temp | model | StrOutputParser()
    ),
    escalate_feedbck_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()
chain = classification_chain | branches

# review = "this product is useless, not nice, poor quality"
review = "this product is good but lacks in reusaility and priced high"
result = chain.invoke({"feedback": review})
print(result)