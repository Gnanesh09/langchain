from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm= GoogleGenerativeAI(model="gemini-1.5-flash")


# example 1
# temp = "write a startup pitch form in a {tone} to {comany} telling about the {idea} in a brief about my startup {statup_name}, explain the {funds_needed},and {regions} to be served , aboyt 12 lines   "
# prompt_template = ChatPromptTemplate.from_template(temp)

# prompt = prompt_template.invoke({
#     "tone":"professional",
#     "comany":"Vanguard",
#     "idea":"a new AI product in agricultue",
#     "funds_needed":"180 million dollars",
#     "regions":"North America and Europe",
#     "statup_name":"Agroolia"}
# )
# print(prompt)
# print("***********************************")

# result = llm.invoke(prompt)
# print(result)



# ////////////////////////EXAMPLE 2////////////////////////
# PROMPTING BOTH SYSTEM AND HUMAN MESSAGE
messages = [
    ("human", "shwo me available {airline} flights from {from_city} to {to_city}"),
    ("system", "You are a helpful travel assistant that provides flight information {flight_info}, give response in format "),

]

prompt_template = ChatPromptTemplate.from_messages(messages)

promt = prompt_template.invoke({
    "airline": ["Delta", "emirates","qatar airways"],
    "from_city": "New York",
    "to_city": "Los Angeles",
    "flight_info": "randomly say flight times, prices, and airlines mentioned by human."
    
})

print(promt)
print("***********************************")
result = llm.invoke(promt)
print(result)
