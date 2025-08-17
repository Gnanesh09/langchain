from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
load_dotenv()

model = GoogleGenerativeAI(model="gemini-1.5-flash")

animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows about {animal}"),
        ("human", "Tell me some {fact_count} facts about {animal}.")
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translation expert who translates text into {language} "),
        ("human", "Translate the following tex to {language}: {text}"),
    ]
)

prepare_for_translation = RunnableLambda(lambda output:
                                         {"text": output, "language": "kannada"})

chain = animal_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser()

result = chain.invoke({
    "animal": "lion",
    "fact_count": 2
})

print(result)