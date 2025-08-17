from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
model = GoogleGenerativeAI(model="gemini-1.5-flash")
prompt_templates = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a facts xpert who knows about {animal}"),
        ("human","tell me some {fact_count} facts" )
    ]
)

# create the combines chain using langchain expression language (LCEL)
chain = prompt_templates | model | StrOutputParser()
# chain = prompt_templates | model 


# run the chain
result = chain.invoke({
    "animal": "lion",
    "fact_count": 5
})

print(result)


