from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence
load_dotenv()
model = GoogleGenerativeAI(model="gemini-1.5-flash")
prompt_templates = ChatPromptTemplate.from_messages(
    [
        ("system", "you are a facts xpert who knows about {animal}"),
        ("human","tell me some {fact_count} facts" )
    ]
)

# create individual runnables
format_promt = RunnableLambda(lambda x: prompt_templates.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x)

chain   = RunnableSequence(first=format_promt, middle=[invoke_model], last=parse_output)
res = chain.invoke({"animal":"dog", "fact_count": 2 })
print(res)