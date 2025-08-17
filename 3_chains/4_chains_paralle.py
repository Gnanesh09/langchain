from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda,RunnableParallel

load_dotenv()

model = GoogleGenerativeAI(model="gemini-2.5-flash")

# defirn summary template for a movie summary
summary_template = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critics"),
    ("human", "Write a summary of the movie {movie_name}")

])


# define plot analysis template
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages([
        ("system", "You are a amovie critics"),
        ("human", "analyze the plot: {plot}, what are its strength and weakness")

    ])
    return plot_template.format_prompt(plot=plot)
def analyze_characters(characters):
    characters_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie critics"),
        ("human", "analyze the characters: {characters}, what are their strengths and weaknesses")
    ])
    return characters_template.format_prompt(characters=characters)


def combine_verdict(plot_analysis, characters_analysis):
    return f"plot analysis: \n{plot_analysis}\n\ncharacters analysis: \n{characters_analysis}"

plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x) ) | model | StrOutputParser()
) 

characters_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | model | StrOutputParser()
)

chain= (
    summary_template|
    model|
    StrOutputParser()|
    RunnableParallel(
        branches={
            "plot":plot_branch_chain,
            "characters": characters_branch_chain
        },
    )|
    RunnableLambda(lambda x: combine_verdict(x["branches"]["plot"], x["branches"]["characters"])) 


)
movi = input("enter movie name: ")
result = chain.invoke({"movie_name": movi})

print(result)