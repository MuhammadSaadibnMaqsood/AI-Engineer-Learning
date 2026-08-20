from dotenv import load_dotenv

load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

llm = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()
short_prompt = ChatPromptTemplate.from_template("Explain this {topic} in 1-2 lines")
long_prompt = ChatPromptTemplate.from_template("Explain this {topic} in detail")

topic = "Machine Learning"

chain = RunnableParallel(
    {
        "short": RunnableLambda(lambda x: x["short"]) | short_prompt | llm | parser,
        "detailed": RunnableLambda(lambda x: x["detailed"])
        | long_prompt
        | llm
        | parser,
    }
)

response = chain.invoke(
    {"short": {"topic": "Machine learning"}, "detailed": {"topic": "Deep learning"}}
)

print("CHAIN 1 ANSWER\n")
print(response["short"])
print("\n\nCHAIN 2 ANSWER")
print(response["detailed"])
