import getpass
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_tavily import TavilySearch

load_dotenv()

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

tool = TavilySearch(
    max_results=5,
    topic="general",
)

response = tool.invoke({"query": "Latest new of AI in 2026"})

content = [doc["content"] for doc in response["results"]]

llm = ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are an AI assistant that summarizes content into maximum 5 bullet points.",
        ),
        ("human", "summarize this content {content} into 5 bullet points"),
    ]
)

parser = StrOutputParser()
chain = prompt | llm | parser

result = chain.invoke({"content": content})

print("\n", result)