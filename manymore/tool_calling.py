from dotenv import load_dotenv

load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from rich import print


@tool
def get_text_length(text: str) -> int:
    """Return a number of character in a given text"""
    return len(text)

@tool
def greeting(name:str) -> str:
    """This is a simple Greeting tool"""
    return f"Hello {name}! Welcome to the GenAI course"


llm = ChatMistralAI(model="mistral-small-2506")

# tool binding

llm_with_tool = llm.bind_tools([get_text_length])

