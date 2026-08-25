from dotenv import load_dotenv

load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from rich import print
from langchain_core.messages import HumanMessage


@tool
def get_text_length(text: str) -> int:
    """Return a number of character in a given text"""
    return len(text)


@tool
def greeting(name: str) -> str:
    """This is a simple Greeting tool"""
    return f"Hello {name}! Welcome to the GenAI course"


tools = {"get_text_length": get_text_length, "greeting": greeting}

llm = ChatMistralAI(model="mistral-small-2506")

# tool binding
llm_with_tool = llm.bind_tools([get_text_length, greeting])

message = []

user = input("You: ")
query = HumanMessage(user)
message.append(query)


result = llm_with_tool.invoke(message)

message.append(result)


if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)

result = llm_with_tool.invoke(message)
print(result.content)
