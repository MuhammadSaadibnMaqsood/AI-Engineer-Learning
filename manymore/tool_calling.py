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
def greeting(name: str) -> str:
    """This is a simple Greeting tool"""
    return f"Hello {name}! Welcome to the GenAI course"


llm = ChatMistralAI(model="mistral-small-2506")

# tool binding
llm_with_tool = llm.bind_tools([get_text_length])

result = llm_with_tool.invoke(
    "Return the number of character in a given text: 'Hello how are you ? '"
)

if result.tool_calls:
    tool_call = result.tool_calls[0]
    tool_result = get_text_length.invoke(tool_call["args"])

    final_response = llm.invoke(f"The length of the text is {tool_result}")

    print(final_response)
