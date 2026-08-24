from langchain.tools import tool

@tool
def greeting(name:str) -> str:
    """This is a simple Greeting tool"""
    return f"Hello {name}! Welcome to the GenAI course"

result = greeting.invoke({"name": "Saad"})
print(result)
print(greeting.description)
print(greeting.name)
print(greeting.args)