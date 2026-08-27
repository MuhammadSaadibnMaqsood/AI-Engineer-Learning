from dotenv import load_dotenv
load_dotenv()
import os
import requests
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from tavily import TavilyClient


API_KEY = os.getenv("OPENWEATHER_API_KEY")


@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}°C"


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    response = tavily_client.search(
        query=f"latest news in {city}", search_depth="basic", max_results=3
    )
    results = response.get("results", [])

    if not results:
        return f"No news found for {city}"

    news_list = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")
        news_list.append(f"- {title}\n  🔗 {url}\n  📝 {snippet[:100]}...")

    return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)


llm = ChatMistralAI(model="mistral-small-2506")

# Corrected dictionary mapping to match exact function names
tools = {"get_weather": get_weather, "get_news": get_news}
llm_with_tools = llm.bind_tools([get_weather, get_news])

messages = []

print("CITY INTELLIGENCE SYSTEM")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    messages.append(HumanMessage(content=user_input))

    while True:
        result = llm_with_tools.invoke(messages)
        messages.append(result)

        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                confirm = input(
                    f"\nAgent wants to call tool '{tool_name}' with args {tool_call['args']}. Approve? (y/n): "
                )

                if confirm.lower() not in ["y", "yes"]:
                    tool_result = (
                        "User denied permission to run this tool call."
                    )
                else:
                    tool_result = tools[tool_name].invoke(tool_call)

                # Always send a ToolMessage back to keep conversation state intact
                messages.append(
                    ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
                )
            continue
        else:
            print(f"\nAgent: {result.content}\n")
            break