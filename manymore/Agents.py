from dotenv import load_dotenv

load_dotenv()
from langchain_mistralai import ChatMistralAI
import os
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient

API_KEY = os.getenv("OPENWHEATHER_API_KEY")


@tool
def get_wheather(city: str) -> str:
    """Get current wheather of a city"""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={API_KEY}&units=metric"
    response = requests.get(url)  # Fixed: changed `response.get` to `requests.get`
    data = response.json()
    if (
        str(data.get("cod")) != "200"
    ):  # Fixed: changed 'code' to OpenWeather API's 'cod'
        return f"Error: {data.get('message', 'could not fetch wheather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]  # Fixed: changed 'wheather' to 'weather'

    return f"Wheather in {city}: {desc}, {temp}C "


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
