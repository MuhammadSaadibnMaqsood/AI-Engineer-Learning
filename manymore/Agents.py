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
