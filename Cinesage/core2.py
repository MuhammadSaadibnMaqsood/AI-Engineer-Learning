from dotenv import load_dotenv
import sys

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional, List
from langchain_core.output_parsers import PydanticOutputParser


class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)

model = ChatMistralAI(model="mistral-small-2506")
prompt = ChatPromptTemplate.from_messages(
    [
      ("system", """ 
      Extract movie information from the paragraph and return it in the following format:
      {format_instructions}
      """),
      ("human", """ 
      {text}
      """)
    ]
)

para = input("give your paragraph: ")

final_prompt = prompt.invoke({
    "text": para,
    "format_instructions": parser.get_format_instructions()
})
response = model.invoke(final_prompt)
movie_data = parser.parse(response.content)
print(movie_data)
