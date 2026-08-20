from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain.chains import LLMChain

load_dotenv()

prompt = ChatPromptTemplate.from_template("Explain this {topic} in simple words")

llm = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

formated_prompt = prompt.format_messages(topic="Machine learning")

chain = prompt | llm | parser

response = chain.invoke("Machine learning")
print(response)
