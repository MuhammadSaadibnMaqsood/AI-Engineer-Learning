from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = TextLoader("RAG_Project/document-loaders/notes.txt")
docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system", "You are the AI that summarize the content"), ("human", "{data}")]
)
model = ChatMistralAI(model="mistral-small-2506", temperature=0.3)

prompt = template.format_messages(data=docs[0].page_content)
result = model.invoke(prompt)
print(result.content)
