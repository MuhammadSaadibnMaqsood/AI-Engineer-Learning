import getpass
import os
from langchain_community import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(
        page_content="Python is widely use in AI!", metadata={"source": "Python AI"}
    ),
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "LangChain"},
    ),
    Document(
        page_content="MistralAI is a company that provides AI solutions.",
        metadata={"source": "MistralAI"},
    ),
]


if not os.environ.get("MISTRALAI_API_KEY"):
    os.environ["MISTRALAI_API_KEY"] = getpass.getpass("Enter API key for MistralAI: ")


embeddings = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma.from_documents(
    document=docs, embedding=embeddings, persist_directory="chroma-DB"
)
