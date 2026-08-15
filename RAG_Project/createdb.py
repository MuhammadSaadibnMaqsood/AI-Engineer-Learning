from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from pathlib import Path
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()


loader = PyPDFLoader("RAG_Project/document-loaders/Covariance vs Correlation.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

print("Length:", len(chunks))


# 1. Set chroma-DB path inside RAG_Project (one level up from vector-store)
SCRIPT_DIR = Path(__file__).resolve().parent
print("PARENT DIR: ", SCRIPT_DIR.parent)
PERSIST_DIR = SCRIPT_DIR.parent / "RAG_Project/chroma-DB"


# 3. Hardcoded API Key
MISTRAL_API_KEY = ""

embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=MISTRAL_API_KEY)

# 4. Create and persist vector store inside RAG_Project/chroma-DB
vector_store = Chroma.from_documents(
    documents=chunks, embedding=embeddings, persist_directory=str(PERSIST_DIR)
)


print("Embedding create successfull")
