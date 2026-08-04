from pathlib import Path
from langchain_community.document_loaders import TextLoader

BASE_DIR = Path(__file__).parent

loader = TextLoader(BASE_DIR / "notes.txt")

docs = loader.load()

print(docs)