from langchain_text_splitters import CharacterTextSplitter
from pathlib import Path
from langchain_community.document_loaders import TextLoader

BASE_DIR = Path(__file__).parent
splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 50,
    chunk_overlap = 1
)
loader = TextLoader(BASE_DIR / "notes.txt")

docs = loader.load()
chunks = splitter.split_documents(docs)
print(len(chunks))
print(chunks[7].page_content)