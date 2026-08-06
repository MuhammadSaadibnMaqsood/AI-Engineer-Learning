from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
pdf_path = BASE_DIR / "Covariance vs Correlation.pdf"

loader = PyPDFLoader(str(pdf_path))
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
pdf_documents = loader.load()

texts = text_splitter.split_documents(pdf_documents)

print("Length:", len(texts))
print("Content:", texts[0])
