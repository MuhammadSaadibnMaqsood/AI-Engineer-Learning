from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

BASE_DIR = Path(__file__).resolve().parent
pdf_path = BASE_DIR / "Covariance vs Correlation.pdf"

loader = PyPDFLoader(str(pdf_path))
pdf = loader.load()


print(f"Loaded {len(pdf)} pages.")
print(pdf[0].page_content[:500])