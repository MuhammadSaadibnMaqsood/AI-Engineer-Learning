import os
os.environ.setdefault("USER_AGENT", "RAG-Loader/1.0 (+https://example.com)")

from langchain_community.document_loaders import WebBaseLoader

url = 'https://www.apple.com/mac/'

loader = WebBaseLoader(url)

data = loader.load()

print(data[0].page_content)