from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


loader = PyPDFLoader("RAG_Project/document-loaders/Covariance vs Correlation.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_documents(docs)

print("Length:", len(chunks))
print("Content:", chunks[0])

template = ChatPromptTemplate.from_messages(
    [("system", "You are the AI that summarize the content"), ("human", "{data}")]
)
model = ChatMistralAI(model="mistral-small-2506", temperature=0.3)

prompt = template.format_messages(data=docs[0].page_content)
result = model.invoke(prompt)
print(result.content)
