from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")
embedding_model = MistralAIEmbeddings()
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)

# prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)
print("RAG SYSTEM CREATED")
print("press 0 for exit")

while True:
    query = input("You : ")

    if query == "0":
        break

    docs = retriever.invoke(query)
    
    print(docs)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({"context": context, "question": query})

    response = model.invoke(final_prompt)

    print(f"\n\n AI: {response.content}")


#COMPLETE PROJECT