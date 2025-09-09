import os

from dotenv import load_dotenv
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.cache import BaseCache
from langchain_core.callbacks import Callbacks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import SimpleSequentialChain
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

import time

# Try to rebuild the ChatOpenAI model after importing required dependencies
try:
    ChatOpenAI.model_rebuild()
except Exception as e:
    print(f"Warning: Could not rebuild ChatOpenAI model: {e}")

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "site_content_db")


# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Step 1: Retrieval Chain
def retrieve_documents_metadata(query, retrieval_num, metadata):
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": retrieval_num,
            "filter": metadata
        },
    )

    # Retrieve the documents
    relevant_docs = retriever.invoke(query)

    # for i, doc in enumerate():

    return relevant_docs

# Step 1: Retrieval Chain
def retrieve_documents(query, day):
    sites = retrieve_documents_metadata(query, day * 5, {"type": "site"})
    food = retrieve_documents_metadata(query, day * 3, {"type": "food"})

    relevant_docs = sites + food

    return relevant_docs

# Step 2: Combine retrieved content with the original query
def combine_with_retrieval(query, retrieved_docs):
    combined_input = (
        "以下是一些可能有助於回答該問題的文件： "
        + "\n\n相關文件：\n"
        + "\n\n".join([doc.page_content for doc in retrieved_docs])
        + query
    )
    return combined_input

# Create a ChatOpenAI model
def query_llm(combined_input):
    # model = ChatOpenAI(model="gpt-4o-mini")
    model = ChatOpenAI(model="gpt-4o-mini")  # Changed from o1-mini to gpt-4o-mini
    # model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
    messages = [
        SystemMessage(content="你是一個嘉義在地導遊。"),
        HumanMessage(content=combined_input),
    ]
    result = model.invoke(messages)
    return result.content



def main(day: int, preference:str):
    # Define the user's question
    # days = int(input("預計旅遊天數(請輸入數字): "))
    # preference = input("請輸入旅遊偏好: ")
    #template = "我的偏好是 {preference}。請以列表方式，幫我景點推薦"
    template = preference
    prompt_template = ChatPromptTemplate.from_template(template)

# print("-----Prompt from Template-----")
    prompt = prompt_template.invoke({"preference": "preference"})
# print(prompt)
    prompt_content = prompt.messages[0].content  # Assuming prompt contains HumanMessage

# Now create the sequential chain with chain objects
    retrieval_chain = RunnableLambda(retrieve_documents) | RunnableLambda(combine_with_retrieval) | RunnableLambda(query_llm) | RunnableLambda(retrieve_documents)

# Run the chain with the initial query
    start = time.time()
    retrieved_docs = retrieve_documents(prompt_content, day)  # Step 1
    end = time.time()
# print('Time for step1: ', end-start)
    start = time.time()
    combined_query = combine_with_retrieval(prompt_content, retrieved_docs)  # Step 2
    end = time.time()
# print('Time for step2: ', end-start)
    start = time.time()
    llm_response = query_llm(combined_query)  # Step 3
    end = time.time()
# print('Time for step3: ', end-start)
    start = time.time()
    retrieved_docs = retrieve_documents(llm_response, day)  # Step 4
    end = time.time()
# print('Time for step4: ', end-start)

    # for i, doc in enumerate(retrieved_docs, 1):
        # print(f"Document {i}:\n{doc.page_content}\n")
        # print(f"Name: \n{doc.metadata['key']}\n")

    # print([ doc.metadata['key'] for doc in retrieved_docs ])
    return [ doc.metadata['key'] for doc in retrieved_docs ]

if __name__ == "__main__":
    main(2, "親子旅遊")
