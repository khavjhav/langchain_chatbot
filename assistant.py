from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveJsonSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient, models
from langchain_core.documents import Document
import random
from customllm import CustomLLM
import json
from pathlib import Path
from pprint import pprint
from threading import Thread
import threading
import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")

# model = Ollama(model="dolphin-mixtral")
# model= Ollama(model="tinydolphin")
model = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest")
# model=CustomLLM()
ollama_emb = OllamaEmbeddings(
    model="nomic-embed-text",
)
# ollama_emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def createChain():


    template = """
    <s>[INST]
    You are a customer support helpful assistant for an agency that helps student to pursue their academics in abroad based on the context given.
    Your context will contain a list of documents each refering to a postgraduate program.
    [/INST]
    [INST]
    Answer the question based only on the following context. Information that you need is included in the context:
    {context}
    [/INST]
    [INST]
    You are a customer support helpful assistant for an agency that helps student to pursue their academics in abroad based on the context given.
    Your context will contain a list of documents each refering to a relevant postgraduate program retrieved from a vector database based on similarity search.
    You have to answer the following question like an helpful assistant. Do not mention about how you are getting the context. Behave like a human and you already know the context.
    This context is provided by the system instead of customer so do not talk to customer about how you are getting information from the context.
    Rather use the context to answer their query.
    [/INST]
    [INST]
    Question: {question}
    [/INST]</s>
    Your Answer:

    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()


    chain = (
        {"question": RunnablePassthrough(),"context": RunnablePassthrough()}
        | prompt
        | model
        | output_parser
    )


    return {"chain": chain, "prompt": prompt}

# print(chain.invoke({"topic": "ice cream", "name": "Bob"}))

def generator(question):

    chain = createChain()
    context=getContext(question)
    print(chain["prompt"].format_messages(context= context, question= question))
    # for chunk in chain["chain"].stream({"question": question,"context":context}):
    #     print(chunk, end="", flush=True)
    #     yield chunk
    chunk=chain["chain"].invoke({"question": question,"context":context})
    print(chunk)
    return chunk

def process_item(item, index):
    docs = Document(page_content=str(item))
    db = Qdrant.from_documents([docs], ollama_emb, url="http://localhost:6333", collection_name="my_documents_google")
    # Corrected way to get the current thread's name
    print(f"Processed item {index} in thread {threading.current_thread().name}")

def embedJson():
    print("Processing items in parallel...")
    file_path = './oxford.json'
    json_data = json.loads(Path(file_path).read_text())

    threads = []

    for i, item in enumerate(json_data["Programmes"]):
        thread = Thread(target=process_item, args=(item, i))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All items processed.")
    return {"message": "done"}


def getContext(query):
    client = QdrantClient("http://localhost:6333")
    db = Qdrant(client=client, collection_name="my_documents", embeddings=ollama_emb)

    # embedding_vector = ollama_emb.embed_query(query)
    # print(embedding_vector)
    # docs = db.max_marginal_relevance_search(query, k=4, fetch_k=1000)
    docs=db.similarity_search_with_score(query, k=10)


    return str(docs)
