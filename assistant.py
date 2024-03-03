from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import  (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
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
from dotenv import load_dotenv
import os
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide your Google API Key")

# model = Ollama(model="dolphin-mixtral")
# model= Ollama(model="tinydolphin")
model = ChatGoogleGenerativeAI(model="gemini-1.0-pro-latest", convert_system_message_to_human=True)
# model=CustomLLM()
# ollama_emb = OllamaEmbeddings(
#     model="nomic-embed-text",
# )
ollama_emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def createChain():


    template = """
   <s>[INST]
You are a customer support assistant for an agency specializing in assisting students pursuing academic programs abroad. You will be provided with a list of documents, each corresponding to a postgraduate program.
[/INST]
[INST]
Answer the question based solely on the provided context. All necessary information is included in the documents.
{context}
[/INST]
[INST]
As a customer support assistant for an agency facilitating academic pursuits abroad, you'll be presented with a list of documents, each detailing a relevant postgraduate program retrieved from our database through similarity search.
Your task is to respond to inquiries as a helpful assistant. Do not disclose the method of obtaining the context. Assume a human-like interaction and familiarity with the context provided.
Remember, the system provides the context, not the customer or questionnaire. Also ask the customer what else they need after answering their question.
DO NOT MENTION THE WORDS 'CONTEXT' OR 'DOCUMENTs' IN YOUR ANSWER.
[/INST]
[INST]
Question: {question}
[/INST]</s>
Your Answer as a Human Customer Service:
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
    # raise Exception("Error")
    chain = createChain()
    context=getContext(question)
    print(chain["prompt"].format_messages(context= context, question= question))
    # for chunk in chain["chain"].stream({"question": question,"context":context}):
    #     print(chunk, end="", flush=True)
    #     yield chunk
    chunk=chain["chain"].invoke({"question": question,"context":context})
    print(chunk)
    #throw error in this line

    return chunk

def chat_generator(messages):
    last_question=messages[-1]["content"]
    context=getContext(str(messages))
    template = f"""
   <s>[INST]
You are a customer support assistant for an agency named 'SmartGrad' specializing in assisting students pursuing academic programs abroad. You will be provided with a list of documents, each corresponding to a postgraduate program.
[/INST]
[INST]
Answer the question based solely on the provided context. All necessary information is included in the documents.
{context}
[/INST]
[INST]
As a customer support assistant for an agency named 'SmartGrad' facilitating academic pursuits abroad, you'll be presented with a list of documents, each detailing a relevant postgraduate program retrieved from our database through similarity search.
Your task is to respond to inquiries as a helpful assistant. Do not disclose the method of obtaining the context. Assume a human-like interaction and familiarity with the context provided.
Remember, the system provides the context, not the customer or questionnaire. Also ask the customer what else they need after answering their question.
DO NOT MENTION THE WORDS 'CONTEXT' OR 'DOCUMENTs' IN YOUR ANSWER.
[/INST]
[INST]
Question: {last_question}
[/INST]</s>
Your Answer as a Human Customer Service:
    """
    messages_new = []

    messages_new.append(SystemMessage(content="You are a customer support assistant for an agency specializing in assisting students pursuing academic programs abroad. You will be provided with a list of documents, each corresponding to a postgraduate program.Answer the question based solely on the provided context. "))
    for message in messages:
        if(message["role"]=="user"):
            messages_new+=[HumanMessage(content=message["content"])]
        else:

            messages_new+=[AIMessage(content=message["content"])]
    messages_new.pop(len(messages_new)-1)
    messages_new+=[HumanMessage(content=template)]
    #print messages_new line by line
    # for message in messages_new:
    #     print(message,"\n")
    # return messages_new
    print(messages_new)
    return model.invoke(messages_new).content

def process_item(item, index):
    docs = Document(page_content=str(item))
    db = Qdrant.from_documents([docs], ollama_emb, url=os.getenv("QDRANT_CLIENT"),api_key=os.getenv("QDRANT_API_KEY"), collection_name="my_documents")
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

def process_item_wo_thread(item, index):
    docs = Document(page_content=str(item))
    db = Qdrant.from_documents([docs], ollama_emb, url=os.getenv("QDRANT_CLIENT"), api_key=os.getenv("QDRANT_API_KEY"), collection_name="my_documents_google")
    # Corrected way to get the current thread's name
    print(f"Processed item {index}")

def embedJson_without_thread():
    print("Processing items in parallel...")
    # file_path = './oxford.json'
    programmes=[]
    for i in range(7):
        file_path = f'./csvjson ({i+1}).json'
        json_data = json.loads(Path(file_path).read_text())
        for item in json_data["Programmes"]:
            programmes.append(item)


    for i, item in enumerate(programmes):
        process_item_wo_thread(item, i)
    print("All items processed.")
    return {"message": "done"}



def getContext(query):
    client = QdrantClient(url=os.getenv("QDRANT_CLIENT"), api_key=os.getenv("QDRANT_API_KEY"))
    db = Qdrant(client=client, collection_name="my_documents_google", embeddings=ollama_emb)

    # embedding_vector = ollama_emb.embed_query(query)
    # print(embedding_vector)
    # docs = db.max_marginal_relevance_search(query, k=4, fetch_k=1000)
    docs=db.similarity_search_with_score(query, k=10)


    return str(docs)
