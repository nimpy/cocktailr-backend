import json
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import logging
import sys
from icecream import ic
# from utils import openai_call_wrapper

# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma

# from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA

router = APIRouter()


# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# llm = ChatOpenAI(model="gpt-4")

# prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")

# document_chain = create_stuff_documents_chain(llm, prompt)



# @router.post("/send-message")
# async def send_message(request: Request) -> str:

#     body_dict = await request.json()

#     new_message = body_dict.get("newMessage", "")

#     history = body_dict.get("history", [])

#     history.append({"sender": "user", "text": new_message})
#     ic(history)

#     history_string = json.dumps(history)

#     docs = retriever.invoke(history_string)
#     docs = list({doc.metadata["link"]: doc for doc in docs}.values()) # remove duplicates


#     qa_chain = RetrievalQA.from_chain_type(
#         llm,
#         retriever=vectorstore.as_retriever()
#     )

    
#     result = qa_chain({"query": history_string, "context": docs})


#     response_dict = {"message": result["result"], "sources": [doc.metadata["link"] for doc in docs]}
#     print(response_dict)
#     return JSONResponse(content=response_dict)


@router.post("/send-message")
async def send_message(request: Request) -> str:

    body_dict = await request.json()

    new_message = body_dict.get("newMessage", "")
    print(new_message)

    history = body_dict.get("history", [])
    print(history)

    history.append({"sender": "user", "text": new_message})
    ic(history)

    history_string = json.dumps(history)

    # docs = retriever.invoke(history_string)
    # docs = list({doc.metadata["link"]: doc for doc in docs}.values()) # remove duplicates


    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=vectorstore.as_retriever()
    # )
    
    # result = qa_chain({"query": history_string, "context": docs})

    response_dict = {"message": "Lorem ipsum"}
    print(response_dict)
    return JSONResponse(content=response_dict)


@router.get("/health")
async def health(request: Request) -> str:

    return "Healthy"


