import json
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import logging

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)

from utils import set_up_agent, invoke_agent

router = APIRouter()


agent = set_up_agent()

@router.post("/ask-agent")
async def ask_agent(request: Request) -> str:
    
    body_dict = await request.json()

    question = body_dict.get("question", "")
    print(question)

    response = invoke_agent(agent, question)
    print(response)

    response_dict = {"message": response}
    print(response_dict)
    return JSONResponse(content=response_dict)






@router.post("/send-message")
async def send_message(request: Request) -> str:

    body_dict = await request.json()

    new_message = body_dict.get("newMessage", "")
    print(new_message)

    history = body_dict.get("history", [])
    print(history)

    history.append({"sender": "user", "text": new_message})
    # ic(history)

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


