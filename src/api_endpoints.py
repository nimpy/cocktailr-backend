import json
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import logging

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)

from utils import set_up_agent, invoke_agent
from misc_utils import dict_to_conversation
from gemini_utils import call_gemini

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


@router.post("/get-most-similar-cocktail")
async def get_most_similar_cocktail(request: Request) -> str:

    body_dict = await request.json()

    cocktail = body_dict.get("cocktail", "")

    most_similar_cocktail = "Bramble"

    response_dict = {"mostSimilarCocktail": most_similar_cocktail}
    print(response_dict)
    return JSONResponse(content=response_dict)






@router.post("/send-message")
async def send_message(request: Request) -> str:

    body_dict = await request.json()

    new_message = body_dict.get("newMessage", "")
    print(new_message)

    history = body_dict.get("history", [])
    print(history)

    conversation = dict_to_conversation({"newMessage": new_message, "history": history})
    print(conversation)

    response = invoke_agent(agent, conversation)
    print(response)

    gemini_response = None
    if response.startswith("I'm sorry, "):
        print("Calling Gemini!")
        gemini_response = await call_gemini(conversation)
    

    if gemini_response is not None and gemini_response != "":
        response_dict = {"message": gemini_response}
    else:
        response_dict = {"message": response}
    print(response_dict)

    return JSONResponse(content=response_dict)


@router.get("/health")
async def health(request: Request) -> str:

    return "Healthy"


