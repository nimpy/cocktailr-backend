from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from warnings import filterwarnings

from utils import set_up_agent, invoke_agent
from misc_utils import dict_to_conversation, dict_to_messages
# from gemini_utils import call_gemini, let_gemini_rephrase


filterwarnings("ignore", category=UserWarning)
router = APIRouter()

agent = set_up_agent()



@router.post("/send-message")
async def send_message(request: Request) -> str:

    body_dict = await request.json()

    new_message = body_dict.get("newMessage", "")
    print(new_message)

    history = body_dict.get("history", [])
    print(history)

    messages = dict_to_messages(body_dict)

    response = invoke_agent(agent, messages)
    print(response)

    response_dict = {"message": response}

    # if response.startswith("I'm sorry, "):
    #     print("Calling Gemini!")
    #     gemini_response = await call_gemini(conversation)
    #     if gemini_response is not None and gemini_response != "":
    #         response_dict = {"message": gemini_response}
    #     else:
    #         response_dict = {"message": response}
    # else:
    #     print("Letting Gemini rephrase!")
    #     snarky_response = await let_gemini_rephrase(conversation, response)
    #     response_dict = {"message": snarky_response}

    # print(response_dict)

    return JSONResponse(content=response_dict)


@router.get("/health")
async def health(request: Request) -> str:

    return "Healthy"


