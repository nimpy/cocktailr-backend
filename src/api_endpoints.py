from typing import List, Dict, Optional
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from warnings import filterwarnings

from utils import set_up_agent, invoke_agent, sassify_last_response, get_sassy_image_response
from misc_utils import dict_to_messages
from cocktails import COCKTAILS


filterwarnings("ignore", category=UserWarning)
router = APIRouter()

agent = set_up_agent()



@router.post("/send-message")
async def send_message(request: Request) -> str:

    body_dict = await request.json()

    new_message = body_dict.get("newMessage", "")
    print("USER:", new_message)

    messages = dict_to_messages(body_dict)
 
    response = invoke_agent(agent, messages)
    print("AGENT:", response)

    conversation_dict = {
        "user": new_message,
        "agent": response,
    }

    sassy_response = sassify_last_response(conversation_dict)
    print("SASSY AGENT:", sassy_response)

    response_dict = {"message": sassy_response}

    return JSONResponse(content=response_dict)


@router.get("/health")
async def health(request: Request) -> str:

    return "Not great, if you're drinking cocktails all the time."


@router.get("/cocktails")
async def get_all_cocktails() -> List[Dict]:
    return COCKTAILS


@router.get("/cocktails/{id}")
async def get_cocktail_by_id(id: int) -> Dict:
    for cocktail in COCKTAILS:
        if cocktail['id'] == id:
            return cocktail
    raise HTTPException(status_code=404, detail="Cocktail not found")


class ImageUpload(BaseModel):
    newMessage: Optional[str] = None
    image: str


@router.post("/send-image")
async def send_image(upload: ImageUpload):

    if upload.newMessage:
        response = get_sassy_image_response(upload.newMessage, upload.image)
    else:
        response = get_sassy_image_response("I would like to make a cocktail. What could I make with this?", upload.image)

    return {"message": response}
