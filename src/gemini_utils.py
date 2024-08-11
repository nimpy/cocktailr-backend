import google.generativeai as genai
import os
from typing import Dict

from misc_utils import dict_to_conversation

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-pro-latest')


async def call_gemini(conversation: str) -> str:
    """
    Wrapper function to call Gemini with the given input conversation.
    """
    # Generate content using Gemini
    response = model.generate_content(conversation)
    
    try:
        response_text = response.text
    except Exception:
        response_text = ""
    return response_text

