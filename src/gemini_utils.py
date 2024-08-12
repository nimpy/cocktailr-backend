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

    conversation += "\n Respond in a snarky and witty manner, while still providing useful information."

    # Generate content using Gemini
    response = model.generate_content(conversation)
    
    try:
        response_text = response.text
    except Exception:
        response_text = ""
    return response_text


async def let_gemini_rephrase(conversation: str, original_response: str) -> str:
    prompt = f"""
Given the following conversation and original response, rephrase the response to make it more snarky and witty, while maintaining the core information:

Conversation:
{conversation}

Original Response:
{original_response}

Please provide a snarky and witty version of the response:
"""

    gemini_response = await call_gemini(prompt)
    
    if gemini_response is not None and gemini_response != "":
        return gemini_response
    else:
        return original_response  # Fallback to the original response if Gemini fails