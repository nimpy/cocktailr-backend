import os
from typing import List

from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain.agents import AgentExecutor
from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description


from langchain.agents import tool


@tool
def get_most_similar_cocktail(cocktail: str) -> str:
    """Returns the most similar cocktail to the input cocktail."""
    return "Negroni"


@tool
def get_list_of_ingredients_for_a_cocktail(cocktail_name: str) -> str:
    """Returns the list of ingredients for a cocktail."""
    return "ice, ice, baby"


@tool
def get_list_of_ingredients_and_recipe_for_a_cocktail(cocktail_name: str) -> str:
    """Given a cocktail name, returns the list of ingredients for that cocktail and the recipe how to prepare it."""
    return "Ingredients: ice, ice, baby. Recipe: Mix it all together."


@tool
def get_alternatives_for_ingredient(ingredient: str) -> str:
    """Given an ingredient, returns a list of alternatives for that ingredient."""
    return "Vodka, Rum"

@tool
def get_cocktail_given_a_list_of_ingredients(ingredients: str) -> str:
    """Given ingredients, returns one cocktail that can be made from the input ingredients."""
    return "Negroni"




def set_up_agent():

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        convert_system_message_to_human=True,
        handle_parsing_errors=True,
        temperature=0.0,
        max_tokens=None,
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
    )

    tools = [get_most_similar_cocktail, get_list_of_ingredients_for_a_cocktail, get_list_of_ingredients_and_recipe_for_a_cocktail, get_alternatives_for_ingredient, get_cocktail_given_a_list_of_ingredients]

    agent_prompt = hub.pull("mikechan/gemini")

    prompt = agent_prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    llm_with_stop = llm.bind(stop=["\nObservation"])

    memory = ConversationBufferMemory(memory_key="chat_history")

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_stop
        | ReActSingleInputOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax",)
    return agent_executor



def invoke_agent(agent_executor, input_text: str) -> str:
    return agent_executor.invoke({"input": input_text})["output"]


def main():
    question = "Which cocktail can I make, given that I only have vodka and water in my fridge?"
    agent = set_up_agent()
    agent.invoke({"input": question})["output"]


if __name__ == "__main__":
    main()