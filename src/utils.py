import os
from typing import List

from dotenv import load_dotenv

load_dotenv('.env')

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

from cocktails import COCKTAILS, get_cocktail_by_name, get_cocktail_embedding, compute_similarity, ingredient_match


@tool
def get_most_similar_cocktail(cocktail_name: str) -> str:
    """Returns the most similar cocktail to the input cocktail."""
    input_cocktail = get_cocktail_by_name(cocktail_name)
    if not input_cocktail:
        return f"Cocktail '{cocktail_name}' not found."
    print(f"Fetched cocktail: {input_cocktail.get('name')}")

    input_embedding = get_cocktail_embedding(input_cocktail['id'])
    if not input_embedding:
        return f"Embedding for '{cocktail_name}' not found."

    max_similarity = -1
    most_similar_cocktail = None

    for cocktail in COCKTAILS:
        if cocktail['id'] == input_cocktail['id']:
            continue  # Skip the input cocktail itself

        current_embedding = get_cocktail_embedding(cocktail['id'])
        if not current_embedding:
            continue  # Skip cocktails without embeddings

        similarity = compute_similarity(input_embedding, current_embedding)

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_cocktail = cocktail

    if most_similar_cocktail:
        return f"The most similar cocktail to '{cocktail_name}' is '{most_similar_cocktail['name']}' with a similarity of {max_similarity:.4f}"
    else:
        return f"No similar cocktails found for '{cocktail_name}'."



@tool
def get_list_of_ingredients_for_a_cocktail(cocktail_name: str) -> str:
    """Returns the list of ingredients for a cocktail."""
    for cocktail in COCKTAILS:
        if cocktail['name'].lower() == cocktail_name.lower():
            ingredients = [f"{ing.get('quantity', '')} {ing.get('unit', '')} {ing['name']}".strip() 
                           for ing in cocktail['ingredients']]
            return "\n".join(ingredients)
    return ""



@tool
def get_list_of_ingredients_and_recipe_for_a_cocktail(cocktail_name: str) -> str:
    """Given a cocktail name, returns the list of ingredients for that cocktail and the recipe how to prepare it."""
    for cocktail in COCKTAILS:
        if cocktail['name'].lower() == cocktail_name.lower():
            ingredients = [f"{ing.get('quantity', '')} {ing.get('unit', '')} {ing['name']}".strip() 
                           for ing in cocktail['ingredients']]
            return f"Ingredients:\n{', '.join(ingredients)}\n\nRecipe:\n{cocktail['recipe']}"
    return ""


@tool
def get_alternatives_for_ingredient(ingredient: str) -> str:
    """Given an ingredient, returns a list of alternatives for that ingredient."""
    return "Vodka, Rum"

@tool
def get_cocktail_given_a_list_of_ingredients(ingredients: str) -> str:
    """Given ingredients, returns one cocktail that can be made from the input ingredients."""
    available_ingredients = [ing.strip() for ing in ingredients.split(',')]
    matching_cocktails = []
    
    for cocktail in COCKTAILS:
        cocktail_ingredients = [ing['name'] for ing in cocktail['ingredients']]
        if ingredient_match(available_ingredients, cocktail_ingredients):
            matching_cocktails.append(cocktail['name'])
        if "bramble" in cocktail['name'].lower():
            print(cocktail_ingredients)
            print(available_ingredients)
    
    if matching_cocktails:
        return f"Matching cocktails: {', '.join(matching_cocktails)}"
    else:
        return "No cocktails found with those ingredients."




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
    question = "What can I make if I only have vodka, whisky, gin, simple, and coca cola? I also have lemons, oranges and blackberries?"
    agent = set_up_agent()
    agent.invoke({"input": question})["output"]


if __name__ == "__main__":
    main()