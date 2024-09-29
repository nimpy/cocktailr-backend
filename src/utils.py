import os
from typing import List
from difflib import get_close_matches
from dotenv import load_dotenv

load_dotenv('.env')

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from langchain.memory import ConversationBufferMemory
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import (
#     ChatGoogleGenerativeAI,
#     HarmBlockThreshold,
#     HarmCategory,
# )
# from langchain.agents import AgentExecutor
# from langchain import hub
# from langchain.agents.format_scratchpad import format_log_to_str
# from langchain.agents.output_parsers import ReActSingleInputOutputParser
# from langchain.tools.render import render_text_description
from langchain.agents import tool
from sklearn.metrics.pairwise import cosine_similarity

from cocktails import COCKTAILS, get_cocktail_by_name, get_cocktail_embedding, compute_similarity, ingredient_match
from cocktails import INGREDIENTS, INGREDIENT_EMBEDDINGS



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
def get_alternatives_for_ingredient(ingredient: str, num_alternatives: int = 5) -> str:
    """Given an ingredient, returns a human-readable string of alternatives for that ingredient."""
    
    # First, try to find an exact match (case-insensitive)
    ingredient = ingredient.lower()
    exact_match = next((ing for ing in INGREDIENTS if ing.lower() == ingredient), None)
    
    if not exact_match:
        # If no exact match, try to find a close match
        close_matches = get_close_matches(ingredient, INGREDIENTS, n=1, cutoff=0.6)
        if not close_matches:
            return f"I couldn't find any ingredient matching '{ingredient}' in our database."
        exact_match = close_matches[0]
    
    # Get the embedding for the matched ingredient
    base_embedding = INGREDIENT_EMBEDDINGS.get(exact_match)
    if not base_embedding:
        return f"I found '{exact_match}' in our database, but I don't have enough information to suggest alternatives."
    
    # Calculate similarities with all other ingredients
    similarities = []
    for ing, emb in INGREDIENT_EMBEDDINGS.items():
        if ing != exact_match:
            similarity = cosine_similarity([base_embedding], [emb])[0][0]
            similarities.append((ing, similarity))
    
    # Sort by similarity (descending) and get top alternatives
    alternatives = sorted(similarities, key=lambda x: x[1], reverse=True)[:num_alternatives]
    
    # Transform alternatives into a human-readable string
    if alternatives:
        response = f"For '{exact_match}', you could try these alternatives:\n"
        for i, (alt, sim) in enumerate(alternatives, 1):
            response += f"{i}. {alt.capitalize()}\n"
        response += "\nThese ingredients have similar properties or uses in cocktails."
    else:
        response = f"I couldn't find any suitable alternatives for '{exact_match}'."
    
    return response



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
    model = ChatOpenAI(model="gpt-4")
    tools = [get_most_similar_cocktail, get_list_of_ingredients_for_a_cocktail, get_list_of_ingredients_and_recipe_for_a_cocktail, get_cocktail_given_a_list_of_ingredients, get_alternatives_for_ingredient]
    agent_executor = create_react_agent(model, tools)
    return agent_executor


def invoke_agent(agent_executor, messages) -> str:
    agent_invocation = agent_executor.invoke(messages)
    print(agent_invocation)
    return agent_invocation["messages"][-1].content


def main():
    question = "What can I make if I only have vodka, whisky, gin, simple, and coca cola? I also have lemons, oranges and blackberries?"
    agent = set_up_agent()
    agent.invoke({"input": question})["output"]


if __name__ == "__main__":
    main()