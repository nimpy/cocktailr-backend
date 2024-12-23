import json
import os
from openai import OpenAI
from difflib import get_close_matches
from dotenv import load_dotenv

load_dotenv('.env')

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.agents import tool
from sklearn.metrics.pairwise import cosine_similarity

from cocktails import COCKTAILS, get_cocktail_by_name, get_cocktail_embedding, compute_similarity, ingredient_match
from cocktails import INGREDIENTS, INGREDIENT_EMBEDDINGS



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    print("AGENT INVOCATION:", agent_invocation)
    return agent_invocation["messages"][-1].content


def openai_call_wrapper_return_json(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=messages,
        temperature=0.0,
    )
    try:
        response_dict = json.loads(response.choices[0].message.content)
    except json.decoder.JSONDecodeError as e:
        print("There was an error with parsing the json response in openai_call_wrapper_return_json(). The response was:", response.choices[0].message.content)
        raise e
    return response_dict


def sassify_last_response(conversation_dict):
    prompt = f"""You have the following conversation between a user and an agent:
{json.dumps(conversation_dict, indent=2)}
Modify the last agent message to make it snarky and sassy. The agent should tease the user in a flirty way, but the message should retain the information from the original message. Output the modified message in json format like:
{{"agent": "Honey, with that inventory, a Gimlet is about all you can make.  Unless you consider a Screwdriver with a splash of sadness a viable option.  🍸 😉"}}
"""
    
    messages = [{"role": "user", "content": prompt}]
    response = openai_call_wrapper_return_json(messages)
    
    return response.get('agent', '')


def openai_vision_call_wrapper(message_text, b64_image):
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": message_text},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_image}",
                "detail": "high"
            },
            },
        ]
        }
    ]
    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
        )
    except Exception as e:
        print("There was an error with the response in openai_vision_call_wrapper(). The response was:", response)
        raise e

    return response.choices[0].message.content


def get_sassy_image_response(message_text, b64_image):
    prompt = f"""You have the following message from the user about cocktails:
{message_text}
The user has also uploaded an image. 

You should respond to the user's message in a sassy way, while also providing a relevant response to the image. You should be flirty and playful, for example:
Honey, with that inventory, a Gimlet is about all you can make.  Unless you consider a Screwdriver with a splash of sadness a viable option.  🍸 😉
"""
    return openai_vision_call_wrapper(prompt, b64_image)


# # TODO
# def get_list_of_ingredients_from_image(b64_image):
#     ingredients_list = """"
# absinthe, absinthe club soda, agave syrup, aged agricole, aged overproof rum, aged pot still rum, aged rhum agricole, aged rum, aguardiente, allspice dram, allspice liqueur, amaretto, amargo vallet, amaro ciociaro, amaro dellerborista, amaro meletti, amaro montenegro, amaro nardini, amaro nonino, amaro sfumato rabarbaro, amer picon, amontadillo sherry, amontillado sherry, ancho reyes, anejo tequila, ango optional, angostura, angostura amaro, angostura bitters, angostura bitters floated, angostura peychauds, angostura peychauds bitters, angostura peychauds orange bitters, angosutra orange bitters, aperol, apple cider, applejack, appletons, apricot brandy, apricot liqueur, aquavit, averna, averna amaro, back strap rum, banana liqueur, benedictine, berries, biscotti liqueur, black strap rum, blackberries, blackstrap rum, blanc vermouth, blanco tequila, bottle of underberg or 025 oz amargo vallet, bourbon, braulio amaro, brown sugar cube, bruised cucumber, bruto americano, bual madeira, byrrh grand quinquina, cacao, cachaa, cachaca, caf lolita, cafe lolita, calvados, campari, cane syrup, cappelletti apertivo rosso, cassis, champagne, cherry, cherry heering, cherry heering club soda, cherry herring, chocolate bitters, cholula, cholula hot sauce, club soda, cocchi americano, cocchi americano or lillet blanc, cocchi rosa, coco lopez, coconut cream, coconut syrup, coconut water, coffee liqueur, cognac, cointreau, cold brew coffee, cream, creme de cacao, creme de menthe, creme de violette, crme de cacao, crme de cassis, crme de menthe, crme de violetteyvette, cruzan black strap rum, cucumber, cucumber strawberry, curaao, curacao, cynar, dark rum, dash absinthe, dash absinthe cucumber slices club soda, dash absinthe dash angostura bitters, dash angostura bitters club soda, dash angosutra bitters club soda, dash cane syrup club soda, dash cream, demerara rum, demerara syrup, drambuie, dry red wine, dry vermouth, egg, egg white, egg white club soda, egg yolk, elderflower liqueur, falernum, fernet, fernet branca, fernet branca menta, fernet vallet, fino sherry, galliano, gary classico, genever, gin, gin or cognac, ginger, ginger syrup, gold rhum agricole, goslings, goslings rum, gran classico, granny smith apple, grapefruit, grapes, green chartreuse, green chatreuse, grenadine, habanero bitters, half lemon cut into 4 pieces, hand whipped cream float, hand whipped cream float sweetened with coconut syrup, highland scotch, honey, honey syrup, irish, irish whiskey, islay scotch, jamaican rum, japanese whiskey, japanese whisky, juice, lemon, lemon lime orange, lemonlime, lemonlime juice, liberal amount of angosutra bitters club soda, licor 43, light rum, lillet rouge, lime, lime chunks, lime disc, lime wedge, lime wedge muddled, lofi amaro, lofi gentian amaro, lolita, luxardo bitters, manzanilla sherry, maple, maraschino, maraschino liqueur, marschino, mezcal, mint, mint cucumber, mint strawberry, navy strength gin, negra modelo, old tom gin, olorosso sherry, orange, orange and angostura bitters, orange bitters, orange flower, orange flower water, orange slices, orgeat, overproof jamaican rum, passionfruit syrup, peach bitters, peach liquer, peach liqueur, pear, pear brandy, pedro ximenez sherry, pernod, peychaud, peychauds, peychauds bitters, peychauds bitters club soda, pimms, pinch salt, pineapple, pineapple or orange, pineapple rum, pisco, port, pot still black rum, pot still rum, prosecco, punch fantasia, punt e mes, raspberries, raspberry preserves, red wine, reposado tequila, rhum agricole, rose water, rosewater, ruby port, rum, rye, salt, scotch, sfumato amaro, simple, simple syrup, smith cross, spanish brandy, sprite, strawberries, strawberry, strawberry cherry, sugar, suze, sweet vermouth, sweetened ginger, tawny port, tequila, tomrs tonic syrup, velvet falernum, vodka, watermelon, whiskey, white grapefruit, white rum, white sugar cube, white sugar cubes, worcestershire, yellow chartreuse
# """
#     prompt = f"""
# List all the different cocktail ingredients that you see in this image and return it as a comma separated list, e.g. gin, tonic, lime, ice.
# Return only the list.
# Possible ingredients include: {ingredients_list}
# """
#     response = openai_vision_call_wrapper(prompt, b64_image)
#     cocktails = get_cocktail_given_a_list_of_ingredients(response)
    
#     return cocktails


def main():
    question = "What can I make if I only have vodka, whisky, gin, simple, and coca cola? I also have lemons, oranges and blackberries?"
    agent = set_up_agent()
    agent.invoke({"input": question})["output"]


if __name__ == "__main__":
    main()