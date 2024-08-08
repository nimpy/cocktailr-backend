import json
from pprint import pprint
from difflib import get_close_matches
from typing import List, Dict
import re
from nltk.stem import WordNetLemmatizer
import nltk


def read_cocktail_data(file_path):
    with open(file_path, 'r') as file:
        cocktails = json.load(file)
    return cocktails


def trim_strings(obj):
    if isinstance(obj, str):
        return obj.strip()
    elif isinstance(obj, dict):
        return {k: trim_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [trim_strings(item) for item in obj]
    else:
        return obj


def process_cocktails(cocktails):
    processed_cocktails = []
    for cocktail in cocktails:
        processed_cocktail = {
            'id': cocktail.get('id', None),
            'name': cocktail.get('name', '').strip(),
            'ingredients': trim_strings(cocktail.get('ingredients', [])),
            'recipe': cocktail.get('recipe', '').strip(),
            'garnish': cocktail.get('garnish', '').strip(),
            'glass': cocktail.get('glass', '').strip(),
            'history': cocktail.get('history', '').strip()
        }
        processed_cocktails.append(processed_cocktail)
    return processed_cocktails


def initialize_cocktails(file_path: str):
    global COCKTAILS
    raw_cocktails = read_cocktail_data(file_path)
    COCKTAILS = process_cocktails(raw_cocktails)


COCKTAILS = []
initialize_cocktails(file_path='cocktails.json')

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()


def normalize_ingredient(ingredient: str) -> List[str]:
    # Convert to lowercase and remove any non-alphanumeric characters
    ingredient = re.sub(r'[^a-zA-Z\s]', ' ', ingredient.lower())
    
    # Split compound ingredients
    words = ingredient.split()
    
    # Lemmatize each word (convert to singular form)
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    
    # Return both the original split words and the lemmatized version
    return list(set(words + lemmatized))  # Use set to remove duplicates

def ingredient_match(available_ingredients: List[str], cocktail_ingredients: List[str]) -> bool:
    normalized_available = set(word for ing in available_ingredients for word in normalize_ingredient(ing))
    normalized_cocktail = set(word for ing in cocktail_ingredients for word in normalize_ingredient(ing))
    
    # Check if all cocktail ingredients are in the available ingredients
    return normalized_cocktail.issubset(normalized_available)


if __name__ == '__main__':
    file_path = 'cocktails.json'

    cocktails = read_cocktail_data(file_path)
    processed_cocktails = process_cocktails(cocktails)

    pprint(processed_cocktails[0], width=100)
