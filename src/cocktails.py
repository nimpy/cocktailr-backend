import os
import json
import re
from pprint import pprint
from difflib import get_close_matches
from typing import List, Dict
import re
from nltk.stem import WordNetLemmatizer
import nltk
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure Google GenerativeAI
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def get_embedding(content: str) -> List[float]:
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=content,
        task_type="retrieval_document",
        title="Embedding"
    )
    return result['embedding']


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


def save_embeddings(cocktail_id: int, embeddings: Dict[str, List[float]]):
    directory = 'embeddings'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for key, embedding in embeddings.items():
        filename = f"{directory}/{cocktail_id}_{key}.npy"
        np.save(filename, embedding)

def load_embeddings(cocktail_id: int) -> Dict[str, List[float]]:
    directory = 'embeddings'
    embeddings = {}
    for key in ['name', 'ingredients_list', 'ingredients_average']:
        filename = f"{directory}/{cocktail_id}_{key}.npy"
        if os.path.exists(filename):
            embeddings[key] = np.load(filename).tolist()
    return embeddings


def process_cocktails(cocktails: List[Dict]) -> List[Dict]:
    processed_cocktails = []
    for i, cocktail in enumerate(cocktails):
        cocktail_id = cocktail.get('id', None)
        if cocktail_id is None:
            print(f"Skipping cocktail without ID: {cocktail.get('name', 'Unknown')}")
            continue

        # Check if embeddings already exist
        existing_embeddings = load_embeddings(cocktail_id)
        if len(existing_embeddings) < 3:
            print(f"Generating embeddings for cocktail ID {cocktail_id}")
            # Process ingredients
            ingredients = trim_strings(cocktail.get('ingredients', []))
            ingredients_list = ', '.join([ing['name'] for ing in ingredients])
            
            # Generate embeddings
            embedding_name = get_embedding(cocktail.get('name', '').strip())
            embedding_ingredients_list = get_embedding(ingredients_list)
            embedding_ingredients_separate = [get_embedding(ing['name']) for ing in ingredients]
            embedding_ingredients_average = np.mean(embedding_ingredients_separate, axis=0).tolist()

            # Save embeddings
            save_embeddings(cocktail_id, {
                'name': embedding_name,
                'ingredients_list': embedding_ingredients_list,
                'ingredients_average': embedding_ingredients_average
            })

        processed_cocktail = {
            'id': cocktail_id,
            'name': cocktail.get('name', '').strip(),
            'ingredients': trim_strings(cocktail.get('ingredients', [])),
            'recipe': cocktail.get('recipe', '').strip(),
            'garnish': cocktail.get('garnish', '').strip(),
            'glass': cocktail.get('glass', '').strip(),
            'history': cocktail.get('history', '').strip(),
        }
        processed_cocktails.append(processed_cocktail)
    
        # if i > 157:
        #     break

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


# Constant to determine which embedding to use for similarity comparison
# Options: 'name', 'ingredients_list', 'ingredients_average'
SIMILARITY_EMBEDDING = 'ingredients_average'



def get_cocktail_by_name(name: str) -> Dict:
    # First, try an exact match (case-insensitive)
    for cocktail in COCKTAILS:
        if cocktail['name'].lower() == name.lower():
            return cocktail
    
    # If no exact match, try to find a close match
    cocktail_names = [cocktail['name'] for cocktail in COCKTAILS]
    close_matches = get_close_matches(name, cocktail_names, n=1, cutoff=0.6)
    
    if close_matches:
        close_match = close_matches[0]
        for cocktail in COCKTAILS:
            if cocktail['name'] == close_match:
                return cocktail
    
    return None


def get_cocktail_embedding(cocktail_id: int) -> List[float]:
    embeddings = load_embeddings(cocktail_id)
    return embeddings.get(SIMILARITY_EMBEDDING, [])

def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    return cosine_similarity([embedding1], [embedding2])[0][0]



INGREDIENTS = []

def load_ingredients():
    global INGREDIENTS
    all_ingredients = set()

    for cocktail in COCKTAILS:
        for ingredient in cocktail.get('ingredients', []):
            ingredient_name = clean_ingredient_name(ingredient['name'])
            all_ingredients.add(ingredient_name)

    INGREDIENTS = sorted(list(all_ingredients))

def clean_ingredient_name(name: str) -> str:
    # Convert to lowercase
    name = name.lower()
    
    # Remove any text in parentheses
    name = re.sub(r'\([^)]*\)', '', name)
    
    # Remove common prefixes/suffixes and extra whitespace
    name = re.sub(r'^(fresh |dried |whole |chopped |sliced |diced |crushed |ground |powdered |grated )', '', name)
    name = re.sub(r'( juice| peel| zest| twist| slice| wedge| sprig| leaf| leaves)$', '', name)
    
    # Remove any non-alphanumeric characters (except spaces)
    name = re.sub(r'[^a-z0-9\s]', '', name)
    
    # Remove extra whitespace and strip
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

# Call this function after COCKTAILS is populated
load_ingredients()


if __name__ == '__main__':
    file_path = 'cocktails.json'
    initialize_cocktails(file_path)
    print(len(COCKTAILS), "cocktails loaded.")

    pprint(COCKTAILS[0], width=100)

    # Load and print embeddings for the first cocktail as an example
    first_cocktail_id = COCKTAILS[0]['id']
    embeddings = load_embeddings(first_cocktail_id)
    print(f"\nEmbeddings for cocktail ID {first_cocktail_id}:")
    for key, value in embeddings.items():
        print(f"{key}: {value[:5]}...")

    print("\nTesting get_cocktail_by_name:")
    print(get_cocktail_by_name("38 special"))


    print(f"Total unique ingredients: {len(INGREDIENTS)}")
    for ingredient in INGREDIENTS:
        print(f"{ingredient},", end=' ')