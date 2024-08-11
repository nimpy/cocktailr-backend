from utils import get_most_similar_cocktail, get_alternatives_for_ingredient

if __name__ == "__main__":
    print(get_most_similar_cocktail("Moscow Mule"))

    test_ingredient = "lemon juice"
    result = get_alternatives_for_ingredient(test_ingredient)
    print(result)