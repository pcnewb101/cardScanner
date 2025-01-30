## retrieving all card names takes about 10 minutes and about 18500 entries

import json
from pokemontcgsdk import RestClient, Card

# Set API key
RestClient.configure("b97ddab5-5569-4912-91b7-a3a93776f78d")

# Fetch all cards only if not already saved
try:
    with open("fetched_cards.json", "r") as f:
        all_card_names = json.load(f)
    print("Loaded cached card names from file.")
except FileNotFoundError:
    print("Fetching all card data...")
    cards = Card.all()
    all_card_names = [card.name for card in cards]

    # Save to file for future use
    with open("fetched_cards.json", "w") as f:
        json.dump(all_card_names, f)
