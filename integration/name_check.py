import os
import csv
from pokemontcgsdk import Card
from datetime import datetime
from processing.write_file import csv_file, write_name_file
from processing.API_name_processing import pokemon_names


current_date = datetime.now()
current_date = current_date.strftime("%Y-%m-%d")


# Track API calls
api_call_count = 0  

#new processed file
bulk_csv_path = f"processed_data\\processed_bulk-{current_date}.csv"


def create_full_csv(reader, bulk_csv):
    # Check if the file exists and if it's empty
    if not os.path.exists(bulk_csv) or os.path.getsize(bulk_csv) == 0:
        write_header = True
    else:
        write_header = False

    with open(bulk_csv, mode="a", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file)

        # Write header if the file is empty
        if write_header:
            writer.writerow([
                "Name", "Number", "Set", "Rarity", "HP",
                "Attacks", "Weaknesses", "Low Price", 
                "Mid Price", "Market Price", "Image URL"
            ])

        for row in reader:
            name = row[1]  
            number = row[2]
            set_total = row[3]

            if name in pokemon_names:
                print(f"{name} found!")

                # **API Call to Fetch Pokémon Information**
                query_result = Card.where(q=f"name:{name} number:{number}")

                global api_call_count
                api_call_count += 1  # Increment API call counter

                if not query_result:
                    continue

                # **Loop through each returned card**
                for card in query_result:
                    card_name = card.name
                    card_number = card.number
                    set_name = card.set.name if card.set else "Unknown"
                    rarity = card.rarity if card.rarity else "Unknown"
                    hp = card.hp if card.hp else "N/A"

                    # Extract Attacks
                    attacks = "; ".join([f"{atk.name} ({', '.join(atk.cost)})" for atk in card.attacks]) if card.attacks else "None"

                    # Extract Weaknesses
                    weaknesses = "; ".join([f"{w.type} ({w.value})" for w in card.weaknesses]) if card.weaknesses else "None"

                    # Extract Prices
                    prices = card.tcgplayer.prices if card.tcgplayer else None
                    low_price = prices.normal.low if prices and prices.normal else "N/A"
                    mid_price = prices.normal.mid if prices and prices.normal else "N/A"
                    market_price = prices.normal.market if prices and prices.normal else "N/A"

                    # Extract Image URL
                    image_url = card.images.large if card.images else "N/A"

                    # Write data to CSV (Each card gets a separate row)
                    writer.writerow([
                        card_name, card_number, set_name, rarity, hp,
                        attacks, weaknesses, low_price, mid_price, 
                        market_price, image_url
                    ])
            else: 
                print(f"Card Name '{name}' not in the fetched cards from API")



def name_check():
    global api_call_count  # Allow modifying the global counter

    if not os.path.exists(csv_file):
        write_name_file()

    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        
        print("\nDo you want\n 1. Create a New CSV Database\n 2. Add to Existing Database")
        
        mode_answer = input("Please Enter 1 or 2 and press Enter").strip()
        
        if mode_answer == '1':
            # Open the file in write mode to clear its contents
            with open(bulk_csv_path, mode='w', newline='', encoding="utf-8") as out_file:
                pass  # The file is now empty, but still exists

            create_full_csv(reader, bulk_csv_path)
            
        else:
            create_full_csv(reader, bulk_csv_path)

    print(f"Total API Calls: {api_call_count}")  # Final count after completion


if __name__ == "__main__":
    name_check()  # This will only run if executing `name_check.py` directly