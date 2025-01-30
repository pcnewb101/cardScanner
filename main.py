from datetime import datetime
import csv

current_date = datetime.now()

current_date = current_date.strftime("%Y-%m-%d")
csv_file = f"processed_data\\bulk_cards-{current_date}.csv"


if csv_file in os.listdir("C:/Users/Loya/source/cardScanner"):
    
    print("File for today exists.")
    
    key_num = 0

    answer = input("Process New Data? (y/n)".strip().lower())

    if answer == "y":
        run read card             
    else:
        print("No data processed.")