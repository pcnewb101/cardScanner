from datetime import datetime
import csv
import os
from processing.read_Card import read_card


current_date = datetime.now()

current_date = current_date.strftime("%Y-%m-%d")


csv_file = f"processed_data\\imaged_cards-{current_date}.csv"


def write_name_file():
    
    
    
    processed_data = read_card()
    
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Write header
        writer.writerow(["Index", "name", "number", "printedTotal"])

        # Process the dictionary
        for key, value in processed_data.items():
            writer.writerow([key, value["name"], value["number"], value["printedTotal"]])




#writing the initial file occurs in the name_check file

if os.path.exists(csv_file):
    
    print("File with extracted names and number for today already exists.")
    print("Do you want to Overwrite Processed Image Data?")
    answer = input("(y/n)").strip().lower()


    if answer == "y":

        write_name_file()





