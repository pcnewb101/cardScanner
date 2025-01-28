from datetime import datetime
import csv
from read_Card import processed_data


current_date = datetime.now()

current_date = current_date.strftime("%Y-%m-%d")


csv_file = f"processed_data\\bulk_cards-{current_date}.csv"


with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # Write header
    writer.writerow(["Index", "name", "number", "printedTotal"])

    # Process the dictionary
    for key, value in processed_data.items():
        writer.writerow([key, value["name"], value["number"], value["printedTotal"]])
    