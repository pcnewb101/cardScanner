from processing.API_name_processing import pokemon_names
from processing.write_file import csv_file
import csv


print(f"Sample: {pokemon_names[:10]}")

with open(csv_file, mode="r") as file:
    # Create a CSV reader object
    reader = csv.reader(file)
    
    # Skip the header if there's one
    next(reader)  # Uncomment this line if the CSV has a header
    
    # Iterate through each row in the CSV
    for row in reader:
        name = row[1]  # 'name' is in the second column (index 1)
        
        if name in pokemon_names:
            print(f"{name} found!")
        else: 
            cleaned_name = 