
import csv

""" Running this file as a module (python -m main) will
    - check the images file for content
    - process the images (processing directory)
        - check for type of card (energy, trainer, or Pokémon)  # card_classifier
        - adjust ROI of OCR for name and number  # read_Card # read_number 
        - store the processed information into file  # write_file
    - compare the processed information to existing list of Pokémon data  # integration\\check_name
    - if match is found:
        - structure an API call to retrieve all data, including prices
        - populate a new CSV file with all the card’s details"""

def main():
    print("Hello! Would you like to start the process of checking for images, processing them and generating new CSV files?")
    answer = input("y/n: ").strip().lower()

    if answer == "y":
        from integration.name_check import name_check
        name_check()
        
    else:
        print("Thank you. Program ending.")

# Prevent execution on import
if __name__ == "__main__":
    main()