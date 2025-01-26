from card_classifier import *
from read_number import *
import os
import re




# Path to the trainer logo template
template_path = "C:/Users/Loya/source/cardScanner/images/templates/trainer_logo.jpg" 


# Set the Tesseract path if necessary
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# Iterates through the files found in the directory and:
#         -Checks if image type
#         -identifies the ROIs
#         -preprocesses the image
#         -extracts name from the rois
#         


#         -TODO: Add card number rois
#         -saves card name, number and type to csv

#Image Directory
image_folder = "C:/Users/Loya/source/cardScanner/images"

for file in os.listdir(image_folder):
    
    # Check for jpeg type file
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        
        file_path = os.path.join(image_folder, file)
        
        # Get the card type and appropriate ROIs
        try:
            print("Attempting file " + file_path)
            result = get_card_rois(file_path, template_path)
            if result is None:
                print("image = none")
                exit()
            card_type = result["card_type"]
            name_roi = result["name_roi"]
            #card_number_roi = result["card_number_roi"]
            trainer_logo_rois = result["trainer_logo_rois"]
        except FileNotFoundError as e:
            print(e)
            exit()

        # Extract text using the determined ROIs
        image, thresh = preprocess_image(file_path)

        if image is None or thresh is None:
            print("Error processing image.")
        else:
            #show image with overlay of ROI
            # use this one you want the card number 
            # # rois = [name_roi, card_number_roi] + trainer_logo_rois
            rois = [name_roi] + trainer_logo_rois
            overlayed_image = visualize_rois(image, rois, text=card_type)
            
            #get text from ROI
            name = extract_text(thresh, name_roi)
            #card_number = extract_text(thresh, card_number_roi)
            #card_number = extract_text(thresh)

            print(f"Card Type: {card_type}")
            print(f"Name: {name}")
            #print(f"Card Number: {card_number}")
            
            #show overlay
            cv2.imshow("ROIs Overlay", overlayed_image)
            cv2.waitKey(0)  # Wait until a key is pressed
            cv2.destroyAllWindows()
            
        image, thresh = preprocess_number(file_path)

        if image is not None and thresh is not None:
            result = ROI_number_check(thresh, number_roi_list)
            set_numbers = result.split('/')
            card_number = set_numbers[0]
            set_total = set_numbers[1]
            if result:
                print(f"Number detected in ROI:{card_number}")
            else:
                print("No valid number found.")




