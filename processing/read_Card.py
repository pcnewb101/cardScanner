from card_classifier import *
from read_number import *
import os
import numpy as np
from scipy.stats import mode



# Path to the trainer logo template
template_path = "C:/Users/Loya/source/cardScanner/images/templates/trainer_logo.jpg" 


# Set the Tesseract path if necessary
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# Iterates through the files found in the directory and:
#         -Checks if image type
#         -identifies the ROIs
#         -preprocesses the image
#         -extracts name from the rois
#         -extracts card number rois
#         -saves card name, number and set total to csv

#Image Directory
image_folder = "C:/Users/Loya/source/cardScanner/images"

# Dictionary of processed info. Key = auto incremented index. Structure = [key] = [name, number, total set number]

processed_data = {}

key_num = 0

for file in os.listdir(image_folder):
    
    # Check for jpeg type file
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        
        key_num += 1
        
        file_path = os.path.join(image_folder, file)
        
        # Get the card type and appropriate ROIs
        try:
            print("Attempting file " + file_path)
            result = get_card_rois(file_path)
            if result is None:
                print("image = none")
                exit()
            card_type = result["card_type"]
            name_roi = result["name_roi"]
            #card_number_roi = result["card_number_roi"]
            roi = result["roi"]
        except FileNotFoundError as e:
            print(e)
            exit()

        # Extract text using the determined ROIs
        image, thresh = preprocess_image(file_path)

        if image is None or thresh is None:
            print("Error processing image.")
        else:
            ## DEBUGGING
            # show image with overlay of ROI

            # rois = [name_roi] + roi
            # overlayed_image = visualize_rois(image, rois, text=card_type)
            
            #get text from ROI
            name = extract_text(thresh, name_roi)
            #card_number = extract_text(thresh, card_number_roi)
            #card_number = extract_text(thresh)

            print(f"Card Type: {card_type}")
            print(f"Name: {name}")
            #print(f"Card Number: {card_number}")
            
            #show overlay
            # cv2.imshow("ROIs Overlay", overlayed_image)
            # cv2.waitKey(0)  # Wait until a key is pressed
            # cv2.destroyAllWindows()
            
        images = test_varients(file_path)

        values = []
        
        set_values = []
        
        for image in images:
            
            str_list = ROI_number_check(image, number_roi_list)
            
            for set in str_list:
                set = re.sub(r'[^d\d/]', '', set.strip())              
                
                split_value = set.split('/')
                    
                card_number = split_value[0]
                set_total = split_value[1]
                values.append(int(card_number))
                set_values.append(int(set_total))
                
            mode_val = mode(values)
            mode_set = mode(set_values)
                
            
        if np.isnan(mode_val.mode):
            print("Unable to read Number\n")
            processed_data[key_num] = {
                "name": name,
                "number": "N/A",  # Handle NaN as a string
                "printedTotal": "N/A"
            }
        else:
            print(f"Number detected in ROI: {mode_val.mode}\n")
            processed_data[key_num] = {
                "name": name,
                "number": mode_val.mode,
                "printedTotal": mode_set.mode
            }





