from card_classifier import *



#image_path = "images/templates/pokemon_template.JPG"
image_path = "images/templates/trainer_template2.JPG"
template_path = "images/templates/trainer_logo.jpg"  # Path to the trainer logo template

# Get the card type and appropriate ROIs
try:
    result = get_card_rois(image_path, template_path)
    if result is None:
        exit()
    card_type = result["card_type"]
    name_roi = result["name_roi"]
    #card_number_roi = result["card_number_roi"]
    trainer_logo_rois = result["trainer_logo_rois"]
except FileNotFoundError as e:
    print(e)
    exit()

# Extract text using the determined ROIs
image, thresh = preprocess_image(image_path)

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


# Preprocess the template image (optional)
def preprocess_template(image_loc):
    #call the image fils and applies a blur and converts to B&W
    template = cv2.imread(image_loc, cv2.IMREAD_GRAYSCALE)
    blurred_template = cv2.GaussianBlur(template, (5, 5), 0)
    _, template = cv2.threshold(blurred_template, 127, 255, cv2.THRESH_BINARY)   
    
    return template

precessed_template = preprocess_template(template_path)

