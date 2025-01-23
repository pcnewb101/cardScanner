import cv2
import pytesseract
import numpy as np

# Set the Tesseract path if necessary
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def preprocess_image(image_path):
    """Preprocess the image to improve clarity for text extraction."""
    image = cv2.imread(image_path)
    if image is None:
        print("Image failed to upload.")
        return None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast enhancement (optional step)
    contrast_img = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    
    # Reduce or remove Gaussian blur to preserve text details
    blurred = cv2.GaussianBlur(contrast_img, (3, 3), 0)  # Apply blur after contrast enhancement
    
    # Use adaptive thresholding for better contrast between text and background
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
        
    return image, thresh

def visualize_rois(image, rois, text=None):
    """Overlay ROIs and optional text on the image for debugging."""
    for i, roi in enumerate(rois):
        x, y, w, h = roi
        color = (0, 255, 0)  # Green for ROIs
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        if text:
            cv2.putText(image, f"{text} ROI {i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def match_logo(thresh, trainer_logo_path, roi_list, min_scale=0.5, max_scale=1.5, scale_step=0.1):
    """
    Check if the trainer logo exists in any of the specified ROIs with multi-scale matching.

    Args:
        thresh (numpy.ndarray): Preprocessed thresholded image.
        trainer_logo_path (str): Path to the trainer logo template.
        roi_list (list): List of ROIs (x, y, w, h) to check.
        min_scale (float): The minimum scale to resize the logo.
        max_scale (float): The maximum scale to resize the logo.
        scale_step (float): The step size for scaling the logo.

    Returns:
        bool: True if a match is found, False otherwise.
    """
    trainer_logo = cv2.imread(trainer_logo_path, cv2.IMREAD_GRAYSCALE)
    if trainer_logo is None:
        raise FileNotFoundError(f"Trainer logo template not found: {trainer_logo_path}")

    # Apply adaptive thresholding to the logo to convert it to binary (black and white)
    _, trainer_logo_bin = cv2.threshold(trainer_logo, 127, 255, cv2.THRESH_BINARY)

    for roi in roi_list:
        x, y, w, h = roi
        cropped = thresh[y:y+h, x:x+w]

        # Apply adaptive thresholding to the cropped image to make it binary (black and white)
        _, cropped_bin = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)

        # Perform multi-scale matching within the ROI
        for scale in np.arange(min_scale, max_scale, scale_step):
            # Resize the trainer logo at the current scale
            scaled_logo = cv2.resize(
                trainer_logo_bin, 
                (int(trainer_logo_bin.shape[1] * scale), int(trainer_logo_bin.shape[0] * scale))
            )

            # Skip if the scaled logo is larger than the cropped ROI
            if scaled_logo.shape[0] > cropped_bin.shape[0] or scaled_logo.shape[1] > cropped_bin.shape[1]:
                continue

            # Perform template matching
            result = cv2.matchTemplate(cropped_bin, scaled_logo, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            # Check if the match value exceeds the threshold
            if max_val > 0.8:  # Adjust threshold as needed
                print(f"Match found in ROI with scale {scale}. Max value: {max_val}")
                return True

    print("No match found at any scale or ROI.")
    return False


def extract_text(thresh, roi):
    """Extract text from a given ROI in the thresholded image."""
    x, y, w, h = roi
    roi_image = thresh[y:y+h, x:x+w]
    text = pytesseract.image_to_string(roi_image, config='--psm 13')  # PSM 8 and 13 seems to do the best for single line extract.
    return text.strip()



def get_card_rois(image_path, trainer_logo_path):
    """
    Determine the card type and return appropriate ROIs.
    
    Args:
        image_path (str): Path to the card image.
        trainer_logo_path (str): Path to the trainer logo template.
    
    Returns:
        dict: Dictionary containing the determined card type and relevant ROIs.
    """
    image, thresh = preprocess_image(image_path)
    if image is None or thresh is None:
        print("Error processing image.")
        return None
    
    height, width = image.shape[:2]
    
    # Define ROIs for trainer logo
    trainer_logo_rois = [
        (15, 22, 275, 40),  # ROI 1 for trainer logo
        (465, 15, width, 65),  # ROI 2 for trainer logo
    ]
    
    # Check if it's a trainer card
    is_trainer = match_logo(thresh, trainer_logo_path, trainer_logo_rois)
    
    if is_trainer:
        return {
            "card_type": "Trainer",
            "name_roi": (50, 80, 250, 50),
            #"card_number_roi": (100, 100, 1000, 150),
            "trainer_logo_rois": trainer_logo_rois,
        }
    else:
        return {
            "card_type": "Pokemon",
            "name_roi": (135, 35, 400, 60),
            #"card_number_roi": (1500, 800, height - 300, 100),
            "trainer_logo_rois": trainer_logo_rois,
        }

# Main code
#image_path = "images/templates/scanned_pokemon.JPG"
image_path = "images/templates/trainer_template1.JPG"
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
    
# Example: Display the preprocessed template



# Preprocess the template image (optional)
def preprocess_template(image_loc):
    #call the image fils and applies a blur and converts to B&W
    template = cv2.imread(image_loc, cv2.IMREAD_GRAYSCALE)
    blurred_template = cv2.GaussianBlur(template, (5, 5), 0)
    _, template = cv2.threshold(blurred_template, 127, 255, cv2.THRESH_BINARY)   
    
    return template

precessed_template = preprocess_template(template_path)


# Show the processed image
cv2.imshow("Preprocessed Template", precessed_template)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()
