import cv2
import pytesseract
import numpy as np
import re

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
    contrast_img = cv2.convertScaleAbs(gray, alpha=1.5, beta=-10)
    
    # Reduce or remove Gaussian blur to preser e text details
    blurred = cv2.GaussianBlur(contrast_img, (15, 17), 0)  # Apply blur after contrast enhancement
    
    # Use adaptive thresholding for better contrast between text and background
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    
    return image, thresh



def trainer_text_test(thresh, roi_list):
    """
    Check if the "Trainer" word exists in any of the specified ROIs with multi-scale matching.

    Args:
        thresh (numpy.ndarray): Preprocessed thresholded image.
        roi_list (list): List of ROIs (x, y, w, h) to check.


    Returns:
        bool: True if a match is found, False otherwise.
    """
    for roi in roi_list:
        x, y, w, h = roi
        cropped = thresh[y:y+h, x:x+w]

        # Apply adaptive thresholding to the cropped image to make it binary (black and white)
        _, cropped_bin = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)

        logo_text = pytesseract.image_to_string(cropped_bin, config ='--psm 13')

        if re.match(r"(?i)Trainer\s*", logo_text):
            return True
                
    return False



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
    is_trainer = trainer_text_test(thresh, trainer_logo_rois)
    
    if is_trainer:
        return {
            "card_type": "Trainer",
            "name_roi": (30, 70, 300, 65),
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



def extract_text(thresh, roi):
    """Extract text from a given ROI in the thresholded image."""
    x, y, w, h = roi
    roi_image = thresh[y:y+h, x:x+w]
    text = pytesseract.image_to_string(roi_image, config='--psm 13')  # PSM 8 and 13 seems to do the best for single line extract.
    return text.strip()




        
def visualize_rois(image, rois, text=None):
    """Overlay ROIs and optional text on the image for debugging."""
    for i, roi in enumerate(rois):
        x, y, w, h = roi
        color = (0, 255, 0)  # Green for ROIs
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        if text:
            cv2.putText(image, f"{text} ROI {i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image