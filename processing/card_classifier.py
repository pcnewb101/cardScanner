import cv2
import pytesseract
import numpy as np

# Set the Tesseract path if necessary
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def preprocess_image(image_path):
    """Preprocess the image for thresholding."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image failed to load. Check the path: {image_path}")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return image, thresh

def match_logo(thresh, trainer_logo_path, roi_list):
    """
    Check if the trainer logo exists in any of the specified ROIs.
    
    Args:
        thresh (numpy.ndarray): Preprocessed thresholded image.
        trainer_logo_path (str): Path to the trainer logo template.
        roi_list (list): List of ROIs (x, y, w, h) to check.
    
    Returns:
        bool: True if a match is found, False otherwise.
    """
    trainer_logo = cv2.imread(trainer_logo_path, cv2.IMREAD_GRAYSCALE)
    if trainer_logo is None:
        raise FileNotFoundError(f"Trainer logo template not found: {trainer_logo_path}")
    
    for roi in roi_list:
        x, y, w, h = roi
        cropped = thresh[y:y+h, x:x+w]

        # Check if template is larger than cropped and resize if necessary
        if cropped.shape[0] < trainer_logo.shape[0] or cropped.shape[1] < trainer_logo.shape[1]:
            print(f"Resizing template from {trainer_logo.shape} to {cropped.shape}")
            trainer_logo = cv2.resize(trainer_logo, (cropped.shape[1], cropped.shape[0]))

        result = cv2.matchTemplate(cropped, trainer_logo, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > 0.8:  # Adjust threshold as needed
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
    
    height, width = image.shape[:2]
    
    # Define ROIs for trainer logo
    trainer_logo_rois = [
        (165, 100, 900, 250),  # ROI 1 for trainer logo
        (1845, 70, width - 1845, 180),  # ROI 2 for trainer logo
    ]
    
    # Print ROIs for debugging
    print(f"Trainer Logo ROIs: {trainer_logo_rois}")

    # Check if it's a trainer card
    is_trainer = match_logo(thresh, trainer_logo_path, trainer_logo_rois)
    
    if is_trainer:
        print("Trainer card identified.")
        return {
            "card_type": "Trainer",
            "name_roi": (150, 100, 1850, 350),
            "card_number_roi": (100, 100, 1000, 150),
        }
    else:
        print("Pokemon card identified.")
        return {
            "card_type": "Pokemon",
            "name_roi": (475, 135, 1200, 200),
            "card_number_roi": (1500, 800, 2000, 100),
        }
        


