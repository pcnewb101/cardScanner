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


def multi_scale_match(cropped, trainer_logo, min_scale=0.5, max_scale=1.5, scale_step=0.1):
    """
    Perform multi-scale matching to detect the trainer logo within a cropped region.
    
    Args:
        cropped (numpy.ndarray): The image cropped from the original image to check.
        trainer_logo (numpy.ndarray): The template image of the trainer logo.
        min_scale (float): The minimum scale to resize the logo.
        max_scale (float): The maximum scale to resize the logo.
        scale_step (float): The step size for scaling the logo.
    
    Returns:
        bool: True if a match is found, False otherwise.
    """
    crop_height, crop_width = cropped.shape[:2]
    logo_height, logo_width = trainer_logo.shape[:2]

    # Check if the logo is smaller than the cropped image (resize if necessary)
    if logo_height > crop_height or logo_width > crop_width:
        scale_factor = min(crop_height / logo_height, crop_width / logo_width)
        resized_logo = cv2.resize(trainer_logo, (int(logo_width * scale_factor), int(logo_height * scale_factor)))
        print(f"Resizing logo to fit cropped region: {resized_logo.shape}")
    else:
        resized_logo = trainer_logo

    # Ensure that the resized logo is not still larger than the cropped area
    if resized_logo.shape[0] > crop_height or resized_logo.shape[1] > crop_width:
        print(f"Skipping scale match: Resized logo size {resized_logo.shape} is still larger than cropped size {cropped.shape}")
        return False

    # Perform template matching on different scales
    for scale in np.arange(min_scale, max_scale, scale_step):
        scaled_logo = cv2.resize(resized_logo, (int(resized_logo.shape[1] * scale), int(resized_logo.shape[0] * scale)))
        
        # Perform template matching
        result = cv2.matchTemplate(cropped, scaled_logo, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # If the match value exceeds the threshold, return True
        if max_val > 0.8:  # Adjust threshold as needed
            print(f"Match found with scale {scale}. Max value: {max_val}")
            return True

    print("No match found at any scale.")
    return False


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
    
    # Apply adaptive thresholding to the logo to convert it to binary (black and white)
    _, trainer_logo_bin = cv2.threshold(trainer_logo, 127, 255, cv2.THRESH_BINARY)

    for roi in roi_list:
        x, y, w, h = roi
        cropped = thresh[y:y+h, x:x+w]

        # Apply adaptive thresholding to the cropped image to make it binary (black and white)
        _, cropped_bin = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)

        # Resize the trainer logo to match the cropped region size if necessary
        if cropped_bin.shape[0] < trainer_logo_bin.shape[0] or cropped_bin.shape[1] < trainer_logo_bin.shape[1]:
            trainer_logo_bin_resized = cv2.resize(trainer_logo_bin, (cropped_bin.shape[1], cropped_bin.shape[0]))
        else:
            trainer_logo_bin_resized = trainer_logo_bin

        # Perform template matching
        result = cv2.matchTemplate(cropped_bin, trainer_logo_bin_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > 0.8:  # Adjust threshold as needed
            return True
    return False


def extract_text(thresh, roi):
    """Extract text from a given ROI in the thresholded image."""
    x, y, w, h = roi
    roi_image = thresh[y:y+h, x:x+w]
    text = pytesseract.image_to_string(roi_image, config='--psm 6')  # Using PSM 6 for sparse text
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
        (165, 100, 900, 250),  # ROI 1 for trainer logo
        (1845, 70, width - 1845, 180),  # ROI 2 for trainer logo
    ]
    
    # Check if it's a trainer card
    is_trainer = match_logo(thresh, trainer_logo_path, trainer_logo_rois)
    
    if is_trainer:
        return {
            "card_type": "Trainer",
            "name_roi": (150, 100, 1850, 350),
            "card_number_roi": (100, 100, 1000, 150),
        }
    else:
        return {
            "card_type": "Pokemon",
            "name_roi": (475, 135, 1200, 200),
            "card_number_roi": (1500, 800, 2000, 100),
        }

# Main code
image_path = "images/trainer1_image.jpg"
trainer_logo_path = "images/templates/trainer_logo.jpg"  # Path to the trainer logo template

# Get the card type and appropriate ROIs
try:
    result = get_card_rois(image_path, trainer_logo_path)
    if result is None:
        exit()
    card_type = result["card_type"]
    name_roi = result["name_roi"]
    card_number_roi = result["card_number_roi"]
except FileNotFoundError as e:
    print(e)
    exit()

# Extract text using the determined ROIs
image, thresh = preprocess_image(image_path)

if image is None or thresh is None:
    print("Error processing image.")
else:
    name = extract_text(thresh, name_roi)
    card_number = extract_text(thresh, card_number_roi)

    print(f"Card Type: {card_type}")
    print(f"Name: {name}")
    print(f"Card Number: {card_number}")
    
# Example: Display the preprocessed template
template_path = "images/templates/trainer_logo.jpg"
template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

# Preprocess the template image (optional)
blurred_template = cv2.GaussianBlur(template_image, (5, 5), 0)
_, thresh_template = cv2.threshold(blurred_template, 127, 255, cv2.THRESH_BINARY)

# Show the processed image
cv2.imshow("Preprocessed Template", thresh_template)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()
