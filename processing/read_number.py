import cv2
import pytesseract
import re

# only one roi to check at the moment but allows quick additional rois if needed
number_roi_list = [(115, 980, 75, 30), (120, 990, 80, 25)]

# parameter inputs for varient of processed image
gran_varients = [[71,71], [99,99], [51,51], [71,13], [13,71], [13,13]]


def test_varients(image_path):
    """Loop through varients of granular size/blur to obtain a set of images to be tested"""
    
    image_vaients = []
    
    # load image
    image = cv2.imread(image_path)
    if image is None:
        print("Image failed to upload.")
        return None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast (optional)
    contrast_img = cv2.convertScaleAbs(gray, alpha=1.5, beta=1)
    
    # Loop for Adaptive Thresholding for binarization
    for set in gran_varients:
        gran1 = set[0]
        gran2 = set[1]
        thresh = cv2.adaptiveThreshold(
            contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, gran1, gran2
        )
        
        image_vaients.append(thresh)
        
    return image_vaients



def ROI_number_check(thresh, number_roi_list):
    """
    Check if numbers exist in any of the specified ROIs and returns a list of found values
    """
    found_values = []
    
    for roi in number_roi_list:
        x, y, w, h = roi
        cropped = thresh[y:y+h, x:x+w]

        # OCR on cropped ROI
        numbers = pytesseract.image_to_string(cropped, config='--psm 7')
        
        #Debugging statements
        # print(f"OCR output: '{numbers}'")  # Debugging
        # cv2.imshow("Cropped ROI", cropped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Remove any non-numeric characters except for the '/' (this handles garbage inputs)
        cleaned_numbers = re.sub(r"[^\d/]", "", numbers.strip())
        
        # Check if the cleaned string matches the format of "xx/xx"
        checked_format = re.fullmatch(r"\d+/\d+", cleaned_numbers)
        
        if checked_format is not None:
            found_values.append(checked_format.group())  # Return the valid match found
            
            
    return found_values



    

# TODO: make numbers 0 when no numbers are located