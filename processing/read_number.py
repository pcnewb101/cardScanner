import cv2
import pytesseract
import re


# only one roi to check at the moment but allows quick additional rois if needed
number_roi_list = [(115, 980, 90, 25)]

def preprocess_number(image_path):
    """Preprocess the image to improve clarity for text extraction."""
    image = cv2.imread(image_path)
    if image is None:
        print("Image failed to upload.")
        return None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast (optional)
    contrast_img = cv2.convertScaleAbs(gray, alpha=1.5, beta=1)
    
    # Adaptive thresholding for binarization
    thresh = cv2.adaptiveThreshold(
        contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 99
    )
    return image, thresh


def ROI_number_check(thresh, number_roi_list):
    """
    Check if numbers exist in any of the specified ROIs.
    """
    for roi in number_roi_list:
        x, y, w, h = roi
        cropped = thresh[y:y+h, x:x+w]

        # OCR on cropped ROI
        numbers = pytesseract.image_to_string(cropped, config='--psm 7')
        
        #Debugging statements
        """ print(f"OCR output: '{numbers}'")  # Debugging
        cv2.imshow("Cropped ROI", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
        
        if re.match(r"\d+/\d+", numbers.strip()):
            return numbers.strip()
        

    
    print("No numbers found")
    return False

# TODO: make numbers 0 when no numbers are found
