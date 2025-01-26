## will define the process to find ROIs of set numbers and the preprocessing of those cropped images.
## Different preprocessing setting are likely needed and as such will have a different function calls.
## Ideally, these functions will be called in read_card py so each image only has to be access in one section of time/memory.


# THREE DIFFERENT METHODS ARE PROVIDED HERE. NONE ARE PRODUCING THE DESIRED OUTPUT YET

import cv2
import numpy as np

# list of symbol location ROIs

# roi_locations = [
#     (35, 970, 200, 40),
#     (),
#     ]

# loop through 

# def locate_symbol(image, template, search_region=(35, 970, 200, 40), threshold=0.6):
#     """
#     Locate the symbol within a defined search region using ORB feature matching.
    
#     Args:
#         image (numpy.ndarray): The input image.
#         template (numpy.ndarray): The symbol template (grayscale).
#         search_region (tuple): The (x, y, width, height) of the search area.
#         threshold (float): The matching threshold to identify the symbol.
    
#     Returns:
#         tuple: (x, y, w, h) of the detected symbol's bounding box, or None if not found.
#     """
#     # Convert the input image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply ORB feature matching
#     orb = cv2.ORB_create()
#     keypoints_image, descriptors_image = orb.detectAndCompute(gray_image, None)
#     keypoints_template, descriptors_template = orb.detectAndCompute(template, None)

#     # Check if descriptors are None (ORB didn't detect features)
#     if descriptors_image is None or descriptors_template is None:
#         print("Error: No descriptors found in image or template.")
#         return None

#     # Ensure descriptors are of the same type
#     descriptors_image = np.uint8(descriptors_image)
#     descriptors_template = np.uint8(descriptors_template)

#     # Use a matcher (e.g., BFMatcher) to find best matches
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(descriptors_image, descriptors_template)

#     # Sort them based on distance (closer means better match)
#     matches = sorted(matches, key=lambda x: x.distance)

#     # Draw the top matches
#     result_image = cv2.drawMatches(image, keypoints_image, template, keypoints_template, matches[:10], None)

#     # Debug: Show the top matches
#     cv2.imshow("ORB Match", result_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # If enough matches are found, calculate the bounding box
#     if len(matches) > 5:  # At least 5 good matches to consider
#         # Get the matching points
#         points_image = np.float32([keypoints_image[m.queryIdx].pt for m in matches])
#         points_template = np.float32([keypoints_template[m.trainIdx].pt for m in matches])

#         # Use findHomography to compute the bounding box of the detected symbol
#         matrix, mask = cv2.findHomography(points_template, points_image, cv2.RANSAC, 5.0)

#         # Get the bounding box coordinates of the matched template
#         h, w = template.shape
#         corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
#         transformed_corners = cv2.perspectiveTransform(corners, matrix)

#         # Get the bounding box (min and max x, y coordinates)
#         x, y, w, h = cv2.boundingRect(transformed_corners)
#         return (x, y, w, h)

#     else:
#         print(f"Not enough matches found. Matches: {len(matches)}")
#         return None

# # Example usage
# image = cv2.imread("images\\pokemon_template.jpg")
# template = cv2.imread(r"images\set_symbols\Darkness Ablaze.png", cv2.IMREAD_GRAYSCALE)

# symbol_location = locate_symbol(image, template, search_region=(35, 970, 200, 40))
# if symbol_location:
#     x, y, w, h = symbol_location
#     print(f"Symbol found at: {x}, {y}, {w}, {h}")
#     # Draw a rectangle around the detected symbol
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.imshow("Detected Symbol", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Symbol not found. Debugging information displayed.")




# def locate_symbol(image, template, search_region=(35, 970, 200, 40), threshold=0.6):
#     """
#     Locate the symbol within a defined search region using template matching and edge detection.
    
#     Args:
#         image (numpy.ndarray): The input image.
#         template (numpy.ndarray): The symbol template (grayscale).
#         search_region (tuple): The (x, y, width, height) of the search area.
#         threshold (float): The matching threshold to identify the symbol.
    
#     Returns:
#         tuple: (x, y, w, h) of the detected symbol's bounding box, or None if not found.
#     """
#     # Convert the input image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply Canny edge detection to both the image and the template
#     edges_image = cv2.Canny(gray_image, 225, 255)
#     edges_template = cv2.Canny(template, 1, 255)

#     # Resize the template to match the expected size
#     resized_template = cv2.resize(edges_template, (40, 40))
    
#     _, binary_template = cv2.threshold(resized_template, 1, 255, cv2.THRESH_BINARY)
    
    
#     # Define the search area
#     x, y, w, h = search_region
#     cropped = edges_image[y:y + h, x:x + w]
    
    
#     # Use adaptive thresholding for better contrast between text and background
#     thresh = cv2.adaptiveThreshold(
#         cropped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 1
#     )
    

#     # Perform template matching on the edge-detected images
#     result = cv2.matchTemplate(thresh, binary_template, cv2.TM_CCOEFF_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

#     # Check if the match exceeds the threshold
#     if max_val >= threshold:
#         match_x, match_y = max_loc  # Top-left corner of the match in the cropped region
#         h, w = binary_template.shape  # Template dimensions
#         return (x + match_x, y + match_y, w, h)  # Adjust to original image coordinates
#     else:
#         # Debugging: Show the cropped search area and the input image if no match is found
#         print(f"Symbol not found. Max value: {max_val}")
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw search area
#         cv2.imshow("Search Area on Original Image", image)
#         cv2.imshow("Cropped Search Area", thresh)
#         cv2.imshow("Edge-detected Template", binary_template)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         return None

# # Example usage
# image = cv2.imread(r"images\pokemon_template.jpg")
# template = cv2.imread(r"images\set_symbols\Darkness Ablaze.png", cv2.IMREAD_GRAYSCALE)

# symbol_location = locate_symbol(image, template, search_region=(35, 970, 200, 40))
# if symbol_location:
#     x, y, w, h = symbol_location
#     print(f"Symbol found at: {x}, {y}, {w}, {h}")
#     # Draw a rectangle around the detected symbol
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.imshow("Detected Symbol", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Symbol not found. Debugging information displayed.")


# def visualize_rois(image, rois, text=None):
#     """Overlay ROIs and optional text on the image for debugging."""
#     for i, roi in enumerate(rois):
#         x, y, w, h = roi
#         color = (0, 255, 0)  # Green for ROIs
#         cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
#         if text:
#             cv2.putText(image, f"{text} ROI {i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#     return image




def locate_symbol(image, template, search_region=(30, 920, 250, 100), threshold=0.5):
    """
    Locate the symbol within a defined search region using contour matching.
    
    Args:
        image (numpy.ndarray): The input image.
        template (numpy.ndarray): The symbol template (grayscale).
        search_region (tuple): The (x, y, width, height) of the search area.
        threshold (float): The similarity threshold to identify the symbol (lower is better).
    
    Returns:
        tuple: (x, y, w, h) of the detected symbol's bounding box, or None if not found.
    """
    # Convert the input image and template to grayscale if needed
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Preprocess the template to extract contours
    _, binary_template = cv2.threshold(template, 1, 255, cv2.THRESH_BINARY)
    contours_template, _ = cv2.findContours(binary_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour from the template
    if len(contours_template) == 0:
        print("No contours found in the template.")
        return None
    template_contour = max(contours_template, key=cv2.contourArea)

    # Define the search area in the image
    x, y, w, h = search_region
    cropped = gray_image[y:y + h, x:x + w]

    # Preprocess the cropped search area
    _, binary_cropped = cv2.threshold(cropped, 220, 255, cv2.THRESH_BINARY)
    contours_cropped, _ = cv2.findContours(binary_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("processed_cropped", binary_cropped)
    cv2.imshow("template", binary_template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Compare the template contour with each contour in the search area
    best_match = None
    best_score = float('inf')
    for contour in contours_cropped:
        # Compute the similarity using cv2.matchShapes
        similarity_score = cv2.matchShapes(template_contour, contour, cv2.CONTOURS_MATCH_I1, 0.0)
        if similarity_score < best_score:
            best_score = similarity_score
            best_match = contour

    # Check if the best match meets the threshold
    if best_score <= threshold:
        x, y, w, h = cv2.boundingRect(best_match)
        return (x + search_region[0], y + search_region[1], w, h)  # Adjust coordinates to the original image
    else:
        print(f"Symbol not found. Best similarity score: {best_score}")
        return None

# Example usage
image = cv2.imread(r"images\pokemon_template.jpg")
template = cv2.imread(r"images\set_symbols\Darkness Ablaze.png", cv2.IMREAD_GRAYSCALE)

symbol_location = locate_symbol(image, template, search_region=(35, 950, 200, 60))
if symbol_location:
    # Extract the coordinates of the detected symbol
    match_x, match_y, match_w, match_h = symbol_location
    print(f"Symbol found at: {match_x}, {match_y}, {match_w}, {match_h}")
    
    # Draw a rectangle only around the detected symbol
    cv2.rectangle(image, (match_x, match_y), (match_x + match_w, match_y + match_h), (0, 255, 0), 2)
    
    # Show the result
    cv2.imshow("Detected Symbol", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Symbol not found. Debugging information displayed.")

