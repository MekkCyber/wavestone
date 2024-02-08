import cv2
import numpy as np

def preprocess_captcha(captcha_image_path, number_characters=4):
    
    image = cv2.imread(captcha_image_path)
    
    # Convert the image to grayscale
    I1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # GAUSSIAN FILTERING
    I2 = cv2.GaussianBlur(I1, (0, 0), 2)
    
    # BINARIZE THE IMAGE USING OTSU'S THRESHOLDING
    _, thresh = cv2.threshold(I2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    #kernel = np.ones((3, 3), np.uint8)
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    #dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    #_, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    #sure_fg = np.uint8(sure_fg)
    #num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opening, connectivity=8)
    #largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    #character_mask = (labels == largest_label).astype(np.uint8) * 255
    #contours, _ = cv2.findContours(character_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # min_area = 0  
    # character_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    #character_contours = sorted(character_contours, key=lambda c: cv2.boundingRect(c)[0])
    
    # total_width = gray.shape[1]
    # segment_width = total_width // number_characters

    # character_images = []
    # for i in range(number_characters):
    #     # Calculate the start and end columns for the current character segment
    #     start_col = i * segment_width
    #     end_col = start_col + segment_width
        
    #     # Crop the current character segment
    #     character_segment = thresh[:, start_col:end_col]
        
    #     # Append the character segment to the list
    #     character_images.append(character_segment)
        

    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # mask = cv2.drawContours(
    #     np.zeros_like(gray),  # Create a black image with the same size as the original image
    #     contours, 
    #     contourIdx=-1,  # Draw all contours
    #     color=(255, 255, 255),  # Draw white contours
    #     thickness=cv2.FILLED  # Fill the contours
    # )

    # Apply the mask to the original image to remove the background
    # result = cv2.bitwise_and(image, image, mask=mask)

    # Display the result
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    return thresh

import os
import cv2
import imutils
import glob
import numpy as np
from imutils import paths, resize

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    """
    (h, w) = image.shape[:2]

    # resize along the largest axis
    if w > h:
        image = resize(image, width=width)
    else:
        image = resize(image, height=height)

    # padding values
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height)) # only to avoid rounding issues

    return image

def simple_imgs2chars(input_folder, output_folder):
    """
    Detects the characters in the simple captcha images
    Each character is saved in a separate file in gray scale
    """
    # Get a list of all the captcha images we need to process
    images = glob.glob(os.path.join(input_folder, "*"))
    counts = {}

    # loop over the image paths
    for (i, img_path) in enumerate(images):
        print("[INFO] processing image {}/{}".format(i + 1, len(images)))

        # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
        # grab the base filename as the text
        filename = os.path.basename(img_path)
        img_text = os.path.splitext(filename)[0]

        img, gray = simple_thresh(img_path)

        char_bound_boxes = img2boxes(img, lambda w, h: w / h > 1.25, num_chars = 4, cv2_chain = cv2.CHAIN_APPROX_SIMPLE)
        
        if char_bound_boxes is None:
            continue

        for char_bound_box, char in zip(char_bound_boxes, img_text):
            counts = crop_and_save(char_bound_box, char, gray, output_folder, counts)

def hard_imgs2char(input_folder, output_folder):
    """
    Detects the characters in the hard captcha images
    Each character is saved in a separate file in gray scale
    """
    # Get a list of all the captcha images we need to process
    images = glob.glob(os.path.join(input_folder, "*"))
    counts = {}

    for (i, img_path) in enumerate(images):
        print(f"INFO: processing image {i+1}/{len(images)}")

        # Extract name of the file since it contains the captcha characters
        filename = os.path.basename(img_path)
        img_text = os.path.splitext(filename)[0]

        img, gray = complex_thresh(img_path)

        char_bound_boxes = img2boxes(img, lambda w, h: (((w / h) > 1) and (w > 100)), num_chars = 6, cv2_chain = cv2.CHAIN_APPROX_NONE)

        if char_bound_boxes is None:
            continue

        for char_bound_box, char in zip(char_bound_boxes, img_text):
            counts = crop_and_save(char_bound_box, char, gray, output_folder, counts)

def simple_thresh(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return img, gray

def complex_thresh(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # threshold the image with a costum mask because it is the best way to process this specific images
    lower = np.array([220,220,220])
    upper = np.array([255,255,255])
    my_mask = cv2.inRange(img, lower, upper)
    img = cv2.threshold(my_mask, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return img, gray


def img2boxes(img, conjoined_condition, num_chars, cv2_chain):
    """
    Find the bounding boxes of the characters in the image
    """
    contours = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2_chain)
    contours = contours[1] if imutils.is_cv3() else contours[0]
    # print(contours)
    char_bound_boxes = []
    print(len(contours))
    for contour in contours:
        area = cv2.contourArea(contour)
        if (num_chars == 6) and  (area < 8.05):
            continue
        
        (x, y, w, h) = cv2.boundingRect(contour)

        if conjoined_condition(w, h):
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            char_bound_boxes.append((x, y, half_width, h))
            char_bound_boxes.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            char_bound_boxes.append((x, y, w, h))
    print(len(char_bound_boxes))
    if len(char_bound_boxes) != num_chars:
        return None

    char_bound_boxes = sorted(char_bound_boxes, key=lambda x: x[0])
    return char_bound_boxes

def find_chars(img_path, simple = True):

    if simple:

        img, gray = simple_thresh(img_path)

        char_regions = img2boxes(img, lambda w, h: w / h > 1.25, num_chars = 4, cv2_chain = cv2.CHAIN_APPROX_SIMPLE)
    else:

        img, gray = complex_thresh(img_path)

        char_regions = img2boxes(img, lambda w, h: (((w / h) > 1.35) and (w > 22)) or (h > 28), num_chars = 10, cv2_chain = cv2.CHAIN_APPROX_NONE)

    if char_regions is None:
        return None, None
    
    chars = []

    for char_region in char_regions:
        (x, y, w, h) = char_region
        
        char_img = gray[y-2:y+h+2, x-2:x+w+2]

        char_img = resize_to_fit(char_img, 20, 20)

        char_img = np.expand_dims(char_img, axis=-1)

        char_img = char_img / 255

        chars.append(char_img)

        gray = cv2.rectangle(gray, (x-2, y-2), (x+w+4, y+h+4), (0, 255, 0), 1)

    return gray, chars


def crop_and_save(char_bound_box, char, img, out_folder, counts):
    """
    Crop the character from the image and saves it as a new image
    """
    x, y, w, h = char_bound_box
    
    # get the letter from the original image with a 2 pixels margin around the edges
    char_img = img[y-2:y+h+2, x-2:x+w+2]

    save_path = os.path.join(out_folder, char)

    # if the output directory does not exist, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # write the letter image to a file
    count = counts.get(char, 1)
    p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
    cv2.imwrite(p, char_img)

    counts[char] = count + 1

    return counts

if __name__ == "__main__":
    captcha_image_path = 'generated_captchas/bd0SKD.png'  # Change this to your captcha image path
    # character_images = preprocess_captcha(captcha_image_path)
    
    # Save each character image
    # for i, char_img in enumerate(character_images):
    #     cv2.imwrite(f'preprocessed_images/character_{i}.png', char_img)
    
    #hard_imgs2char('generated_captchas', 'preprocessed_images')
    import cv2 
    import pytesseract 
    
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    

    #img = preprocess_captcha('generated_captchas/Capture.jpg')
    img = preprocess_captcha('generated_captchas/6sKX.jpeg') 
    #img = cv2.imread('generated_captchas/6sKX.jpeg')
    
    text = pytesseract.image_to_string(img)

    # Print the extracted text
    print("Extracted Text:", text)

    cv2.imshow("result", img) 
    cv2.waitKey(0)