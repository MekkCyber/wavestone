import cv2
import numpy as np
import os 

def feature_extraction(I, num_chars=4):    
    I1 = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I2 = cv2.GaussianBlur(I1, (0, 0), 2)
    _, I3 = cv2.threshold(I2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #print(sum(I3))
    # EROSION
    # kernel = np.ones((8, 8), np.uint8)
    # I3 = cv2.erode(I3, kernel, iterations=1)
    
    # # REMOVE UNWANTED REGIONS
    # I3 = cv2.morphologyEx(I3, cv2.MORPH_CLOSE, kernel)
    # I3 = cv2.morphologyEx(I3, cv2.MORPH_OPEN, kernel)
    
    # DILATION
    # kernel = np.ones((6, 6), np.uint8)
    # I4 = cv2.dilate(I3, kernel, iterations=1)
    #cv2.imshow('Image', I3)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # Extract the 3 Regions (Digits)
    I3 = cv2.transpose(I3)
    # cv2.imshow('Transposed Image', I3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    characters_raw = []
    indexes_raw = []
    characters_processed = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(I3)
    for label in range(1, np.max(labels) + 1):
        mask = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 1000
        if cv2.contourArea(contours[0]) < min_area :
            continue
        x, y, w, h = cv2.boundingRect(contours[0])
        character = mask[y:y+h, x:x+w]
        #cv2.imwrite(f'character_{label}.jpg', character)
        indexes_raw.append(label)
        characters_raw.append(character)
    if len(characters_raw) == 1:
        h, w = stats[1][cv2.CC_STAT_HEIGHT], stats[1][cv2.CC_STAT_WIDTH]
        split_height = round(h / num_chars)
        for i in range(4):
            start_line = i * split_height
            end_line = (i + 1) * split_height
            split_region = characters_raw[0][start_line:end_line, :]
            characters_processed.append(split_region)
        
    elif len(characters_raw) == 2:
        h1, w1 = stats[indexes_raw[0]][cv2.CC_STAT_HEIGHT], stats[indexes_raw[0]][cv2.CC_STAT_WIDTH]
        h2, w2 = stats[indexes_raw[1]][cv2.CC_STAT_HEIGHT], stats[indexes_raw[1]][cv2.CC_STAT_WIDTH]

        if abs(h2-h1)<20:  # each image contains two characters
            characters_processed.append(characters_raw[0][round(h1/2), : ])
            characters_processed.append(characters_raw[0][:round(h1/2), : ])
            characters_processed.append(characters_raw[1][round(h2/2), : ])
            characters_processed.append(characters_raw[1][:round(h2/2), : ])
        elif h2 > h1 :  # h2 contains 3 
            characters_processed.append(characters_raw[0])
            split_height = round(h2 / 3)
            for i in range(3):
                start_line = i * split_height
                end_line = (i + 1) * split_height
                split_region = characters_raw[1][start_line:end_line, : ]
                characters_processed.append(split_region)
        elif h1 > h2 :  # h2 contains 3 
            split_height = round(h1 / 3)
            for i in range(3):
                start_line = i * split_height
                end_line = (i + 1) * split_height
                split_region = characters_raw[0][start_line:end_line, : ]
                characters_processed.append(split_region)
            characters_processed.append(characters_raw[1])
        
    elif len(characters_raw) == 3 : 
        max_index = np.argmax([stats[indexes_raw[j]][cv2.CC_STAT_HEIGHT] for j in range(len(characters_raw))])
        hi, wi = stats[indexes_raw[max_index]][cv2.CC_STAT_HEIGHT], stats[indexes_raw[max_index]][cv2.CC_STAT_WIDTH]
        split_height = round(hi / 2)
        for j in range(2) : 
            if j < max_index : 
                characters_processed.append(characters_raw[j])
        for i in range(2):
            start_line = i * split_height
            end_line = (i + 1) * split_height
            split_region = characters_raw[max_index][start_line:end_line, : ]
            characters_processed.append(split_region)
        for i in range(1,4) :
            if i > max_index+1 : 
                characters_processed.append(characters_raw[i-1])
    
    elif len(characters_raw) == 4:
        for i in range(4):
            characters_processed.append(characters_raw[i])
    for i,char in enumerate(characters_processed) : 
        characters_processed[i] = np.transpose(characters_processed[i])
    return characters_processed

def shape_feats(S):
    contours, _ = cv2.findContours(S.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    area = cv2.contourArea(max_contour)
    perimeter = cv2.arcLength(max_contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    centroid_x = int(M['m10'] / M['m00'])
    centroid_y = int(M['m01'] / M['m00'])
    orientation = 0.5 * np.arctan2((2 * M['mu11']), (M['mu20'] - M['mu02']))
    solidity = area / cv2.contourArea(cv2.convexHull(max_contour))
    return [circularity, area, centroid_x, centroid_y, np.degrees(orientation), solidity]

def feature_extraction_diagrams(image) : 
    I1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    I2 = cv2.GaussianBlur(I1, (0, 0), 2)
    _, I3 = cv2.threshold(I2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    w = len(I3[0])
    histogram = sum(I3)
    intervals = []
    latest_non_zero = 0
    for i in range(1, w) : 
        if histogram[i] == 0 and histogram[i-1] != 0 :
            intervals.append((latest_non_zero, i))
        if histogram[i] != 0 and histogram[i-1] == 0 : 
            latest_non_zero = i

# Load the image
def test_feature_extractor() :
    captchas = os.listdir("generated_captchas")
    i = 0
    # Call the function
    print(captchas)
    for captcha in captchas : 
        image = cv2.imread(f'generated_captchas/{captcha}')
        characters = feature_extraction(image)
        for index, character in enumerate(characters) : 
            resized_image = cv2.resize(character, (28, 28), interpolation=cv2.INTER_AREA)
            #resized_image = (resized_image/255. >= 0.5).astype(np.uint8)*255
            cv2.imwrite(f'characters/character_{i}.jpg', resized_image)
            i += 1

test_feature_extractor()
# def test(histogram) : 
#     intervals = []
#     for i in range(1, w) : 
#         if histogram[i] == 0 and histogram[i-1] != 0 :
#             intervals.append((latest_non_zero, i))
#         if histogram[i] != 0 and histogram[i-1] == 0 : 
#             latest_non_zero = i
#     return intervals

# I1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# I2 = cv2.GaussianBlur(I1, (0, 0), 2)
# _, I3 = cv2.threshold(I2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# w = len(I3[0])
# histogram = sum(I3)
# print(test(histogram))
        

def read_character() :
    for i in range(1) : 

        character = cv2.imread(f'characters/character_{i}.jpg', cv2.IMREAD_GRAYSCALE)
        #new_character = (character/255. >= 0.5).astype(np.uint8)*255
        # cv2.imshow('Transposed Image', new_character)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print(character)
#read_character()

# def read_eminst() : 
#     character = cv2.imread(f'tmp_emnist/a/6a9fd758-c75a-11ee-9e7d-e0d045d9590d.png')
#     print(character.shape)

# read_eminst()
