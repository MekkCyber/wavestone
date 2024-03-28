import string
from captcha.image import ImageCaptcha
import os
import random
import cv2
from feature_extractor import feature_extraction
from get_attacker_from_ckpt import get_attacker_from_ckpt_python
import numpy as np
from utils import convert_to_tfds, label_to_chr_emnist
import tensorflow as tf
import shutil
import time


def generate_captcha(captcha_length, width=500, height=150):
    characters = string.ascii_letters + string.digits
    #characters = 'abcdefghijklmnpqrtuvwxyzABCDEFGHJKLMNOPQRTUVWXYZ2346789'

    captcha_text = ''.join(random.choice(characters) for _ in range(captcha_length))  # You can adjust the length as needed

    captcha = ImageCaptcha(
                           font_sizes=(100,100),
                           width=width, 
                           height=height, 
                          )

    folder_path = os.path.join(os.getcwd(), 'generated_captcha_for_acc')

    captcha_image_file = os.path.join(folder_path, f'{captcha_text}.jpeg')
    captcha.write(captcha_text, captcha_image_file, 'jpeg')

def feature_extractor_and_labeler(model, captcha, print_in_test=True) :
    #captcha = os.listdir("generated_captcha_for_acc")[0]
    real_characters = list(captcha[:4])
    # folder_path = os.path.join(os.getcwd(),'characters_for_acc')
    # if os.path.exists(folder_path):
    #     shutil.rmtree(folder_path)
    # os.makedirs(folder_path)
    #i = 0
    image = cv2.imread(f'generated_captcha_for_acc/{captcha}')
    characters = feature_extraction(image)
    if len(characters)<4 : 
        return 2
    images = []
    for index, character in enumerate(characters) : 
        if character.size == 0 : 
            return 2
        resized_image = cv2.resize(character, (28, 28), interpolation=cv2.INTER_AREA)
        pixels = np.array(resized_image).reshape(28, 28, 1)
        images.append(pixels)
        #cv2.imwrite(f'characters_for_acc/character_{i}.jpg', resized_image)
        #i += 1
    images = convert_to_tfds(images)
    labeled_data = model.predict(images)
    result = label_to_chr_emnist(tf.argmax(tf.nn.softmax(labeled_data, axis=-1), axis=-1))
    if print_in_test :
        print(''.join([str(j).lower() for j in result]), ''.join([str(j).lower() for j in real_characters]))
    for i,char in enumerate(result) : 
        if str(char).lower() != str(real_characters[i]).lower() : 
            return 0
    return 1
    

def label(model, print_in_test=True) : 
    real_characters = []
    captcha = os.listdir('generated_captcha_for_acc')[0]
    real_characters = list(captcha[:4])
    images = []
    for i in range(4):
        
        character = cv2.imread(f'characters_for_acc/character_{i}.jpg', cv2.IMREAD_GRAYSCALE)
        #character = cv2.imread(f'tmp_emnist/b/{files[i]}', cv2.IMREAD_GRAYSCALE)
        pixels = np.array(character).reshape(28, 28, 1)
        images.append(pixels)
    
    images = convert_to_tfds(images)
    if len(images) == 0 : 
        return 2
    labeled_data = model.predict(images)
    result = label_to_chr_emnist(tf.argmax(tf.nn.softmax(labeled_data, axis=-1), axis=-1))
    if print_in_test :
        print(''.join([str(j).lower() for j in result]), ''.join([str(j).lower() for j in real_characters]))
    for i,char in enumerate(result) : 
        if str(char).lower() != str(real_characters[i]).lower() : 
            return 0
    return 1

# def data_generator():
#     indice = -1
#     captchas = os.listdir('generated_captcha_for_acc')
#     while True:
#         indice += 1
#         yield captchas[indice]


def test(n, print_in_test=True) : 
    model = get_attacker_from_ckpt_python()
    iterator = 0
    accuracy = 0
    matches = []
    folder_path = os.path.join(os.getcwd(), 'generated_captcha_for_acc')
    
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    
    for i in range(n) : 
        generate_captcha(4)
    captchas = os.listdir('generated_captcha_for_acc')
    for i in range(n) : 
        print(i)
        cracked = feature_extractor_and_labeler(model, captchas[i], print_in_test=True)
        # time.sleep(0.5)
        # cracked = label(model)
        # time.sleep(0.1)
        if cracked == 2 : 
            continue
        if cracked == 1: 
            matches.append(i)
            accuracy += 1
        iterator += 1
    if print_in_test : 
        print(iterator)
        print(accuracy)
        print(f"the captcha accuracy is : {accuracy/iterator*100:.2f}%")
        print(matches)
    gap = 0
    maximal_gap = 5
    for i in range(len(matches)-1) : 
        if matches[i+1]-matches[i] > maximal_gap : 
            gap += 1
    if print_in_test : 
        print(gap)
        print(f"The pourcentage of large gaps (>{maximal_gap}) between correct answers : {gap/len(matches)*100:.2f}%")
    return accuracy/iterator*100

test(200)
    
def low_loops(n, m) : 
    error = 0
    for i in range(n) : 
        acc = test(m, print_in_test=False)
        print(f"Model accuracy for iteration {i} using {m} captchas : {acc}")
        if acc < 100/m - 1 : 
            error += 1
    print(f"number of times the model didnt find any correct answer : {error}")

#low_loops(200,5)