import string
from captcha.image import ImageCaptcha
import os
import random
import cv2
from feature_extractor import feature_extraction
from get_labeler_from_ckpt import get_labeler_from_ckpt_emnist
import numpy as np
from utils import convert_to_tfds, label_to_chr_emnist
import tensorflow as tf
import shutil
import time


def generate_captcha(captcha_length, width=500, height=150):
    characters = string.ascii_letters + string.digits
    characters = 'abcdefghijklmnpqrtuvwxyzABCDEFGHJKLMNOPQRTUVWXYZ2346789'

    captcha_text = ''.join(random.choice(characters) for _ in range(captcha_length))  # You can adjust the length as needed

    captcha = ImageCaptcha(
                           font_sizes=(100,100),
                           width=width, 
                           height=height, 
                          )

    folder_path = os.path.join(os.getcwd(), 'generated_captcha_for_acc')

    captcha_image_file = os.path.join(folder_path, f'{captcha_text}.jpeg')
    captcha.write(captcha_text, captcha_image_file, 'jpeg')

def feature_extractor_and_labeler(model, captcha) :
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
    print(''.join([str(j).lower() for j in result]), ''.join([str(j).lower() for j in real_characters]))
    for i,char in enumerate(result) : 
        if str(char).lower() != str(real_characters[i]).lower() : 
            return 0
    return 1
    

def label(model) : 
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


def test(n) : 
    model = get_labeler_from_ckpt_emnist()
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
    for i in range(n-10) : 
        print(i)
        cracked = feature_extractor_and_labeler(model, captchas[i])
        # time.sleep(0.5)
        # cracked = label(model)
        # time.sleep(0.1)
        if cracked == 2 : 
            continue
        if cracked == 1: 
            matches.append(i)
            accuracy += 1
        iterator += 1
    print(iterator)
    print(accuracy)
    print(f"the captcha accuracy is {accuracy/iterator}")
    print(matches)
    gap = 0
    for i in range(len(matches)-1) : 
        if matches[i+1]-matches[i] > 5 : 
            gap += 1
    print(gap)

test(1000)
    