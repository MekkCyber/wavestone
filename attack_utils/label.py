from get_labeler_from_ckpt import get_labeler_from_ckpt_mnist, get_labeler_from_ckpt_emnist
import tensorflow as tf
from utils import convert_to_tfds, label_to_chr_emnist
from PIL import Image
import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def label_mnist(data_captcha) : 
    print("### Loading Labeler for MNIST ###")
    model = get_labeler_from_ckpt_mnist()
    labeled_data = model.predict(data_captcha)
    
    return tf.argmax(tf.nn.softmax(labeled_data, axis=-1), axis=-1)


def label_emnist(data_captcha, num=20) : 
    print("### Loading Labeler for EMNIST ###")
    real_characters = []
    captchas = os.listdir('generated_captchas')
    for captcha in captchas : 
        real_characters.extend(list(captcha[:4]))
    model = get_labeler_from_ckpt_emnist()
    images = []
    accuracy = 0
    for i in range(num):
        character = cv2.imread(f'characters/character_{i}.jpg', cv2.IMREAD_GRAYSCALE)
        #character = cv2.imread(f'tmp_emnist/b/{files[i]}', cv2.IMREAD_GRAYSCALE)
        pixels = np.array(character).reshape(28, 28, 1)
        images.append(pixels)
    images = convert_to_tfds(images)
    labeled_data = model.predict(images)
    result = label_to_chr_emnist(tf.argmax(tf.nn.softmax(labeled_data, axis=-1), axis=-1))
    for i,char in enumerate(result) : 
        if str(char).lower() == str(real_characters[i]).lower() : 
            accuracy += 1
    print(f"The model accuracy on python generated captchas is : {accuracy/num*100}%")
    return result, tf.argmax(tf.nn.softmax(labeled_data, axis=-1), axis=-1)


# result = label_emnist(2, num=120)
# print(result[0])
# print(result[1])