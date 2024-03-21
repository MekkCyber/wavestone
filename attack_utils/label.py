from get_labeler_from_ckpt import get_labeler_from_ckpt_mnist, get_labeler_from_ckpt_emnist
import tensorflow as tf
from utils import convert_to_tfds, label_to_chr_emnist
from PIL import Image
import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def label_mnist(data_captcha) : 
    print("### Loading Labeler for MNIST ###")
    model = get_labeler_from_ckpt_mnist()
    labeled_data = model.predict(data_captcha)
    
    return tf.argmax(tf.nn.softmax(labeled_data, axis=-1), axis=-1)


def label_emnist(data_captcha) : 
    print("### Loading Labeler for EMNIST ###")
    files = os.listdir('tmp_emnist/b/')
    model = get_labeler_from_ckpt_emnist()
    images = []
    for i in range (20):
        character = cv2.imread(f'characters/character_{i}.jpg', cv2.IMREAD_GRAYSCALE)
        #character = cv2.imread(f'tmp_emnist/b/{files[i]}', cv2.IMREAD_GRAYSCALE)
        print(character.shape)
        pixels = np.array(character).reshape(28, 28, 1)
        images.append(pixels)
    images = convert_to_tfds(images)
    labeled_data = model.predict(images)

    return label_to_chr_emnist(tf.argmax(tf.nn.softmax(labeled_data, axis=-1), axis=-1)), tf.argmax(tf.nn.softmax(labeled_data, axis=-1), axis=-1)


result = label_emnist(2)
print(result[0])
print(result[1])