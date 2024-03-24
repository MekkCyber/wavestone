from get_labeler_from_ckpt import get_labeler_from_ckpt_emnist
import tensorflow as tf
from keras.callbacks import *
from utils import convert_to_tfds, chr_to_label_emnist
from feature_extractor import feature_extraction
from PIL import Image
import numpy as np
import cv2
import os
import string
import random
from captcha.image import ImageCaptcha
import shutil

def generate_captcha(captcha_length, width=500, height=150):
    characters = string.ascii_letters + string.digits
    characters = 'abcdefghijklmnpqrtuvwxyzABCDEFGHJKLMNOPQRTUVWXYZ2346789'

    captcha_text = ''.join(random.choice(characters) for _ in range(captcha_length))  # You can adjust the length as needed

    captcha = ImageCaptcha(
                           font_sizes=(100,100),
                           width=width, 
                           height=height, 
                          )

    folder_path = os.path.join(os.getcwd(), 'captchas_for_finetuning')
    
    captcha_image_file = os.path.join(folder_path, f'{captcha_text}.jpeg')
    captcha.write(captcha_text, captcha_image_file, 'jpeg')

def finetune_emnist(num_captchas=100, epochs=100) :
    folder_path = os.path.join(os.getcwd(), 'captchas_for_finetuning')
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path) 
    
    for _ in range(num_captchas) : 
        generate_captcha(4)
    
    labels = []
    captchas = os.listdir('captchas_for_finetuning')
        
    model = get_labeler_from_ckpt_emnist()
    images = []
    for i in range(len(captchas)):
        image = cv2.imread(f'captchas_for_finetuning/{captchas[i]}')
        characters = feature_extraction(image)
        if len(characters) < 4 : 
            continue
        for index, character in enumerate(characters) : 
            # if character.size == 0 : 
            #     break
            resized_image = cv2.resize(character, (28, 28), interpolation=cv2.INTER_AREA)
            pixels = np.array(resized_image).reshape(28, 28, 1)
            images.append(pixels)
        labels.extend(chr_to_label_emnist(captchas[i][:4]))
    print("images : ", len(images))
    print("labels : ", len(labels))
    images = convert_to_tfds(images, labels=labels)
    ################### Finetuing ########################
    checkpoint_path = "checkpoints/labeler_cnn_emnist_finetuned/training_5/best.weights.h5"
    
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    save_best_only=True,
                                                    monitor='sparse_categorical_accuracy',
                                                    mode='max')
    ES = EarlyStopping(monitor='sparse_categorical_accuracy',min_delta=0,verbose=0,restore_best_weights = True,patience=5,mode='max')
    RLP = ReduceLROnPlateau(monitor='loss',patience=5,factor=0.2,min_lr=0.0001)

    model.fit(
        images,
        epochs = epochs,
        callbacks=[cp_callback, ES, RLP]
    )
    ######################################################


finetune_emnist()