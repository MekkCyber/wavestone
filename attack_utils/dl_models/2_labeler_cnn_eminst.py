import numpy as np
#
import pandas as pd
import tensorflow as tf
#
from sklearn.metrics import accuracy_score,f1_score,ConfusionMatrixDisplay,confusion_matrix
from keras.utils import to_categorical
#import np_utils ou from keras.utils import np_utils
import np_utils
import sklearn.metrics as metrics

import cv2 
import matplotlib.pyplot as plt
import plotly.graph_objects 
from plotly.subplots import make_subplots
import plotly.express as p
import seaborn as sns

import time
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from keras import layers
from keras.layers import *
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras import backend as K

from emnist import extract_training_samples
from emnist import extract_test_samples
from tensorflow.keras.models import load_model
# #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# # print("Number of training examples:", len(x_train))
# # print("Number of test examples:", len(x_test))
# # print("Image dimensions:", x_train[0].shape)
# # print("Number of classes:", len(np.unique(y_train)))

# #on normalise

# x_train=x_train/255
# x_test=x_test/255

# #array NumPy
# x_train = np.array(x_train).reshape(-1, 28, 28, 1)
# x_test =  np.array(x_test).reshape(-1, 28, 28, 1)
# print(x_train.shape)
# print(x_test.shape)

#encodage one-hot : permet de représenter chaque classe comme vecteur binaire distinct
#(0,0,1,0,...,0) par exemple est le vecteur binaire associé au label 3
# y_train = to_categorical(y_train)

TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.8

def get_dataset_keras(batch_size=128) : 
    
    (train_images_raw, train_labels_raw), (test_images, test_labels) = extract_training_samples('byclass'), extract_test_samples('byclass')

    train_images = train_images_raw[int(len(train_images_raw)*0.1):int(len(train_images_raw)*TRAIN_SPLIT)]
    train_labels = train_labels_raw[int(len(train_labels_raw)*0.1):int(len(train_labels_raw)*TRAIN_SPLIT)]

    val_images = train_images_raw[int(len(train_images_raw)*TRAIN_SPLIT):int(len(train_images_raw)*VAL_SPLIT)]
    val_labels = train_labels_raw[int(len(train_labels_raw)*TRAIN_SPLIT):int(len(train_labels_raw)*VAL_SPLIT)]

    #encodage one-hot : permet de représenter chaque classe comme vecteur binaire distinct
    #(0,0,1,0,...,0) par exemple est le vecteur binaire associé au label 3
    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)
    
    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size=10000)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    ds_val = ds_val.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.shuffle(buffer_size=10000)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.cache().batch(32).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test





#CNN process

#K.clear_session()

# Define the model using the functional approach
def create_model():
    cnn= Sequential()
    cnn.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='tanh', input_shape=(28, 28, 1)))
    cnn.add(MaxPooling2D(strides=2))
    cnn.add(Conv2D(filters=48, kernel_size=(5,5), padding='same', activation='tanh'))
    cnn.add(MaxPooling2D( strides=2))
    cnn.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='tanh'))
    cnn.add(Flatten())
    cnn.add(Dense(512, activation='tanh'))
    cnn.add(Dense(84, activation='tanh'))
    cnn.add(Dense(62, activation='softmax'))

    #optimazer
    opt=Adam(learning_rate=1e-4)

    cnn.compile(optimizer=opt,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    cnn.summary()
    return cnn

def get_model():
    return  tf.keras.models.load_model("checkpoints/labeler_cnn_emnist/training_2/cp-{epoch:02d}.ckpt")


#keras.utils.plot_model(cnn, "model.png", show_shapes=True)
def train(model_conv, ds_train, ds_val, batch_size=128, epochs=2):
    checkpoint_path = "checkpoints/labeler_cnn_emnist/training_2/cp-{epoch:02d}.ckpt"
    # keras_callbacks = [
    #     EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=0.001)
    # ]
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1)

    #enlever cllbacks=[cp_callback] pour ne plus overwright les poids une fois le modele entrainé
    model_conv.fit(
        ds_train,
        epochs = epochs,
        validation_data = ds_val,
        callbacks=[cp_callback]
    )

def visualisation(model, train, ds_test, ds_val, ds_train):
    # Visu


    x_test_list = []
    y_test_list = []

    # Itérer sur le dataset et extraire les valeurs
    for x, y in ds_test:
        x_test_list.append(x.numpy())  # Convertir le tenseur TensorFlow en tableau NumPy
        y_test_list.append(y.numpy())  # Convertir le tenseur TensorFlow en tableau NumPy

    # Convertir les listes en tableaux NumPy
    x_test = np.concatenate(x_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # Sélectionnez une image à partir de l'ensemble de test EMNIST
    image_index = 0  # Remplacez 0 par l'indice de l'image que vous souhaitez utiliser
    image = x_test[image_index]
    label = y_test[image_index]

    # Prétraitement de l'image pour qu'elle soit dans le bon format pour le modèle (déjà présent dans votre code)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    # Prédire le label de l'image en utilisant votre modèle
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    # Afficher l'image avec le label prédit
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Label réel : {label}, Label prédit : {predicted_label}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__" : 
    ds_train, ds_val, ds_test = get_dataset_keras()
    model = create_model()
    train = train(model, ds_train, ds_val)
    visualisation(model, train, ds_train, ds_val, ds_test)
