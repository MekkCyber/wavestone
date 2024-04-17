import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical

import matplotlib.pyplot as plt

from keras.layers import *

from emnist import extract_training_samples
from emnist import extract_test_samples

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.callbacks import ModelCheckpoint

TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.8

def get_model(batch_size=128):

    def extract_data(data_dir):
        images = []
        labels = []
        # Parcourir les sous-répertoires
        for category in os.listdir(data_dir):
            category_dir = os.path.join(data_dir, category)
            if os.path.isdir(category_dir):
                # Parcourir les fichiers d'images dans chaque sous-répertoire
                for file_name in os.listdir(category_dir):
                    # Charger l'image
                    image = cv2.imread(os.path.join(category_dir, file_name))
                    images.append(image)
                    labels.append(category)
        return images, labels

    def split_data(images, labels):
        # Diviser les données en ensembles d'entraînement, de validation et de test
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2 => 20% de la base de données pour le validation set

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    def get_dataset(data_dir, batch_size=128):
        # Extraction des données
        images, labels = extract_data(data_dir)
        # Division des données
        train_images, train_labels, val_images, val_labels, test_images, test_labels = split_data(images, labels)

        return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

    # Utilisation de la fonction pour obtenir les ensembles d'entraînement, de validation et de test
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = get_dataset('../train_data/Large')

    #visu
    image = test_images[35]
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    print((test_labels[35]))

    def normalize_img(image, label):
        return tf.cast(image, tf.float32)/255., label

    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    train_labels_encoded2 = to_categorical(train_labels_encoded)

    val_labels_encoded = label_encoder.fit_transform(val_labels)
    val_labels_encoded2 = to_categorical(val_labels_encoded)

    test_labels_encoded = label_encoder.fit_transform(test_labels)
    test_labels_encoded2 = to_categorical(test_labels_encoded)

    new_size = (120, 120) 
    train_images_resized = [tf.image.resize(image, new_size) for image in train_images]
    val_images_resized = [tf.image.resize(image, new_size) for image in val_images]
    test_images_resized = [tf.image.resize(image, new_size) for image in test_images]

    ds_train = tf.data.Dataset.from_tensor_slices((train_images_resized, train_labels_encoded2))
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size=10000)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((val_images_resized, val_labels_encoded2))
    ds_val = ds_val.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.shuffle(buffer_size=10000)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((test_images_resized,test_labels_encoded2))
    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.cache().batch(32).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test


def create_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(120,120,3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        #applatir l'image avant le dense
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(12, activation='softmax'),
    ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(0.001),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ['accuracy']
    )
    
    return model

def train(model, ds_train, ds_val, batch_size=128, epochs=20) : 
    checkpoint_path = "checkpoints/labeler/training_ReCaptcha2/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1)

    model.fit(
        ds_train,
        epochs = epochs,
        validation_data = ds_val,
        callbacks=[cp_callback]
    )

def visualisation(model, ds_test, ds_val, ds_train, n):
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

    image_index = n 
    image = x_test[image_index]
    label = y_test[image_index]

    # Prétraitement de l'image pour qu'elle soit dans le bon format pour le modèle (déjà présent dans votre code)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    # Prédire le label de l'image
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Label réel : {label}, Label prédit : {predicted_label}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__" : 
    ds_train, ds_val, ds_test = get_model()
    model = create_model()
    model.load_weights("checkpoints/labeler/training_ReCaptcha2/cp.ckpt")
    # train(model, ds_train, ds_val)
    visualisation(model, ds_test, ds_val, ds_train, 17)

