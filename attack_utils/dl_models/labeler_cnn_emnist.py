import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical

import matplotlib.pyplot as plt

from keras.layers import *

from emnist import extract_training_samples
from emnist import extract_test_samples

TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.8

def get_dataset_keras(batch_size=128) : 
    
    (train_images_raw, train_labels_raw), (test_images, test_labels) = extract_training_samples('byclass'), extract_test_samples('byclass')

    train_images = train_images_raw[int(len(train_images_raw)*0.1):int(len(train_images_raw)*TRAIN_SPLIT)]
    train_labels = train_labels_raw[int(len(train_labels_raw)*0.1):int(len(train_labels_raw)*TRAIN_SPLIT)]

    val_images = train_images_raw[int(len(train_images_raw)*TRAIN_SPLIT):int(len(train_images_raw)*VAL_SPLIT)]
    val_labels = train_labels_raw[int(len(train_labels_raw)*TRAIN_SPLIT):int(len(train_labels_raw)*VAL_SPLIT)]

    # train_labels = to_categorical(train_labels)
    # val_labels = to_categorical(val_labels)
    # test_labels = to_categorical(test_labels)
    
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

IMG_SIZE = 32

def create_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        Resizing(IMG_SIZE, IMG_SIZE),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    # model_conv = tf.keras.models.Sequential([
    #     data_augmentation,
    #     # tf.keras.layers.Conv2D(8, 3, padding="same", activation = "relu", input_shape=(28,28,1)),
    #     # tf.keras.layers.MaxPool2D(2,2),
    #     # tf.keras.layers.BatchNormalization(),
    #     # tf.keras.layers.Conv2D(16, 3, padding="same", activation = "relu",),
    #     # tf.keras.layers.MaxPool2D(2,2),
    #     # tf.keras.layers.BatchNormalization(),
    #     # tf.keras.layers.Conv2D(32, 3, padding="same", activation = "relu",),
    #     # tf.keras.layers.MaxPool2D(2,2),
    #     # tf.keras.layers.BatchNormalization(),
    #     # tf.keras.layers.Conv2D(64, 3, padding="same", activation = "relu",),
    #     # tf.keras.layers.MaxPool2D(2,2),
    #     # tf.keras.layers.BatchNormalization(),
    #     # tf.keras.layers.Flatten(),
    #     # tf.keras.layers.Dense(256, activation = "relu"),
    #     # tf.keras.layers.Dense(62)
    #     Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    #     Conv2D(64, (3, 3), padding='same', activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
    #     Conv2D(128, (3, 3), padding='same', activation='relu'),
    #     Conv2D(128, (3, 3), padding='same', activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
    #     Conv2D(256, (3, 3), padding='same', activation='relu'),
    #     Conv2D(256, (3, 3), padding='same', activation='relu'),
    #     Conv2D(256, (3, 3), padding='same', activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        
    #     Flatten(),
        
    #     Dense(512, activation='relu'),
    #     Dropout(0.5),
    #     Dense(512, activation='relu'),
    #     Dropout(0.5),
    #     Dense(62, activation='softmax')
        
    # ])

    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) 

    for layer in base_model.layers:
        layer.trainable = False
    model_conv = tf.keras.Sequential([
        data_augmentation,
        Conv2D(3, (3, 3), activation='relu', padding='same'),
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dense(62, activation='softmax')
    ])
    

    model_conv.compile(
        optimizer = tf.keras.optimizers.Adam(0.001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model_conv



# def get_model():
#     return  tf.keras.models.load_model("checkpoints/labeler_cnn_emnist/training_2/cp-{epoch:02d}.ckpt")


#keras.utils.plot_model(cnn, "model.png", show_shapes=True)
def train(model_conv, ds_train, ds_val, batch_size=128, epochs=10):
    checkpoint_path = "checkpoints/labeler_cnn_emnist/training_9/cp-{epoch:02d}.ckpt"
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
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
    train(model, ds_train, ds_val)
    #visualisation(model, train, ds_train, ds_val, ds_test)



