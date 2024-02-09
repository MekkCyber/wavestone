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

from emnist import list_datasets
list_datasets()

#I extract training samples from the EMNIST dataset.
#The 'balanced' argument indicates the extraction of a balanced subset of data.
#It includes an equal number of samples for each character class.

from emnist import extract_training_samples
x_train, y_train = extract_training_samples('balanced')
from emnist import extract_test_samples
x_test, y_test = extract_test_samples('balanced')
#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# print("Number of training examples:", len(x_train))
# print("Number of test examples:", len(x_test))
# print("Image dimensions:", x_train[0].shape)
# print("Number of classes:", len(np.unique(y_train)))

#on normalise

x_train=x_train/255
x_test=x_test/255

#array NumPy
x_train = np.array(x_train).reshape(-1, 28, 28, 1)
x_test =  np.array(x_test).reshape(-1, 28, 28, 1)
print(x_train.shape)
print(x_test.shape)

#encodage one-hot : permet de représenter chaque classe comme vecteur binaire distinct
#(0,0,1,0,...,0) par exemple est le vecteur binaire associé au label 3
y_train = to_categorical(y_train)


#CNN process

#K.clear_session()

# Define the model using the functional approach
cnn= Sequential()
cnn.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='tanh', input_shape=(28, 28, 1)))
cnn.add(MaxPooling2D(strides=2))
cnn.add(Conv2D(filters=48, kernel_size=(5,5), padding='same', activation='tanh'))
cnn.add(MaxPooling2D( strides=2))
cnn.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='tanh'))
cnn.add(Flatten())
cnn.add(Dense(512, activation='tanh'))
cnn.add(Dense(84, activation='tanh'))
cnn.add(Dense(47, activation='softmax'))

#optimazer
opt=Adam(learning_rate=1e-4)

#Let's compile the model before training.
cnn.compile(optimizer=opt,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

cnn.summary()

#keras.utils.plot_model(cnn, "model.png", show_shapes=True)

keras_callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=0.001)
]

start_time = time.time() 
history = cnn.fit(x_train, y_train, epochs=1, batch_size=64, verbose=1,
                    validation_split=0.2, callbacks=keras_callbacks)

end_time = time.time() 

#visualisation résultats

# cols = 8
# rows = 4

# plt.figure(figsize=(cols, rows))

# for image in x_test[1] :
#     for label in y_test[1] :
#         preds = model_conv.predict(image)
#         preds = tf.nn.softmax(preds, axis=-1)
#         for i in range(32) :
#             plt.subplot(rows, cols, i + 1)
#             plt.title(f"{label[i].numpy()}-{np.argmax(preds[i])}")
#             plt.imshow(image[i])
#             plt.axis('off')
#     break