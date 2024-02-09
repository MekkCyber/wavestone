import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os 

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


def create_model() : 
    model_conv = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, 3, padding="same", activation = "relu", input_shape=(28,28,1)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation = "relu",),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation = "relu",),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation = "relu",),
        tf.keras.layers.Dense(10)
    ])


    model_conv.compile(
        optimizer = tf.keras.optimizers.Adam(0.001),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return model_conv



def train(model_conv, ds_train, ds_val=None, epochs=20) : 
    checkpoint_path = "checkpoints/attacker/training_1/cp-{epoch:02d}.ckpt"

    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                 save_weights_only=True,
    #                                                 verbose=1)
    model_conv.fit(
        ds_train,
        epochs = epochs,
    )

    return model_conv
