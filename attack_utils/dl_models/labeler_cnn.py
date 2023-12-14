import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os 

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

#choix base de données
def get_dataset_tf(batch_size=128) : 
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train[10%:70%]','train[70%:]', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.cache().batch(32).prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_val, ds_test

def get_dataset(batch_size=128) : 
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train[10%:70%]','train[70%:]', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.cache().batch(32).prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_val, ds_test





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



def train(model_conv, ds_train, ds_val, batch_size=128, epochs=20) : 
    checkpoint_path = "checkpoints/labeler/training_1/cp-{epoch:02d}.ckpt"
    n_batches = len(ds_train) // batch_size

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1)

    model_conv.fit(
        ds_train,
        epochs = epochs,
        validation_data = ds_val,
        callbacks=[cp_callback]
    )


if __name__ == "__main__" : 
  ds_train, ds_val, ds_test = get_dataset()
  model = create_model()
  train(model, ds_train, ds_val)