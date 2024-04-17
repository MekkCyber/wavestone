import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import numpy as np
import os 

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.8

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



def get_dataset_keras(batch_size=128) : 
    mnist = tf.keras.datasets.mnist
    (train_images_raw, train_labels_raw), (test_images, test_labels) = mnist.load_data()

    train_images = train_images_raw[int(len(train_images_raw)*0.1):int(len(train_images_raw)*TRAIN_SPLIT)]
    train_labels = train_labels_raw[int(len(train_labels_raw)*0.1):int(len(train_labels_raw)*TRAIN_SPLIT)]

    val_images = train_images_raw[int(len(train_images_raw)*TRAIN_SPLIT):int(len(train_images_raw)*VAL_SPLIT)]
    val_labels = train_labels_raw[int(len(train_labels_raw)*TRAIN_SPLIT):int(len(train_labels_raw)*VAL_SPLIT)]
    
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







def create_model(lr=0.001) : 
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
        optimizer = tf.keras.optimizers.Adam(lr),
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return model_conv



def train(model_conv, ds_train, ds_val, batch_size=128, epochs=20) : 
    checkpoint_path = "checkpoints/labeler_mnist/training_3/best.weights.h5"
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
  ds_train, ds_val, ds_test = get_dataset_keras()
  model = create_model()
  train(model, ds_train, ds_val)