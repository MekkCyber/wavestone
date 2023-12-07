import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train[:40%]','train[40%:60%]', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.cache().batch(32).prefetch(tf.data.AUTOTUNE)

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

model_conv.fit(
    ds_train,
    epochs = 6,
    validation_data = ds_test
)

