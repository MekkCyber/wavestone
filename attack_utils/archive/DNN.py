import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

#choix database
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

#on normalise
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

#modèle

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28,1)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)
#affichage


cols = 8
rows = 4

plt.figure(figsize=(cols, rows))

for image, label in ds_test.take(1) :
  preds = model.predict(image)
  preds = tf.nn.softmax(preds, axis=-1)
  for i in range(32) :
    plt.subplot(rows, cols, i + 1)
    plt.title(f"{label[i].numpy()}-{np.argmax(preds[i])}")
    plt.imshow(image[i])
    plt.axis('off')
  break