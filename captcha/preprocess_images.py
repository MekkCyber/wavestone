import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib

data_dir = os.path.join("captcha","images_dirs")

# train_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=10)
data_dir = pathlib.Path(data_dir).with_suffix('')
images = list(data_dir.glob('*.png'))

img_height = 128
img_width = 128

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  subset="training",
  validation_split = 0.2,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=10)

for image in train_ds.take(1) : 
    print(image[0].shape)
    break