import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import uuid
from PIL import Image

# (ds_train), ds_info = tfds.load(
#     'mnist',
#     split='train[:10%]',
#     shuffle_files=True,
#     as_supervised=True,
#     with_info=True,
# )

for i in range(10) : 
    os.mkdir(os.path.join("attack_utils", "images_dirs",str(i)))

mnist = tf.keras.datasets.mnist
(train_images, train_labels), _ = mnist.load_data()

train_images = train_images[:int(len(train_images)*0.1)]
train_labels = train_labels[:int(len(train_labels)*0.1)]


# for index, (image, label)in enumerate(ds_train.take(1)) :
#     print(image)
#     image_ = Image.fromarray(image)
#     image_.save('mnist_example.png')

# for index, (image, label)in enumerate(ds_train.shuffle(100)) : 
#     fig = plt.figure()
#     #plt.imshow(image)
#     if index == 1000 : 
#         break
#     plt.axis('off')
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     plt.margins(0, 0)
#     plt.savefig(os.path.join("attack_utils","images_dirs",str(label.numpy()),str(uuid.uuid1())))
#     plt.close()

for index, (image, label) in enumerate(zip(train_images, train_labels)) : 
    if index == 1000 : 
        break
    image_ = Image.fromarray(image)
    image_.save(os.path.join("attack_utils","images_dirs",str(label),str(uuid.uuid1())+".png"))
