import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import uuid
import cv2

(ds_train), ds_info = tfds.load(
    'mnist',
    split='train[10%]',
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

for i in range(10) : 
    os.mkdir(os.path.join("attack_utils", "images_dirs",str(i)))

for index, (image, label)in enumerate(ds_train.shuffle(100)) : 
    fig = plt.figure()
    #plt.imshow(image)
    if index == 1000 : 
        break
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)
    plt.savefig(os.path.join("attack_utils","images_dirs",str(label.numpy()),str(uuid.uuid1())))
    plt.close()
