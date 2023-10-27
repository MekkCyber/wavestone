import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import uuid

(ds_train), ds_info = tfds.load(
    'mnist',
    split='train',
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

for index, (image, label)in enumerate(ds_train.shuffle(100)) : 
    fig = plt.figure()
    plt.imshow(image)
    if index == 1000 : 
        break
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)
    plt.savefig(os.path.join("captcha","images",str(label.numpy())+'_'+str(uuid.uuid1())))
    plt.close()
