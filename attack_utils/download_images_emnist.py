import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import uuid
from PIL import Image
import string
# (ds_train), ds_info = tfds.load(
#     'mnist',
#     split='train[:10%]',
#     shuffle_files=True,
#     as_supervised=True,
#     with_info=True,
# )

# for i in range(10) : 
#     os.mkdir(os.path.join("tmp_emnist",str(i)))

# characters = string.ascii_letters

# for i in characters : 
#     os.mkdir(os.path.join("tmp_emnist",i))

from emnist import extract_training_samples
x_train, y_train = extract_training_samples('byclass')

x_train = x_train[:int(len(x_train)*0.01)]
y_train = y_train[:int(len(y_train)*0.01)]

print(x_train.shape)
# for char in y_train[:100] : 
#     print(char)
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

for index, (image, label) in enumerate(zip(x_train, y_train)) : 
    if index == 1000 : 
        break
    image_ = Image.fromarray(image)
    if label < 10 :   
        image_.save(os.path.join("tmp_emnist",str(label),str(uuid.uuid1())+".png"))
    elif label < 36 : 
        label = (label-10) + ord('a')
        image_.save(os.path.join("tmp_emnist",chr(label),str(uuid.uuid1())+".png"))
    else : 
        label = (label-36) + ord('a')
        image_.save(os.path.join("tmp_emnist",chr(label),str(uuid.uuid1())+".png"))
    
    
