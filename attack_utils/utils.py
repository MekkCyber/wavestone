import tensorflow as tf

def convert_to_tfds(data, labels=None, batch_size=4) :
    if labels is None : 
        def normalize_img(image):
            return tf.cast(image, tf.float32) / 255.
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    else : 
        def normalize_img(image, label):
            return tf.cast(image, tf.float32) / 255., label
        ds = tf.data.Dataset.from_tensor_slices((data, labels))
        ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def label_to_chr_emnist(arr) : 
    result = []
    for elt in arr : 
        if elt <= 9 : 
            result.append(elt.numpy())
        elif elt < 36 : 
            result.append(chr(elt - 10 + ord('A')))
        else : 
            result.append(chr(elt - 36 + ord('a')))
    return result

def chr_to_label_emnist(chars):
    result = []
    for char in chars:
        if char.isdigit():
            result.append(int(char))
        elif char.isupper():
            result.append(ord(char) - ord('A') + 10)
        elif char.islower():
            result.append(ord(char) - ord('a') + 36)
    return result

def normalize_and_expand_img(image, label):
        if len(image.shape) == 2:
            image = tf.expand_dims(image, axis=-1)
        return tf.cast(image, tf.float32) / 255., label