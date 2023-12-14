import tensorflow as tf

def convert_to_tfds(data, labels=None, batch_size=8) :
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