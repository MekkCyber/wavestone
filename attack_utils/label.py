from get_labeler_from_ckpt import get_labeler_from_ckpt
import tensorflow as tf



def label(data_captcha) : 
    print("### Loading Labeler ###")
    model = get_labeler_from_ckpt()
    labeled_data = model.predict(data_captcha)
    
    return tf.argmax(tf.nn.softmax(labeled_data, axis=-1), axis=-1)


