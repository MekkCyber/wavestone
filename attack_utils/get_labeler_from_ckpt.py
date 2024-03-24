from dl_models import labeler_cnn_mnist
from dl_models import labeler_cnn_emnist
import tensorflow as tf 

def get_labeler_from_ckpt_mnist(checkpoint_path=None) : 
    model = labeler_cnn_mnist.create_model()
    if checkpoint_path is None : 
        checkpoint_path = "checkpoints/labeler/training_2/cp-08.ckpt"

    model.load_weights(checkpoint_path).expect_partial()
    _, _, ds_test = labeler_cnn_mnist.get_dataset_keras()
    loss, acc = model.evaluate(ds_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    return model

def get_labeler_from_ckpt_emnist(checkpoint_path=None) : 
    model = labeler_cnn_emnist.create_model()
    if checkpoint_path is None : 
        checkpoint_path = "checkpoints/labeler_cnn_emnist_finetuned/training_5/best.weights.h5"
    
    model.load_weights(checkpoint_path)
    #model = tf.keras.models.load_model(checkpoint_path)
    _, _, ds_test = labeler_cnn_emnist.get_dataset_keras()
    #ds_test = labeler_cnn_emnist.get_test_dataset_kaggle()
    #loss, acc = model.evaluate(ds_test, verbose=2)
    #print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    return model


if __name__ == "__main__" : 
    get_labeler_from_ckpt_mnist()