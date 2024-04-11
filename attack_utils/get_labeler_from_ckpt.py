from dl_models import labeler_cnn_mnist
import tensorflow as tf 

def get_labeler_from_ckpt_mnist(checkpoint_path=None) : 
    model = labeler_cnn_mnist.create_model()
    if checkpoint_path is None : 
        checkpoint_path = "checkpoints/labeler_mnist/training_3/best.weights.h5"

    model.load_weights(checkpoint_path)
    _, _, ds_test = labeler_cnn_mnist.get_dataset_keras()
    loss, acc = model.evaluate(ds_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    
    return model


if __name__ == "__main__" : 
    get_labeler_from_ckpt_mnist()