from dl_models import labeler_cnn_mnist
import tensorflow as tf 
from metrics import output_metrics


def get_labeler_from_ckpt_mnist(checkpoint_path=None) : 
    model = labeler_cnn_mnist.create_model()
    if checkpoint_path is None : 
        checkpoint_path = "checkpoints/labeler_mnist/training_3/best.weights.h5"
    model.load_weights(checkpoint_path, skip_mismatch=True)

    #print model metrics
    # output_metrics(0, model)
    
    return model

if __name__ == "__main__" : 
    get_labeler_from_ckpt_mnist()
