from dl_models import attacker_emnist
import tensorflow as tf 
from metrics import output_metrics

def get_attacker_from_ckpt_emnist(checkpoint_path=None) : 
    model = attacker_emnist.create_model()
    if checkpoint_path is None : 
        checkpoint_path = "checkpoints/attacker_emnist/training_13/best.weights.h5"
    model.load_weights(checkpoint_path)
    
    #print model metrics
    output_metrics(1, model)

    return model



def get_attacker_from_ckpt_python(checkpoint_path=None) : 
    model = attacker_emnist.create_model()
    if checkpoint_path is None : 
        checkpoint_path = "checkpoints/attacker_emnist_finetuned/training_8/best.weights.h5"
 
    model.load_weights(checkpoint_path)
    return model