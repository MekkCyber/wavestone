from dl_models import attacker_emnist

def get_attacker_from_ckpt_emnist(checkpoint_path=None) : 
    model = attacker_emnist.create_model()
    if checkpoint_path is None : 
        checkpoint_path = "checkpoints/attacker_emnist/training_13/best.weights.h5"
    model.load_weights(checkpoint_path)
    #model = tf.keras.models.load_model(checkpoint_path)
    #_, _, ds_test = attacker_emnist.get_dataset_keras()
    #ds_test = attacker_emnist.get_test_dataset_kaggle()
    #loss, acc = model.evaluate(ds_test, verbose=2)
    #print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    return model



def get_attacker_from_ckpt_python(checkpoint_path=None) : 
    model = attacker_emnist.create_model()
    if checkpoint_path is None : 
        checkpoint_path = "checkpoints/attacker_emnist_finetuned/training_2/best.weights.h5"
 
    model.load_weights(checkpoint_path)
    return model