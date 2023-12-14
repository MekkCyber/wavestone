from dl_models import labeler_cnn

def get_labeler_from_ckpt(checkpoint_path=None) : 
    model = labeler_cnn.create_model()
    if checkpoint_path is None : 
        checkpoint_path = "checkpoints/labeler/training_2/cp-08.ckpt"

    model.load_weights(checkpoint_path).expect_partial()
    _, _, ds_test = labeler_cnn.get_dataset_keras()
    loss, acc = model.evaluate(ds_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    return model


if __name__ == "__main__" : 
    get_labeler_from_ckpt()