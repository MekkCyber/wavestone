import os
import PIL
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from dl_models import labeler_cnn_mnist





def output_metrics(CaptchaType, model):
    """Print loaded model metrics to the screen depending of the CaptchaType."""

    # Model MNIST
    if CaptchaType == 0:
        # Print model summary
        print("Model Summary for MNIST labeler :")
        model.summary()

        # Load MNIST test dataset
        _, _, ds_test = labeler_cnn_mnist.get_dataset_keras()

        # Evaluate the model on the test dataset
        loss, accuracy = model.evaluate(ds_test, verbose=2)

        # Print evaluation metrics
        print("\nEvaluation Metrics:")
        print("Loss: {:.4f}".format(loss))
        print("Accuracy: {:.2f}%".format(accuracy * 100))

        # Additional Information
        print("\nAdditional Information:")
        print("Number of trainable parameters: {}".format(
            sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])))
        print("Number of non-trainable parameters: {}".format(
            sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])))
        print("Input shape: {}".format(model.input_shape))
        print("Output shape: {}".format(model.output_shape))
        print("Optimizer: {}".format(model.optimizer.get_config()))
        print("Learning rate: {}".format(model.optimizer.learning_rate.numpy()))
    
        # print("Training configuration: {}".format(model.optimizer.get_config()))
    
    # Model EMNIST
    elif CaptchaType == 1:
        # Print model summary
        print("Model Summary:")
        model.summary()

        # Generating custom EMNIST test dataset
        # Directory containing the EMNIST images organized by labels
        data_dir = os.getcwd() + '/images_dirs/'

        batch_size = 2
        img_height = 28
        img_width = 28


        # Create a TensorFlow dataset
        ds_emnist = tf.keras.utils.image_dataset_from_directory(data_dir,color_mode='grayscale',seed=123,image_size=(img_height, img_width),batch_size=batch_size)


        # Set up model and evaluate similar to MNIST
        loss, accuracy = model.evaluate(ds_emnist, verbose=2)




        #model = tf.keras.models.load_model(checkpoint_path)
        # _, _, ds_test = attacker_emnist.get_dataset_keras()
        #ds_test = attacker_emnist.get_test_dataset_kaggle()
        # loss, acc = model.evaluate(ds_test, verbose=2)
        # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    
    #Model PYTHON
    elif CaptchaType == 2:
        pass
    
    #Model ATTACKER for MNIST
    elif CaptchaType == 3:
        # Print model summary
        print("Model Summary for MNIST Attacker :")
        model.summary()

        # Load MNIST test dataset
        _, _, ds_test = labeler_cnn_mnist.get_dataset_keras()

        # Evaluate the model on the test dataset
        loss, accuracy = model.evaluate(ds_test, verbose=2)

        # Print evaluation metrics
        print("\nEvaluation Metrics:")
        print("Loss: {:.4f}".format(loss))
        print("Accuracy: {:.2f}%".format(accuracy * 100))

        # Additional Information
        print("\nAdditional Information:")
        print("Number of trainable parameters: {}".format(
            sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])))
        print("Number of non-trainable parameters: {}".format(
            sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])))
        print("Input shape: {}".format(model.input_shape))
        print("Output shape: {}".format(model.output_shape))
        print("Optimizer: {}".format(model.optimizer.get_config()))
        print("Learning rate: {}".format(model.optimizer.learning_rate.numpy()))





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


