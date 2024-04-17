import os
import PIL
import PIL.Image
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from dl_models import labeler_cnn_mnist
from dl_models import attacker_emnist
import shutil
import string
import random
import cv2
from feature_extractor import feature_extraction
from utils import convert_to_tfds
from captcha.image import ImageCaptcha
from utils import chr_to_label_emnist

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
        evaluate_with_metrics(model, ds_test)
        

        
        # Additional Information
        print("\nAdditional Information:")

        print("Number of parameters: {}".format(model.count_params()))
        print("Number of trainable parameters: {}".format(
            sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])))
        print("Number of non-trainable parameters: {}".format(
            sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])))
        print("Input shape: {}".format(model.input_shape))
        print("Output shape: {}".format(model.output_shape))
        print("Optimizer: {}".format(model.optimizer.get_config()))
        print("Learning rate: {}".format(model.optimizer.learning_rate.numpy()))

        print("Number of layers: {}".format(len(model.layers)))
        print("Number of weights in the model: {}".format(len(model.weights)))
        
        
    
    # Model EMNIST
    elif CaptchaType == 1:
        # Print model summary
        print("Model Summary:")
        model.summary()

        # Generating custom EMNIST test dataset
        # Directory containing the EMNIST images organized by labels
        # data_dir = os.getcwd() + '/images_dirs/'

        # Create a TensorFlow dataset
       
        _, _, ds_test = attacker_emnist.get_dataset_keras()

        evaluate_with_metrics(model, ds_test)
        additional_informations(model)
    
    #Model PYTHON
    elif CaptchaType == 2:
        print("Model Summary:")
        model.summary()
       
        evaluate_with_metrics_python(model, 100)
        additional_informations(model)
        
    
    #Model ATTACKER for MNIST
    elif CaptchaType == 3:
        # Print model summary
        print("Model Summary for MNIST Attacker :")
        model.summary()

        # Load MNIST test dataset
        _, _, ds_test = labeler_cnn_mnist.get_dataset_keras()

        evaluate_with_metrics(model, ds_test)
        additional_informations(model)


    elif CaptchaType == 4:
        print("Model Summary:")
        model.summary()
       
        evaluate_with_metrics_python(model, 100)
        additional_informations(model)
        






def additional_informations(model):
    # Additional Information
    print("\nAdditional Information:")
    print("Number of parameters: {}".format(model.count_params()))
    print("Number of trainable parameters: {}".format(
        sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])))
    print("Number of non-trainable parameters: {}".format(
        sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])))
    print("Input shape: {}".format(model.input_shape))
    print("Output shape: {}".format(model.output_shape))
    print("Optimizer: {}".format(model.optimizer.get_config()))
    print("Learning rate: {}".format(model.optimizer.learning_rate.numpy()))
    
    print("Number of layers: {}".format(len(model.layers)))
    print("Number of weights in the model: {}".format(len(model.weights)))



def evaluate_with_metrics(model, ds_test=None, y_true=None, y_pred=None, predictions=None):
    # Evaluate the model
    num_classes = get_num_classes_from_model(model)
    
    # Predict on test data
    if ds_test : 
        predictions = model.predict(ds_test)
    if y_true==None : 
        y_true = np.concatenate([y for x, y in ds_test], axis=0)
    if y_pred==None : 
        y_pred = np.argmax(predictions, axis=1)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    
    # Compute false positive and false negative rates
    fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
    fn = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    tp = np.diag(conf_matrix)
    tn = conf_matrix.sum() - (fp + fn + tp)
    
    loss = log_loss(y_true, predictions)
    accuracy = (tp + tn)/(tp + fp + tn + fn)
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)
    precision = sum(tp) / (sum(tp) + sum(fp))
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
        
    # Print evaluation metrics
    print("\nEvaluation Metrics:")

    # TO SHOW ACCURACY = PRECISION
    # loss, acc = model.evaluate(ds_test, verbose=2)
    # print("Loss evaluate : {:5.2f}%".format(loss))
    # print("Accuracy  evaluate : ", 100 * acc)
    
    print("Loss:", loss)
    print("Accuracy:", 100*sum(accuracy)/len(accuracy))
    print("False Positive Rate:", "{:.2f}".format(100*sum(false_positive_rate)/len(false_positive_rate)))
    print("False Negative Rate:", "{:.2f}".format(100*sum(false_negative_rate)/len(false_negative_rate)))
    print("Precision:", 100*precision)
    print("Recall:", "{:.2f}".format(100*sum(recall)/len(recall)))
    print("F1-Score:", "{:.2f}".format(100*sum(f1_score)/len(f1_score)))


def get_num_classes_from_model(model):
    # Get the last layer of the model
    last_layer = model.layers[-1]
    # If the last layer is a Dense layer, return the number of units
    if isinstance(last_layer, tf.keras.layers.Dense):
        return last_layer.units
    else:
        raise ValueError("Last layer is not a Dense layer")
    

def evaluate_with_metrics_python(model, num_captchas) : 
    extended_alphabet = string.ascii_letters + string.digits
    folder_path = os.path.join(os.getcwd(), 'generated_captcha_for_acc')
    
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    
    for i in range(5 * num_captchas) : 
        generate_captcha(4)
    captchas = os.listdir('generated_captcha_for_acc')

    true_labels = []
    predicted_labels = []
    images = []
    counter = 0
    j = 0
    present_characters = set()
    while counter < 62 or j < num_captchas:
        captcha = captchas[j]
        image = cv2.imread(f'generated_captcha_for_acc/{captcha}')
        characters = feature_extraction(image) 
        i = 0
        while i < len(characters) : 
            if captcha[i] not in present_characters:
                present_characters.add(captcha[i])
                counter += 1
            if characters[i].size == 0 : 
                break
            resized_image = cv2.resize(characters[i], (28, 28), interpolation=cv2.INTER_AREA)
            pixels = np.array(resized_image).reshape(28, 28, 1)
            images.append(pixels)
            true_labels.extend(chr_to_label_emnist(list(captcha[i])))
            i += 1
        j += 1
        
        
    images = convert_to_tfds(images)
    predicted_labels = model.predict(images)
    predictions = tf.nn.softmax(predicted_labels, axis=-1)
    predicted_labels = tf.argmax(predictions, axis=-1)
    evaluate_with_metrics(model,y_true=true_labels,y_pred=predicted_labels, predictions=predictions)


def generate_captcha(captcha_length, width=500, height=150):
    characters = string.ascii_letters + string.digits
    #characters = 'abcdefghijklmnpqrtuvwxyzABCDEFGHJKLMNOPQRTUVWXYZ2346789'

    captcha_text = ''.join(random.choice(characters) for _ in range(captcha_length))  # You can adjust the length as needed

    captcha = ImageCaptcha(
                           font_sizes=(100,100),
                           width=width, 
                           height=height, 
                          )

    folder_path = os.path.join(os.getcwd(), 'generated_captcha_for_acc')

    captcha_image_file = os.path.join(folder_path, f'{captcha_text}.jpeg')
    captcha.write(captcha_text, captcha_image_file, 'jpeg')
