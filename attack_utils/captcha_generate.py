import sys
from captcha.image import ImageCaptcha
import random
import string
import os

def generate_captcha(number_characters=4, width=500, height=150):
    # Set the captcha characters
    characters = string.ascii_letters + string.digits

    # Generate a random captcha text
    captcha_text = ''.join(random.choice(characters) for _ in range(number_characters))  # You can adjust the length as needed

    # Create an ImageCaptcha object
    captcha = ImageCaptcha(fonts=['fonts/Roboto-Light.ttf'], 
                           font_sizes=(100,100),
                           width=width, 
                           height=height, 
                          )

    # Generate the captcha image
    #captcha_image = captcha.generate(captcha_text)

    # Save the captcha image to a file
    folder_path = 'generated_captchas'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    captcha_image_file = os.path.join(folder_path, f'{captcha_text}.jpeg')
    captcha.write(captcha_text, captcha_image_file, 'jpeg')

    print(f'Captcha image saved to: {captcha_image_file}')

if __name__ == "__main__":
    # Extract the captcha text from command line arguments
    try :
        captcha_length = int(sys.argv[1])
    except:
        captcha_length = None

    if captcha_length:
        generate_captcha(captcha_length)
    else:
        print("Captcha text not provided.")
