from captcha.image import ImageCaptcha
import random
import string
import os

def generate_captcha():
    # Set the captcha characters
    characters = string.ascii_letters + string.digits

    # Generate a random captcha text
    captcha_text = ''.join(random.choice(characters) for _ in range(6))  # You can adjust the length as needed

    # Create an ImageCaptcha object
    captcha = ImageCaptcha()

    # Generate the captcha image
    #captcha_image = captcha.generate(captcha_text)

    # Save the captcha image to a file
    folder_path = 'generated_captchas'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    captcha_image_file = os.path.join(folder_path, f'{captcha_text}.png')
    captcha.write(captcha_text, captcha_image_file)

    print(f'Captcha image saved to: {captcha_image_file}')

if __name__ == "__main__":
    for i in range(10) :    
        generate_captcha()
