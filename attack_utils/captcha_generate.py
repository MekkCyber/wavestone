import sys
from captcha.image import ImageCaptcha
import random
import string
import os
from PIL import Image

def generate_captcha(captcha_length, width=500, height=150):
    # Set the captcha characters
    characters = string.ascii_letters + string.digits

    # Generate a random captcha text
    captcha_text = ''.join(random.choice(characters) for _ in range(captcha_length))  # You can adjust the length as needed
    # Create an ImageCaptcha object
    captcha = ImageCaptcha(fonts=[os.getcwd() + '/attack_utils/fonts/Roboto-Light.ttf'], 
                           font_sizes=(100,100),
                           width=width, 
                           height=height, 
                          )

    # Generate the captcha image
    #captcha_image = captcha.generate(captcha_text)

    # Save the captcha image to a file
    folder_path = os.getcwd() + '/attack_utils/generated_captchas'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    captcha_image_file = os.path.join(folder_path, f'{captcha_text}.jpeg')
    captcha.write(captcha_text, captcha_image_file, 'jpeg')

    THRESHOLD = 220
    BLACK = (0, 0, 0)


    img = Image.open(captcha_image_file)
    img = img.convert("RGBA")
    width, height = img.size
    new_img = Image.new("RGB", (width, height), (0, 0, 0))

    pixels = img.load()
    
    # Iterate through each pixel
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            # Check if the pixel is close to white
            if r > THRESHOLD and g > THRESHOLD and b > THRESHOLD:
                # If close to white, set pixel color to dark_color
                new_img.putpixel((x, y), BLACK)
            else:
                # Otherwise, keep the original pixel color
                new_img.putpixel((x, y), (r, g, b, a))
    
    # Save the modified image
    new_img.save(captcha_image_file)
    

    print(f'{captcha_image_file}')

if __name__ == "__main__":
    # Extract the captcha text from command line arguments
    try :
        captcha_length = int(sys.argv[1])
    except:
        captcha_length = None

    if captcha_length != None:
        generate_captcha(captcha_length)
    else:
        print("Captcha text not provided.")
