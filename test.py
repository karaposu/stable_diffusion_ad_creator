logo_path="/home/enes/lab/case_study/pepsi_logo.png"
base_image_path= "base_image_sample.jpg"
hex_code_diffusion = "#3A7BCD"
hex_code_button = "#34ebab"
hex_code_punchline = "#eb345f"
punchline = "AI add banners lead to higher conversion rates"
button_text = "Call to action text here!"



import requests
import base64
import cv2
import numpy as np


def encode_img(img):
    _, img_buffer = cv2.imencode('.png', img)
    encoded_img = base64.b64encode(img_buffer)
    # return encoded_img
    return encoded_img.decode('utf-8')

def decode_img(encoded_img):
    decoded_img = base64.b64decode(encoded_img)
    img_np_arr = np.frombuffer(decoded_img, np.uint8)
    img = cv2.imdecode(img_np_arr, cv2.IMREAD_UNCHANGED)
    return img


base_img = cv2.imread(base_image_path, cv2.IMREAD_UNCHANGED)
base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
base_img = cv2.resize(base_img, (256, 256))  # resize to 300x200

base_img=encode_img(base_img)

logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
logo = cv2.resize(logo, (100, 100))  # resize to 300x200
print("logo.size: ", logo.shape)
logo=encode_img(logo)

logo_decoded= decode_img(logo)

print("decoded logo size:", logo_decoded.shape)


data = {
    "operation": "task_ai",
    "base_image":base_img,
    "logo_image":logo,
    "hex_code_diffusion": hex_code_diffusion,
    "hex_code_punchline": hex_code_punchline,
    "hex_code_button": hex_code_button,
    "punchline": punchline,
    "button_text": button_text
}


response = requests.post("http://34.91.134.186/task2/", json=data)



if response.status_code == 200 and response.json().get('success') == 1:
    # Get the base64 encoded result
    encoded_image = response.json()["result"]

    # Decode the image
    decoded_image = base64.b64decode(encoded_image)

    # Save the image to your computer
    with open('output_image.png', 'wb') as f:
        f.write(decoded_image)

else:
    print("Request failed. Status code:", response.status_code)

    print("Response content:", response.text)

