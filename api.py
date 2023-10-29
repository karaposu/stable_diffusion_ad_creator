from flask import Flask, request, send_from_directory, jsonify
from utils import decode_img, encode_img
from utils import encode_img as utils_encode_img

from make_stable_dif2 import make_image_dif, create_ad
from PIL import Image, ImageDraw, ImageFont
import numpy as np

app = Flask(__name__)




@app.route('/task2/', methods=['GET', 'POST'])
def task():
    print("girdi0")
    # return {"success": "true"}

    json = request.get_json()

    print("girdi1")


    operation = json["operation"]
    base_image = json["base_image"]
    logo_image = json["logo_image"]
    hex_code_diffusion = json["hex_code_diffusion"]
    hex_code_punchline = json["hex_code_punchline"]
    hex_code_button = json["hex_code_button"]
    punchline = json["punchline"]
    button_text = json["button_text"]

    base_img=decode_img(base_image)
    logo_img=decode_img(logo_image,  alpha=True)
    print("shape of decoded_logo_img:", logo_img.shape)
    logo_img = Image.fromarray(logo_img).convert('RGBA')


    sim_img=make_image_dif(base_img,hex_code_diffusion)[0]


    result=create_ad(sim_img, logo_img, punchline, hex_code_punchline, button_text, hex_code_button)

    result= np.array(result)
    print("result.shape:", result.shape)

    import cv2
    import base64

    # def encode_img(img):
    #     print("result.shape2:", img.shape)
    #     _, img_buffer = cv2.imencode('.png', img)
    #     encoded_img = base64.b64encode(img_buffer)
    #     return encoded_img.decode('utf-8')
    r=encode_img(result)



    package = {"success": 1,
            "result": r,
            }

    return package



if __name__ == '__main__':
    app.run(debug=True)