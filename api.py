from flask import Flask, request
from utils import decode_img, encode_img
from PIL import Image
import numpy as np
from create_ad import generate_ad_from_image_and_logo, make_stable_diffusion


app = Flask(__name__)

@app.route('/task2/', methods=['GET', 'POST'])
def task():

    json = request.get_json()

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

    sim_img=make_stable_diffusion(base_img,hex_code_diffusion)[0]
    result=generate_ad_from_image_and_logo(sim_img, logo_img, punchline, hex_code_punchline, button_text, hex_code_button)

    result= np.array(result)
    print("result.shape:", result.shape)

    r=encode_img(result)

    package = {"success": 1,
            "result": r,
            }

    return package



if __name__ == '__main__':
    app.run(debug=True)