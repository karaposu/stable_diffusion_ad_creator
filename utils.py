import base64
import cv2
import numpy as np
from PIL import Image, ImageFont



def adjust_font_size_for_text(text, max_width, start_font_size, font_path, draw):
    font_size = start_font_size
    font = ImageFont.truetype(font_path, font_size)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]

    while text_width > max_width:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]

    return font

def encode_img(img):
    _, img_buffer = cv2.imencode('.png', img)
    encoded_img = base64.b64encode(img_buffer)
    # return encoded_img
    return encoded_img.decode('utf-8')

def decode_img(encoded_img,mask=False, alpha=False):
    decoded_img = base64.b64decode(encoded_img)
    img_np_arr = np.frombuffer(decoded_img, np.uint8)
    if mask:
        img = cv2.imdecode(img_np_arr,cv2.IMREAD_UNCHANGED)
        if img is not None and len(img.shape) == 3 and img.shape[2] == 2:
            pass
        else:
            img= img[:,:,2]
    elif alpha:
        img = cv2.imdecode(img_np_arr, cv2.IMREAD_UNCHANGED)
    else:
         img = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
    return img

def load_img(path, COLORTRANSFORMATION):
    temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    temp = cv2.cvtColor(temp, COLORTRANSFORMATION)
    return temp


