import base64
import io
import random
import string

import cv2
import numpy as np
import urllib3
from botocore.exceptions import NoCredentialsError
from flask import request, jsonify
from PIL import Image, UnidentifiedImageError

from time import time

from collections import OrderedDict

import json


from flask import Flask, jsonify

def encode_img(img):
    _, img_buffer = cv2.imencode('.png', img)
    encoded_img = base64.b64encode(img_buffer)
   
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

def make_stable_diff_from_api():
    pass


