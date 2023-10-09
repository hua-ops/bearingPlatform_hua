"""
Created on 2023/10/09 11:36

@Author: Sun Jiahua

@File  : utils.py

@Desc  : None
"""
import joblib
import hashlib
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.models import load_model


def generate_md5(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()

    md5hash = hashlib.md5(content)
    md5 = md5hash.hexdigest()
    return md5
