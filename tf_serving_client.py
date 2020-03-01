import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

import requests
import json


image_path = '/home/segmind/Downloads/doggy.jpeg'

im = Image.open(image_path)
image = np.array(im).astype(tf.keras.backend.floatx())

payload = {
  "instances": [{'input_image': image.tolist()}]
}
r = requests.post('http://localhost:8501/v1/models/xception:predict', json=payload)
result = json.loads(r.content)['predictions']

print(np.argmax(result), np.max(result))