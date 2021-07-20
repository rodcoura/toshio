import requests
from PIL import Image
from pathlib import Path
import os
from app import to_base64, from_base64

path = Path(__file__).parent
image = Image.open(os.path.join(path, 'data/input/163da1d7-bf86-4bcd-86cd-71723890c25b.jpg'))
image_64 = to_base64(image, 'png')
img_width = 600
pyramid_size = 4 # de 2 a 5
layer = 3 #de 1 a 8

resp = requests.post('http://127.0.0.1:5000/',
                     json={"img": image_64, "img_width": img_width, "pyramid_size": pyramid_size, "layer": layer})

if resp.ok:
    img_str = resp.json()["image"]
    print("Great! A stylized image was created!")

    img = from_base64(img_str)
    img.save(os.path.join(path, 'data/out-images/test1.png'))
else:
    print(resp.status_code)