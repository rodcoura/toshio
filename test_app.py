import requests
from pathlib import Path
import os
from app import from_base64

path = Path(__file__).parent
image_id = "cat"
img_width = 600
pyramid_size = 4 # de 2 a 5
layer = 5 #de 1 a 8

resp = requests.post('http://127.0.0.1:5000/',
                     json={"img_id": image_id, "img_width": img_width, "pyramid_size": pyramid_size, "layer": layer})

if resp.ok:
    img_str = resp.json()["image"]
    print("Great! A stylized image was created!")

    img = from_base64(img_str)
    img.save(os.path.join(path, 'data/out-images/test1.png'))
else:
    print(resp.status_code)