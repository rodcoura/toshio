import requests
from pathlib import Path
import time
from app import from_base64
import os

path = Path(__file__).parent
image_id = 'eth'  # opção unica por enquanto
img_width = 512  # 512 ou 1024
pyramid_size = 5  # de 2 a 5
layer = [2, 5, 8]  # de 1 a 8 ou a combinação disso
cmap = 8  # de 1 a 9
cmap_r = 1  # 0 ou 1
peril_noise = 16  # 2, 4, 8 ou 16

start = time.time()
#'https://toshiopaintings.herokuapp.com/
resp = requests.post('http://127.0.0.1:5000/',
                     json={"img_id": image_id, "img_width": img_width, "pyramid_size": pyramid_size,
                           "layer": layer, "cmap": cmap, "cmap_r": cmap_r, "peril_noise": peril_noise})

if resp.ok:
    img_str = resp.json()["image"]
    print("Great! A stylized image was created!")

    img = from_base64(img_str)
    img.save(os.path.join(path, 'deepfinal_8_r.png'))
else:
    print(resp.status_code)
    print(f"p-{pyramid_size} - l-{layer} - pn-{peril_noise}")
end = time.time()

print(f"Time passed: {end-start}")



# import itertools
# layers = []
# stuff = [1, 2, 3, 4, 5, 6, 7, 8]
# for L in range(1, len(stuff)+1):
#     for subset in itertools.combinations(stuff, L):
#         layers.append(subset)
#
# path = Path(__file__).parent
# image_id = 'eth'
# img_width = 64 # 256 ou 512 caso queira pequeno
# cmap = 4 # de 1 a n
# cmap_r = True
#
#
# for pyramid_size in [2, 3, 4, 5]:
#     for layer in layers:
#         for peril_noise in [2, 4, 8, 16]:
#             start = time.time()
#             resp = requests.post('http://127.0.0.1:5000/',
#                                  json={"img_id": image_id, "img_width": img_width, "pyramid_size": pyramid_size,
#                                        "layer": layer, "cmap": cmap, "cmap_r": cmap_r, "peril_noise": peril_noise})
#
#             if resp.ok:
#                 img_str = resp.json()["image"]
#                 print("Great! A stylized image was created!")
#
#                 img = from_base64(img_str)
#                 img.save(os.path.join(path, 'deepfinal.png'))
#             else:
#                 print(resp.status_code)
#                 print(f"p-{pyramid_size} - l-{layer} - pn-{peril_noise}")
#             end = time.time()
#
#             print(f"Time passed: {end-start}")