from flask import Flask, request, jsonify, make_response, send_file
from PIL import Image
from io import BytesIO
import base64
from toshio import run_toshio
import skimage
from datetime import datetime

def from_base64(base64_str):
    plain_text = str(base64_str).split(",")[1]
    byte_data = base64.b64decode(plain_text)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img


def to_base64(img, format='JPEG'):
    im_file = BytesIO()
    img.save(im_file, format=format)
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes).decode()
    data_uri = 'data:image/{};base64,{}'.format(format.lower(), im_b64)
    return data_uri


app = Flask(__name__)


@app.route('/', methods=['GET'])
def style_transfer():
    args = request.args.to_dict()
    img_id = "eth"
    img_width = int(args['img_width'])
    pyramid_size = int(args['pyramid_size'])
    layers = [1, 2]
    peril_noise = int(args['peril_noise'])
    cmap = str(args['cmap'])
    cmap_r = int(args['cmap_r'])

    painting = run_toshio(img_id, img_width, pyramid_size, layers, cmap, cmap_r, peril_noise)
    painting = skimage.util.img_as_ubyte(painting)
    img = Image.fromarray(painting)
    out_base64 = to_base64(img, 'png')

    return jsonify({'image': str(out_base64)})


if __name__ == '__main__':
    app.run(host="0.0.0.0")
