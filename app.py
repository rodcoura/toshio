from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import base64
from toshio import Toshio
import skimage


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


@app.route('/', methods=['GET', 'POST'])
def style_transfer():
    if request.method == 'POST':
        json = request.json
        img_id = json['img_id']
        img_width = int(json['img_width'])
        pyramid_size = int(json['pyramid_size'])
        layer = int(json['layer'])

        painting = Toshio(img_id, img_width, pyramid_size, layer)
        painting = skimage.util.img_as_ubyte(painting)
        img = Image.fromarray(painting)
        out_base64 = to_base64(img, 'png')

        return jsonify({'image': str(out_base64)})
    return '''
           <!doctype html>
           <title>Make your own painting</title>
           '''


if __name__ == '__main__':
    app.run(host="0.0.0.0")
