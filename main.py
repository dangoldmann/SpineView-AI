import io
import os
import requests
import torch
import gdown
import base64
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

# url = 'https://drive.google.com/uc?export=download&confirm=9iBg&id=195yn1JsrNMlzqTxwKXc3ywmC9LixSEm9'
# id = '195yn1JsrNMlzqTxwKXc3ywmC9LixSEm9'
# output = 'test.pt'
# gdown.download(url, output, quiet=False)

cloudinaryApiUrl = 'https://res.cloudinary.com/dmxn0qho3/image/upload'

@app.route('/', methods=["GET"])
def home():
    return 'Home'

@app.route('/predict', methods=["POST"])
def predict():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')
    image_path = request.json['imagePath']

    req = requests.get(cloudinaryApiUrl + image_path)
    base64_data = req.content
    
    im = Image.open(io.BytesIO(base64_data))

    results = model(im)
    results.save()

    image = Image.fromarray(results.ims[0])
    
    buf = io.BytesIO()
    image.save(buf, 'JPEG', quality=95, optimize=True, progressive=True)
   
    byte_im = buf.getvalue()
    byte_im = base64.encodebytes(byte_im).decode('ascii')
    
    return jsonify({
        'image_base64': byte_im
    })

if __name__ == "__main__":
    app.run(port=os.getenv("PORT", default=5000), debug=True)
