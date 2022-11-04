import io
import os
import requests
import torch
import base64
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

cloudinaryApiUrl = 'https://res.cloudinary.com/dmxn0qho3/image/upload'

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt', force_reload=True)

@app.route('/predict', methods=["POST"])
def predict():
    image_path = request.json['imagePath']

    req = requests.get(cloudinaryApiUrl + image_path)
    base64_data = req.content
    
    im = Image.open(io.BytesIO(base64_data))

    results = model(im)
    results.save()

    image = Image.fromarray(results.ims[0])
    
    buf = io.BytesIO()
    image.save(buf, 'JPEG', quality=80, optimize=True, progressive=True)
   
    byte_im = buf.getvalue()
    byte_im = base64.encodebytes(byte_im).decode('ascii')
    
    return jsonify({
        'image_base64': byte_im
    })

if __name__ == "__main__":
    app.run(port=os.getenv("PORT", default=5000), debug=True)
