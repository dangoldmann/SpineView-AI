import io
import requests
import torch
import base64
from base64 import b64encode, encodebytes
import json
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageFile

app = Flask(__name__)

apiUrl = 'http://localhost:3000'
#apiUrl = 'https://osia-api-production.up.railway.app'

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')

@app.route('/predict', methods=["POST"])
def predict():
    if request.method != "POST":
        return

    id_image = request.json['id']
    req = requests.get(apiUrl + '/radiographies/' + id_image)
    base64_data = req.content
    
    im = Image.open(io.BytesIO(base64_data))

    results = model(im)
    results.save()

    image = Image.fromarray(results.ims[0])
    ImageFile.MAXBLOCK = 2**20
    buf = io.BytesIO()
    image.save(buf, 'JPEG', quality=80, optimize=True, progressive=True)
   
    byte_im = buf.getvalue()
    byte_im = base64.encodebytes(byte_im).decode('ascii')
    
    return jsonify({
        'image_base64': byte_im
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)  # debug=True causes Restarting with stat
