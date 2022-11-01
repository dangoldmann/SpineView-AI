import io
import requests
import torch
import base64
from flask import Flask, jsonify, request, send_file
from PIL import Image, ImageFile

app = Flask(__name__)

#apiUrl = 'http://localhost:3000'
apiUrl = 'https://osia-api-production.up.railway.app'

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
    results.show()

    image = Image.fromarray(results.ims[0])
    ImageFile.MAXBLOCK = 2**20
    image.save('output.jpg', 'JPEG', quality=80, optimize=True, progressive=True)
    image_jpg = 'output.jpg'
    
    image_base64 = base64.b64encode(image)
    print(image_base64)
    # return jsonify({
    #     'outputImage': image_jpg
    # })
    return send_file(image_jpg, mimetype='image/jpg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # debug=True causes Restarting with stat
