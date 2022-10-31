from email.mime import base
from pyexpat import model
from itsdangerous import base64_decode
import numpy as np
from flask import Flask, Response, request, jsonify, render_template
import pickle
import requests
from PIL import Image
import io

app = Flask(__name__)
#model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    id_image = request.json['id']
    req = requests.get('http://localhost:3000/radiographies/' + id_image)
    base64_data = req.content

    image = Image.open(io.BytesIO(base64_data))
    
    print(image)


    


if __name__ == '__main__':
    app.run(debug=True, port=5000)