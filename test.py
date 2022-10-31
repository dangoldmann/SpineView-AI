# from flask import Flask, jsonify, request
# from flask_cors import CORS
# import numpy as np
# import requests
# from Threads import CustomThreadCnn, CustomThreadTransformers
# from PIL import Image
# import pydicom as PDCM
# import io
# from ImageConverter.imageConverter import Dicom_to_Image
# from dani_model import image_TBC_location
# import os
# from os.path import abspath
    
# app = Flask(__name__)
# CORS(app)

# #Just a test route
# @app.route('/', methods=['GET'])
# def index():
#     return jsonify("Hola")



# #Predict_jpg_route
# @app.route('/predict_jpg', methods=['POST'])
# def predict_jpg_img():
#     print(request.json)
#     #Recibo el json enviado por node que contiene la ruta de la imagen
#     imagefile = request.json
#     imagefile2 = request.json['path']
#     image_name = os.path.split(imagefile2)[-1]
#     print(image_name)

#     #Hago una petición post a una ruta del back-end que recibe una ruta de una imagen y devuelve la misma. 
#     #Por supuesto, en este caso estoy enviando la ruta de la imagen antes recibida
#     req = requests.post('http://localhost:4000/images/sendFile', json=imagefile)

#     #Recibo la imagen en base64
#     base64_data = req.content


#     #Convierto la información a bytes y la abro con pillow
#     image = Image.open(io.BytesIO(base64_data))

#     #Llamo la clase que me permite ejecutar la función del modelo de cnn que me permite predecir si la imagen tiene tuberculosis o no
#     #Además le paso como paraámetro la imagen
#     thread_cnn = CustomThreadCnn(image)

#     #Hago lo mismo que con la clase de cnn pero ahora con la de transformers
#     thread_transformers = CustomThreadTransformers(image)

#     #Ejercuto el thread de cnn
#     thread_cnn.start()

#     #Ejercuto el thread de trasnformers
#     thread_transformers.start()

#     #Cierro el Thread de cnn
#     thread_cnn.join()

#     #Cierro el Thread de trasnformers.
#     thread_transformers.join()

#     preds_cnn = thread_cnn.value
#     print(preds_cnn)
#     print(thread_cnn.name)
#     preds_transformers = thread_transformers.value
#     print(preds_transformers)
#     print(thread_transformers.name)

#     path = image_TBC_location(image, "image0.jpg")
#     print(path)

#     accuracy_cnn_raw = float(np.max(preds_cnn, axis=1)[0])
#     accuracy_cnn = str(round(accuracy_cnn_raw * 100 , 2))
#     accuracy_cnn = accuracy_cnn + ' %'

#     accuracy_transformers_raw = float(np.max(preds_transformers, axis=1)[0])
#     accuracy_transformers = str(round(accuracy_transformers_raw * 100 , 2))
#     accuracy_transformers = accuracy_transformers + ' %'

#     accuracy_average = (accuracy_cnn_raw + accuracy_transformers_raw)/2
#     accuracy_average = str(round(accuracy_average * 100 , 2))
#     accuracy_average = accuracy_average + ' %'


#     return jsonify({
#         'prediccion_cnn': accuracy_cnn,
#         'prediccion_transformers': accuracy_transformers,
#         'prediccion_promedio': accuracy_average,
#         'new_path': path
#     })


# @app.route('/predict_dicom', methods=['POST'])
# def predict_dicom_img():
#     print(request.json)
#     imagefile = request.json

#     req = requests.post('http://localhost:4000/images/sendFile', json=imagefile)
#     base64_data = req.content

#     dicom_encoding = PDCM.read_file(io.BytesIO(base64_data))
#     dicom_image_path = Dicom_to_Image(dicom_encoding)
#     image = Image.open(dicom_image_path, mode='r')

#     print(image)

#     files = {'file': open(dicom_image_path, 'rb')}
#     image_req = requests.post('http://localhost:4000/images/saveImageRoute', files=files)
#     print(image_req.json()['path'])
#     new_path = image_req.json()['path']

#     thread_cnn = CustomThreadCnn(image)
#     thread_transformers = CustomThreadTransformers(image)

#     thread_cnn.start()
#     thread_transformers.start()

#     thread_cnn.join()
#     thread_transformers.join()

#     preds_cnn = thread_cnn.value
#     preds_transformers = thread_transformers.value
#     print(preds_cnn)
#     print(thread_cnn.name)
#     print(preds_transformers)
#     print(thread_transformers.name)

#     accuracy_cnn_raw = float(np.max(preds_cnn, axis=1)[0])
#     accuracy_cnn = str(round(accuracy_cnn_raw * 100 , 2))
#     accuracy_cnn = accuracy_cnn + ' %'

#     accuracy_transformers_raw = float(np.max(preds_transformers, axis=1)[0])
#     accuracy_transformers = str(round(accuracy_transformers_raw * 100 , 2))
#     accuracy_transformers = accuracy_transformers + ' %'

#     accuracy_average = (accuracy_cnn_raw + accuracy_transformers_raw)/2
#     accuracy_average = str(round(accuracy_average * 100 , 2))
#     accuracy_average = accuracy_average + ' %'


#     return jsonify({
#         'prediccion_cnn': accuracy_cnn,
#         'prediccion_transformers': accuracy_transformers,
#         'prediccion_promedio': accuracy_average,
#         'new_path': new_path
#     })


# if __name__ == '__main__':
#     app.run(port=8000,debug=True)