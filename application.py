from flask import Flask, request, Response, json
from tensorflow import keras
from keras.preprocessing import image
from io import BytesIO
from PIL import Image
import jsonpickle
import numpy as np
from apis import predict_image_class

application = Flask(__name__)


@application.route('/', methods=['POST', 'GET'])
def home_page():
    return 'Welcome To Plant Doctor Server'


@application.route('/process_image', methods=['POST', 'GET'])
def process_image():
    print("started")
    image_path = request.files['image']

    if (image_path.filename != ''):
        img_width, img_height = 224, 224
        img = Image.open(image_path)
        img = img.resize((224, 224))
        #img=misc.imread(image_path)
        #image_nparr=image.img_to_array(Image.open())
        image_nparr = image.img_to_array(img)
        image_nparr = np.expand_dims(image_nparr, axis=0)
        #image_path.save(image_path.filename)
    print(image_nparr)
    prediction = predict_image_class(image_nparr)
    print("prediction is ", prediction)
    print(type(image_nparr))
    res = {
        'plant_name': 'Rice',
        'disease_detected': prediction,
        'more_info': "NA"
    }
    return Response(json.dumps(res), status=200, mimetype='application/json')


if __name__ == '__main__':
    application.run(debug=True)